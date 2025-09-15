import math
import os
import argparse
import yaml

import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field, Field, model_validator


class Config(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,              # 允许使用别名
        # extra="allow",                      # 允许额外的字段
    )

    ################## system params ##################
    nodes: int = 32  
    gpu_per_node: int = 8
    capacity: float = 1.0  # 1.0 for moe ep load balance

    @computed_field
    def gpus(self) -> int:
        return self.nodes * self.gpu_per_node

    ################## parallel params ##################
    ep: int = Field(1, alias="expert_parallel_size")
    pp: int = 1 
    vpp: int = 1  # interleaved pp
    tp: int = 1 
    TP_sp: bool = True 
    etp: int = 1  # expert tensor parallel size
    cp: int = 1
    ulysses: int = 1
    zero_stage: int = 3
    
    @computed_field
    def pp_flying_batches(self) -> float:
        """pp stage中最多驻留的micro batch数"""
        flying_batches = float(self.pp)
        if self.vpp > 1:
            # 参考论文 https://readpaper.com/pdf-annotate/note?pdfId=2742393776474998272&noteId=2870822463330744832 中 Table 1
            flying_batches *= (self.vpp + 1) / self.vpp
        return flying_batches
    
    @computed_field
    def sp(self) -> int:
        return self.cp * self.ulysses
    
    @computed_field
    def dp(self) -> int:
        return self.gpus // (self.sp * self.pp * self.tp)  # type: ignore
    
    @computed_field
    def edp(self) -> int:
        return self.gpus // (self.ep * self.pp * self.etp)  # type: ignore

    @model_validator(mode='after')
    def validate_parallel_params(self):
        if self.pp == 1 and self.vpp != 1:
            print(f"WARN: set vpp {self.vpp} to 1 when pp is 1")
            self.vpp = 1
        
        if self.ep == 1 and self.capacity != 1.0:
            print(f"WARN: set capacity {self.capacity} to 1.0 when ep is 1")
            self.capacity = 1.0
        
        assert self.TP_sp, "only support TP_sp=True"
        assert self.etp == 1, "only support etp=1 now"
        
        # 验证计算结果 (sp, dp, edp 现在是 computed_field)
        assert self.sp == 1, "only support sp=1 now"
        assert self.gpus % (self.sp * self.pp * self.tp) == 0, f"gpus:{self.gpus} must be divisible by sp:{self.sp} * pp:{self.pp} * tp:{self.tp}"  # type: ignore
        assert self.gpus % (self.ep * self.pp * self.etp) == 0, f"gpus:{self.gpus} must be divisible by ep:{self.ep} * pp:{self.pp} * etp:{self.etp}"  # type: ignore

        assert self.zero_stage in [1, 2, 3], "only support zero stage 1, 2, 3"
        return self

    ################## data type ##################
    param_type: int = 2  # 模型参数使用fp16，故参数量x2即为 Bytes
    grad_type: int = 2  # 梯度使用fp16，故参数量x2即为 Bytes
    act_type: int = 2  # 中间激活值使用bf16，故参数量x2即为 Bytes
    linear_act_type: int = 2  # 部分线性层激活值使用bf16，故参数量x2即为 Bytes. 1 for deepseek in case of fp8.
    os_type: int = 4  # optimizer state使用fp32，故参数量x4即为 Bytes. 2 for deepseek
    master_type: int = 4  # master参数分片使用fp32，故参数量x4即为 Bytes
    master_grad_type: int = 4  # master梯度分片使用fp32，故参数量x4即为 Bytes
    # 下面预留给通信量计算，目前不考虑
    # dp_type: int = 4  # 梯度聚合通信使用fp32，故参数量x4即为 Bytes
    # pp_type = tp_type = sp_type = ep_type = act_type # 中间激活值及其梯度传递使用bf16，故参数量x2即为 Bytes

    ################## train params ##################
    MBS: int = 1  # micro batch size
    L: int = 4096  # seq len
    MBN: int = 16  # micro batch num
    
    @computed_field
    def GBS(self) -> int:
        return self.MBN * self.dp * self.MBS  # type: ignore

    @model_validator(mode='after')
    def validate_train_params(self):
        # 调整MBN以优化PP bubble
        if self.pp > 1 and self.MBN < int(math.ceil(16 * self.pp / self.vpp)): 
            print(f"WARN: pp:{self.pp} > 1, set MBN to {int(math.ceil(16 * self.pp / self.vpp))} for low pp bubble = (pp-1) / (pp-1 + MBN*vpp)")
            self.MBN = int(math.ceil(16 * self.pp / self.vpp))
        
        return self

    ################## model params ##################
    # https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
    model: str = "Qwen3-235B-A22B"
    vocab_size: int = 151936
    D: int = Field(4096, alias="hidden_size")  # hidden size
    LN: int = Field(94, alias="num_hidden_layers")  # layer num
    LN_add_for_pp_first_stage: int = 0  # 为了pp负载均衡，第一个stage embedding需要多考虑几个layer, 1 for Qwen3-235B-A22B when pp>1
    LN_add_for_pp_last_stage: int = 0  # 为了pp负载均衡，最后一个stage loss/mtp等需要多考虑几个layer, 1 for Qwen3-235B-A22B when pp>1

    flash_attn: bool = True  # only work for calc_mem
    attn_type: str = "mha"  # "mha" or "mla"
    qk_norm: bool = True  # QKnorm
    qn: int = Field(64, alias="num_attention_heads")  # query head num
    kvn: int = Field(4, alias="num_key_value_heads")  # kv head num (GQA)
    # for mha
    H: int = Field(128, alias="head_dim")  # head dim
    # for mla
    qk_nope_head_dim: int = Field(128)  # qk nope head dim
    qk_rope_head_dim: int = Field(64)  # qk rope head dim
    v_head_dim: int = Field(128)  # v head dim
    q_lora_rank: int = Field(1536)  # q lora rank
    kv_lora_rank: int = Field(512)  # kv lora rank

    @computed_field
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    n_shared_experts: int = Field(0)  # moe shared experts num
    mH: int = Field(1536, alias="moe_intermediate_size")  # moe ffn hidden size
    topk: int = Field(8, alias="num_experts_per_tok")  # topk for moe
    mlp_act_dim: int = 2  # 2 for swiglu, 因为需要gate_proj和up_proj两个tensor
    e: int = Field(128, alias="num_experts")  # expert num
    topk_weights_position: str = "after_fc2"  # "before_fc2" or "after_fc2"
    chunk_loss_size: int = 0  # chunked loss size

    @computed_field
    def LN_per_gpu(self) -> int:
        return (self.LN + self.LN_add_for_pp_first_stage + self.LN_add_for_pp_last_stage) // self.pp  # layer number per gpu
    
    @computed_field(alias="query_hidden_dim")
    def qD(self) -> int:
        if self.attn_type == "mha":
            return self.qn * self.H
        else: #if self.attn_type == "mla":
            return self.qn * self.qk_head_dim
    
    @computed_field(alias="kv_hidden_dim")
    def kvD(self) -> int:
        assert self.attn_type == "mha", "only support mha for kvD"
        return self.kvn * self.H

    @computed_field(alias="k_hidden_dim")
    def kD(self) -> int:
        if self.attn_type == "mha":
            return self.kvn * self.H
        # mla
        return self.kvn * self.qk_head_dim
    
    @computed_field(alias="v_hidden_dim")
    def vD(self) -> int:
        if self.attn_type == "mha":
            return self.kvn * self.H
        # mla
        return self.kvn * self.v_head_dim
    
    @computed_field(alias="intermediate_size")
    def mD(self) -> int:
        return self.topk * self.mH
    
    @computed_field(alias="local_expert_num")
    def le(self) -> int:
        return self.e // self.ep  # local expert num per gpu

    @model_validator(mode='after')
    def validate_model_params(self):
        assert self.topk_weights_position in ["before_fc2", "after_fc2"], f"topk_weights_position:{self.topk_weights_position} not supported"
        assert self.qn % self.kvn == 0, f"qn:{self.qn} must be divisible by kvn:{self.kvn}"
        assert self.kvn % self.tp == 0, f"kvn:{self.kvn} must be divisible by tp:{self.tp}"

        if self.pp == 1:
            if self.LN_add_for_pp_first_stage != 0:
                print(f"WARN: set LN_add_for_pp_first_stage {self.LN_add_for_pp_first_stage} to 0 when pp is 1")
                self.LN_add_for_pp_first_stage = 0
            if self.LN_add_for_pp_last_stage != 0:
                print(f"WARN: set LN_add_for_pp_last_stage {self.LN_add_for_pp_last_stage} to 0 when pp is 1")
                self.LN_add_for_pp_last_stage = 0
            
        # LN/pp/vpp 需要整除
        LN2 = self.LN + self.LN_add_for_pp_first_stage + self.LN_add_for_pp_last_stage
        assert LN2 % self.pp == 0, f"LN:{self.LN} + LN_for_pp_first_stage:{self.LN_add_for_pp_first_stage} + LN_for_pp_last_stage:{self.LN_add_for_pp_last_stage} must be divisible by pp:{self.pp}"
        assert (LN2 // self.pp) % self.vpp == 0, f"LN2/pp:{LN2 // self.pp} must be divisible by vpp:{self.vpp}"
        if self.tp > 1:
            assert self.L % self.tp == 0, f"L:{self.L} must be divisible by tp:{self.tp}"
        return self

    ################## recompute params ##################
    recompute: bool = False  # only work for calc mem
    recompute_swiglu: bool = False  # for deepseek 
    recompute_norm: bool = False  # for deepseek 
    
    @computed_field
    def LN_recompute_per_gpu(self) -> int:
        # 如果开启recompute，则每个GPU上峰值显存只保留1个layer的激活值
        return 1 if self.recompute else self.LN_per_gpu  # type: ignore
    
    @model_validator(mode='after')
    def validate_recompute_params(self):
        if not self.recompute:
            return self
        
        if self.recompute_swiglu:
            print(f"WARN: set recompute_swiglu=False when recompute is True")
            self.recompute_swiglu = False
        
        if self.recompute_norm:
            print(f"WARN: set recompute_norm=False when recompute is True")
            self.recompute_norm = False

        return self


class Calculator:
    """
    显存由几部分组成：
    1. 模型参数
    2. 梯度
    3. 优化器状态
    4. 中间激活值

    举例说明各部分显存占用， 考虑zero1,不开启recompute，tp=1, ep=8, pp=2，并且node=2的情况:
    
    对于Attn层， dp=2*8/2=8
    Node0:  GPU0,  GPU1,  GPU2,  GPU3,  GPU4,  GPU5,  GPU6,  GPU7
          layer0 layer0 layer0 layer0 layer0 layer0 layer0 layer0
          os0s0   os0s1   os0s2   os0s3   os0s4   os0s5   os0s6   os0s7
    Node1:  GPU8,  GPU9,  GPU10, GPU11, GPU12, GPU13, GPU14, GPU15
          layer1 layer1 layer1 layer1 layer1 layer1 layer1 layer1 
          os1s0   os1s1   os1s2   os1s3   os1s4   os1s5   os1s6   os1s7
    PS: 由于开启zero1，os0s0表示layer0优化器状态的s0分片，os0s1表示layer0优化器状态的s1分片，因为dp=8所以有8个分片，以此类推
    
    对于MLP层， edp=2*8/2/8=1
    Node0:  GPU0,  GPU1,  GPU2,  GPU3,  GPU4,  GPU5,  GPU6,  GPU7
            l0e0   l0e1   l0e2   l0e3   l0e4   l0e5   l0e6   l0e7 
    Node1:  GPU8,  GPU9,  GPU10, GPU11, GPU12, GPU13, GPU14, GPU15
            l1e0   l1e1   l1e2   l1e3   l1e4   l1e5   l1e6   l1e7 
    PS: l0e0表示layer0的expert0， l1e7表示layer1的expert7
    
    参考： https://readpaper.com/pdf-annotate/note?pdfId=4622673296512524289&noteId=1750465888039326720 论文中4.1节
    """
    def __init__(self):
        self.embed_params_num = 0
        self.embed_params = 0
        self.embed_grads = 0
        self.embed_opt_states = 0
        self.embed_master_params = 0
        self.embed_master_grads = 0
        self.embed_acts = 0

        self.attn_params_num = 0
        self.attn_params_num_per_layer = 0
        self.perlayer_attn_params = 0
        self.perlayer_attn_grads = 0
        self.attn_opt_states = 0
        self.attn_master_params = 0
        self.attn_master_grads = 0
        self.perlayer_attn_master_grads = 0
        self.perlayer_attn_acts = 0

        self.perlayer_ckp_acts = 0

        self.mlp_params_num = 0
        self.mlp_params_num_per_layer = 0
        self.post_attn_ln_params_num_per_layer = 0
        self.perlayer_mlp_params = 0
        self.perlayer_mlp_grads = 0
        self.mlp_opt_states = 0
        self.mlp_master_params = 0
        self.mlp_master_grads = 0
        self.perlayer_mlp_master_grads = 0
        self.perlayer_mlp_acts = 0

        self.head_params_num = 0
        self.head_params = 0
        self.head_grads = 0
        self.head_opt_states = 0
        self.head_master_params = 0
        self.head_master_grads = 0
        self.head_acts = 0

        assert C.zero_stage in [1, 3], "only support zero stage 1, 3"
        self.param_denom = self.grad_denom = self.os_denom = 1
        if C.zero_stage == 1:
            self.param_denom = self.mlp_param_denom = 1
            self.grad_denom = self.mlp_grad_denom = 1
            self.os_denom = C.dp
            self.mlp_os_denom = C.edp
        # elif C.zero_stage == 2:
        #     self.param_denom = self.mlp_param_denom = 1
        #     self.grad_denom = C.dp
        #     self.mlp_grad_denom = C.edp
        #     self.os_denom = C.dp
        #     self.mlp_os_denom = C.edp
        else:  # C.zero_stage == 3
            # xtuner实现逻辑，只在 (fsdp, ep) 或者 (fsdp, tp) 的fsdp维度上切
            if C.ep > 1:
                fsdp = C.edp
            else:  # 不开ep时，取决于tp
                fsdp = C.dp
            self.param_denom = fsdp
            self.grad_denom = fsdp
            self.os_denom = fsdp
            self.mlp_param_denom = fsdp
            self.mlp_grad_denom = fsdp
            self.mlp_os_denom = fsdp

    def run(self, output_path):
        self.calc_part_info()

        self.aggregate(output_path)
        # self.show()
    
    def calc_part_info(self):
        self.embed_tokens()

        # attn
        self.input_layernorm()
        self.self_attn()
        self.attn_residual()

        # mlp
        self.post_attention_layernorm()
        self.mlp()
        self.mlp_residual()

        # head: final layer norm, lm_head, softmax
        self.head()

    def aggregate(self, output_path):
        # calc total params num
        self.params_num = self.embed_params_num + self.attn_params_num + self.mlp_params_num + self.head_params_num
        print(f"Total_params_num: {self.params_num / 1024**3} B, embed_params_num: {self.embed_params_num / 1024**3} B, attn_params_num: {self.attn_params_num / 1024**3} B, " 
              f"mlp_params_num: {self.mlp_params_num / 1024**3} B, head_params_num: {self.head_params_num / 1024**3} B")

        # calc per gpu memory
        if C.zero_stage == 1:
            self._zero1_aggregate(output_path)
        else:  # C.zero_stage == 2 or C.zero_stage == 3
            self._zero3_aggregate()

        # 控制台柱状图（占比）
        # # Memory category breakdown
        # parts = {
        #     "params": self.embed_params + self.attn_params + self.mlp_params + self.head_params,
        #     "grads": self.embed_grads + self.attn_grads + self.mlp_grads + self.head_grads,
        #     "opt_states": self.embed_opt_states + self.attn_opt_states + self.mlp_opt_states + self.head_opt_states,
        #     "master_params": self.embed_master_params + self.attn_master_params + self.mlp_master_params + self.head_master_params,
        #     "master_grads": self.embed_master_grads + self.attn_master_grads + self.mlp_master_grads + self.head_master_grads,
        #     "acts": self.embed_acts + self.attn_acts * C.pp_flying_batches + self.mlp_acts * C.pp_flying_batches + self.head_acts,  # type: ignore
        # }
        # self._print_bars("Memory category breakdown", parts, total_mem)

        # # Attn memory breakdown
        # attn_parts = {
        #     "params": self.attn_params,
        #     "grads": self.attn_grads,
        #     "opt_states": self.attn_opt_states,
        #     "master_params": self.attn_master_params,
        #     "master_grads": self.attn_master_grads,
        #     "acts (x flying)": self.attn_acts * C.pp_flying_batches,  # type: ignore
        # }
        # self._print_bars("Attn memory breakdown", attn_parts, attn_mem)

        # # MLP memory breakdown
        # mlp_parts = {
        #     "params": self.mlp_params,
        #     "grads": self.mlp_grads,
        #     "opt_states": self.mlp_opt_states,
        #     "master_params": self.mlp_master_params,
        #     "master_grads": self.mlp_master_grads,
        #     "acts (x flying)": self.mlp_acts * C.pp_flying_batches,  # type: ignore
        # }
        # self._print_bars("MLP memory breakdown", mlp_parts, mlp_mem)
    
    def _zero3_aggregate(self):
        ########## 第1个micro batch前向最后一个layer时的显存 ##########
        # 静态显存: master参数，优化器状态
        static_mem = self.embed_master_params + self.embed_opt_states + self.head_master_params + self.head_opt_states
        static_mem += self.attn_master_params + self.attn_opt_states + self.mlp_master_params + self.mlp_opt_states
        # 最后一个layer和head的全部参数和激活
        forward_last = 0
        perlayer_params = self.perlayer_attn_params + self.perlayer_mlp_params
        perlayer_acts = self.perlayer_attn_acts + self.perlayer_mlp_acts
        forward_last += perlayer_params + self.head_params
        forward_last += perlayer_acts + self.head_acts
        if C.recompute:
            # 全部剩余layer激活值的checkpoint
            # todo: 是否加上 embed_acts? 即 embed_layer是否重计算 ?
            forward_last += self.perlayer_ckp_acts * (C.LN_per_gpu - 1)  # type: ignore
        else:
            forward_last += self.embed_acts + perlayer_acts * (C.LN_per_gpu - 1)  # type: ignore
        micro1_forward_last = static_mem + forward_last

        ########## 第1个micro batch反向最后时(第1个layer)的显存 ##########
        # 静态显存: 增加master梯度
        static_grad_mem = self.attn_master_grads + self.mlp_master_grads + self.embed_master_grads + self.head_master_grads
        micro1_backward_last = static_mem + static_grad_mem
        # 第一个layer和embed的全部参数和激活
        micro1_backward_last += perlayer_params + self.embed_params
        micro1_backward_last += perlayer_acts + self.embed_acts
        # 上一个layer正在Reduce Scatter的梯度
        perlayer_grads = self.perlayer_attn_grads + self.perlayer_mlp_grads
        micro1_backward_last += perlayer_grads

        ########## 第2个micro batch前向最后一个layer时的显存 ##########
        # 静态显存: 已经增加了master梯度
        micro2_forward_last = static_mem + static_grad_mem + forward_last

        ####### show memory ##########
        total_ckp = self.perlayer_ckp_acts * C.LN_per_gpu  # type: ignore
        print(f"static_mem: {static_mem / 1024**3} GiB, total_ckp: {total_ckp / 1024**3} GiB, static_grad_mem: {static_grad_mem / 1024**3} GiB")
        print(f"embed_params: {self.embed_params / 1024**3} GiB, embed_grads: {self.embed_grads / 1024**3} GiB, embed_opt_states: {self.embed_opt_states / 1024**3} GiB, embed_master_params: {self.embed_master_params / 1024**3} GiB, embed_master_grads: {self.embed_master_grads / 1024**3} GiB, embed_acts: {self.embed_acts / 1024**3} GiB")
        print(f"head_params: {self.head_params / 1024**3} GiB, head_acts: {self.head_acts / 1024**3} GiB")
        print(f"perlayer_params: {perlayer_params / 1024**3} GiB, perlayer_acts: {perlayer_acts / 1024**3} GiB, perlayer_grads: {perlayer_grads / 1024**3} GiB")
        print(f"per_layer_attn_params: {self.perlayer_attn_params / 1024**3} GiB, per_layer_attn_acts: {self.perlayer_attn_acts / 1024**3} GiB, per_layer_attn_grads: {self.perlayer_attn_grads / 1024**3} GiB")
        print(f"per_layer_mlp_params: {self.perlayer_mlp_params / 1024**3} GiB, per_layer_mlp_acts: {self.perlayer_mlp_acts / 1024**3} GiB, per_layer_mlp_grads: {self.perlayer_mlp_grads / 1024**3} GiB")
        print(f"per_layer_attn_master_grads: {self.perlayer_attn_master_grads / 1024**3} GiB, per_layer_mlp_master_grads: {self.perlayer_mlp_master_grads / 1024**3} GiB")
        print(f"head_params: {self.head_params / 1024**3} GiB, head_acts: {self.head_acts / 1024**3} GiB, head_grads: {self.head_grads / 1024**3} GiB, head_opt_states: {self.head_opt_states / 1024**3} GiB, head_master_params: {self.head_master_params / 1024**3} GiB, head_master_grads: {self.head_master_grads / 1024**3} GiB")
        max_mem1 = max(micro1_forward_last, micro1_backward_last)
        print(f"Max_mem for microbatch=1: {max_mem1 / 1024**3} GiB. micro1_forward_last: {micro1_forward_last / 1024**3} GiB, micro1_backward_last: {micro1_backward_last / 1024**3} GiB")
        max_mem = max(micro1_forward_last, micro1_backward_last, micro2_forward_last)
        print(f"Max_mem for microbatch>1: {max_mem / 1024**3} GiB. micro1_forward_last: {micro1_forward_last / 1024**3} GiB, micro1_backward_last: {micro1_backward_last / 1024**3} GiB, micro2_forward_last: {micro2_forward_last / 1024**3} GiB")

    def _zero1_aggregate(self, output_path):
        if C.pp > 1:
            # todo: 目前只计算了第1个stage的显存占用，后续考虑计算每个stage的显存占用，然后取最大值
            # 同时将embed的act和grad置零，因为峰值一般出现在第一个stage，并且 LN_for_pp_first_stage已经考虑多考虑了1个layer来替代
            self.embed_params = self.embed_grads = self.embed_opt_states = self.embed_master_params = self.embed_master_grads = self.embed_acts = 0
            # deprecated: 如果开启pp将head相关置零，因为峰值出现在第一个stage，所以只考虑embed，不考虑head部分
            self.head_params = self.head_grads = self.head_opt_states = self.head_master_params = self.head_master_grads = self.head_acts = 0
        embed_mem = self.embed_params + self.embed_grads + self.embed_opt_states + self.embed_master_params + self.embed_master_grads + self.embed_acts  # type: ignore
        head_mem = self.head_params + self.head_grads + self.head_opt_states + self.head_master_params + self.head_master_grads + self.head_acts  # type: ignore

        # LN_per_gpu = C.LN_per_gpu - C.LN_add_for_pp_first_stage
        LN_per_gpu = C.LN_per_gpu  # type: ignore
        attn_params = self.perlayer_attn_params * LN_per_gpu  # type: ignore
        attn_grads = self.perlayer_attn_grads * LN_per_gpu  # type: ignore
        if not C.recompute:
            attn_acts = self.perlayer_attn_acts * LN_per_gpu  # type: ignore
        else:
            attn_acts = self.perlayer_attn_acts + self.perlayer_ckp_acts * (LN_per_gpu - 1)  # type: ignore
        attn_mem = attn_params + attn_grads + self.attn_opt_states + self.attn_master_params + self.attn_master_grads + attn_acts * C.pp_flying_batches  # type: ignore
        print(f"Attn_mem: {attn_mem / 1024**3} GiB. attn_params: {attn_params / 1024**3} GiB, attn_grads: {attn_grads / 1024**3} GiB, attn_opt_states: {self.attn_opt_states / 1024**3} GiB, attn_master_params: {self.attn_master_params / 1024**3} GiB, attn_master_grads: {self.attn_master_grads / 1024**3} GiB, attn_acts: {attn_acts / 1024**3} GiB, flying_attn_acts: {attn_acts * C.pp_flying_batches / 1024**3} GiB")

        mlp_params = self.perlayer_mlp_params * LN_per_gpu  # type: ignore
        mlp_grads = self.perlayer_mlp_grads * LN_per_gpu  # type: ignore
        if not C.recompute:
            mlp_acts = self.perlayer_mlp_acts * LN_per_gpu  # type: ignore
        else:
            mlp_acts = self.perlayer_mlp_acts  # type: ignore
        mlp_mem = mlp_params + mlp_grads + self.mlp_opt_states + self.mlp_master_params + self.mlp_master_grads + mlp_acts * C.pp_flying_batches  # type: ignore
        print(f"Mlp_mem: {mlp_mem / 1024**3} GiB. mlp_params: {mlp_params / 1024**3} GiB, mlp_grads: {mlp_grads / 1024**3} GiB, mlp_opt_states: {self.mlp_opt_states / 1024**3} GiB, mlp_master_params: {self.mlp_master_params / 1024**3} GiB, mlp_master_grads: {self.mlp_master_grads / 1024**3} GiB, mlp_acts: {mlp_acts / 1024**3} GiB, flying_mlp_acts: {mlp_acts * C.pp_flying_batches / 1024**3} GiB")  # type: ignore

        total_mem = embed_mem + attn_mem + mlp_mem + head_mem
        print(f"Total_mem: {total_mem / 1024**3} GiB. embed_mem: {embed_mem / 1024**3} GiB, attn_mem: {attn_mem / 1024**3} GiB, mlp_mem: {mlp_mem / 1024**3} GiB, head_mem: {head_mem / 1024**3} GiB")

        # 生成一个2维dataframe来表示单卡上各个模块的显存占用
        # 列为显存类型，分别为: act, param, grad, os, master_param, master_grad, total_row
        # 行为模块，分别为: embed, attn, mlp, head, total_col
        # 举例说明每个元素含义： 
        #  attn行 param 列的值为 attn模块参数量在单卡上的显存占用，单位 GiB，即 attn_params / 1024**3
        #  mlp行 total_row列的值为 mlp模块在单卡上的所有显存占用，相当于对attn行所有列的求和，单位 GiB，即 mlp_mem / 1024**3
        #  total_col行 act 列的值为 所有模块的中间激活值，相当于对act列所有行的求和，单位 GiB，即 (embed_acts + attn_acts + mlp_acts + head_acts) / 1024**3
        #  total_col行 total_row列的值为所有模块在单卡上的所有显存占用，相当于对所有行所有列的求和，单位 GiB，即 total_mem / 1024**3
        df = pd.DataFrame(index=["embed", "attn", "mlp", "head", "total_col"], columns=["act", "param", "grad", "os", "master_param", "master_grad", "total_row"])
        df.loc["embed", "act"] = self.embed_acts / 1024**3
        df.loc["embed", "param"] = self.embed_params / 1024**3
        df.loc["embed", "grad"] = self.embed_grads / 1024**3
        df.loc["embed", "os"] = self.embed_opt_states / 1024**3
        df.loc["embed", "master_param"] = self.embed_master_params / 1024**3
        df.loc["embed", "master_grad"] = self.embed_master_grads / 1024**3
        df.loc["embed", "total_row"] = embed_mem / 1024**3

        df.loc["attn", "act"] = attn_acts * C.pp_flying_batches / 1024**3
        df.loc["attn", "param"] = attn_params / 1024**3
        df.loc["attn", "grad"] = attn_grads / 1024**3
        df.loc["attn", "os"] = self.attn_opt_states / 1024**3
        df.loc["attn", "master_param"] = self.attn_master_params / 1024**3
        df.loc["attn", "master_grad"] = self.attn_master_grads / 1024**3
        df.loc["attn", "total_row"] = attn_mem / 1024**3

        df.loc["mlp", "act"] = mlp_acts * C.pp_flying_batches / 1024**3
        df.loc["mlp", "param"] = mlp_params / 1024**3
        df.loc["mlp", "grad"] = mlp_grads / 1024**3
        df.loc["mlp", "os"] = self.mlp_opt_states / 1024**3
        df.loc["mlp", "master_param"] = self.mlp_master_params / 1024**3
        df.loc["mlp", "master_grad"] = self.mlp_master_grads / 1024**3
        df.loc["mlp", "total_row"] = mlp_mem / 1024**3

        df.loc["head", "act"] = self.head_acts / 1024**3
        df.loc["head", "param"] = self.head_params / 1024**3
        df.loc["head", "grad"] = self.head_grads / 1024**3
        df.loc["head", "os"] = self.head_opt_states / 1024**3
        df.loc["head", "master_param"] = self.head_master_params / 1024**3
        df.loc["head", "master_grad"] = self.head_master_grads / 1024**3
        df.loc["head", "total_row"] = head_mem / 1024**3

        df.loc["total_col", "act"] = (self.embed_acts + attn_acts * C.pp_flying_batches + mlp_acts * C.pp_flying_batches + self.head_acts) / 1024**3
        df.loc["total_col", "param"] = (self.embed_params + attn_params + mlp_params + self.head_params) / 1024**3
        df.loc["total_col", "grad"] = (self.embed_grads + attn_grads + mlp_grads + self.head_grads) / 1024**3
        df.loc["total_col", "os"] = (self.embed_opt_states + self.attn_opt_states + self.mlp_opt_states + self.head_opt_states) / 1024**3
        df.loc["total_col", "master_param"] = (self.embed_master_params + self.attn_master_params + self.mlp_master_params + self.head_master_params) / 1024**3
        df.loc["total_col", "master_grad"] = (self.embed_master_grads + self.attn_master_grads + self.mlp_master_grads + self.head_master_grads) / 1024**3
        df.loc["total_col", "total_row"] = total_mem / 1024**3
        print(df)
        df.to_excel(output_path)
    
    def embed_tokens(self):
        # 参数量
        self.embed_params_num = C.vocab_size * C.D / C.tp
        # 1. 模型参数
        self.embed_params = self.embed_params_num * C.param_type  # type: ignore

        # 2. 本地梯度
        self.embed_grads = self.embed_params_num * C.grad_type  # type: ignore

        # 3. 优化器状态
        self.embed_opt_states = 2 * self.embed_params_num * C.os_type / self.os_denom  # type: ignore
        self.embed_master_params = self.embed_params_num * C.master_type / self.os_denom  # type: ignore
        self.embed_master_grads = self.embed_params_num * C.master_grad_type / self.os_denom  # type: ignore

        # 4. 中间激活值：[MBS, L]
        # todo: recompute 时是否保留？
        self.embed_acts = C.MBS * C.L / C.tp * C.act_type  # type: ignore
    
    def head(self):
        # 参数量
        self.head_params_num += C.D  # final layer norm
        self.head_params_num += C.D * C.vocab_size # lm_head

        # 1. 模型参数
        self.head_params = self.head_params_num * C.param_type  # type: ignore
        # 2. 本地梯度
        self.head_grads = self.head_params_num * C.grad_type  # type: ignore
        # 3. 优化器状态及因为混合精度训练而维护的master参数和梯度分片
        self.head_opt_states = 2 * self.head_params_num * C.os_type / self.os_denom  # type: ignore
        self.head_master_params = self.head_params_num * C.master_type / self.os_denom  # type: ignore
        self.head_master_grads = self.head_params_num * C.master_grad_type / self.os_denom  # type: ignore
        # 4. 中间激活值：
        # todo: 需要考虑recompute?
        # todo: 需要考虑chunked loss?
        self.head_acts = 0
        # final layer norm的输入: [L, MBS, D]
        self.head_acts += C.L / C.tp * C.MBS * C.D * C.act_type  # type: ignore
        # lm_head的输入: [L, MBS, D]
        self.head_acts += C.L / C.tp * C.MBS * C.D * C.act_type  # type: ignore
        # softmax的反向需要输入和输出: 2个[L, MBS, vocab_size]
        vocab_size = C.vocab_size if C.chunk_loss_size <= 0 else C.chunk_loss_size
        self.head_acts += 2 * C.L / C.tp * C.MBS * vocab_size * C.act_type  # type: ignore
    
    def input_layernorm(self):
        # 参数量
        self.attn_params_num_per_layer += C.D

        # 中间激活值：主要考虑反向计算时需要用到的中间结果, 并且假设输出的梯度已经存在，不需要计算空间。
        #    这是因为: 1)如果某些中间结果反向计算时不需要，那么在前向计算时，这些中间结果的显存占用可以被释放。
        #             2)输出梯度在计算完梯度后，可以被释放。
        # 如果开启recompute，则每个layer只保留这1个输入的hidden_states
        self.perlayer_attn_acts += C.L / C.tp * C.MBS * C.D * C.act_type  # type: ignore

        self.perlayer_ckp_acts = C.L / C.tp * C.MBS * C.D * C.act_type  # type: ignore
    
    def self_attn(self):
        # 模型参数量
        if C.attn_type == "mha":
            # for mha:Q_W=D*qD, K_W=D*kvD, V_W=D*kvD, O_W=D*qD
            self.attn_params_num_per_layer += (C.D * C.qD + C.D * C.kvD + C.D * C.kvD + C.D * C.qD) / C.tp  # type: ignore
        else: #if C.attn_type == "mla":
            # for mla: D for down, U for up, R for rope
            # W_DQ=D*q_lora_rank, W_UQ=q_lora_rank*qn*qk_nope_head_dim, W_QR=q_lora_rank*qn*qk_rope_head_dim, 注意 W_QR有qn个，是多头的
            self.attn_params_num_per_layer += (C.D * C.q_lora_rank + C.q_lora_rank * C.qn * C.qk_nope_head_dim + C.q_lora_rank * C.qn * C.qk_rope_head_dim) / C.tp  # type: ignore
            # W_DKV=D*kv_lora_rank
            self.attn_params_num_per_layer += C.D * C.kv_lora_rank / C.tp  # type: ignore
            # W_UK=kv_lora_rank*kvn*qk_nope_head_dim, W_KR=D*qk_rope_head_dim  注意 W_KR只有1个，不是多头的
            self.attn_params_num_per_layer += (C.kv_lora_rank * C.kvn * C.qk_nope_head_dim + C.D * C.qk_rope_head_dim) / C.tp  # type: ignore
            # W_UV=kv_lora_rank*kvn*v_head_dim
            self.attn_params_num_per_layer += (C.kv_lora_rank * C.kvn * C.v_head_dim) / C.tp  # type: ignore
            # W_O=vD*D=kvn*v_head_dim*D
            self.attn_params_num_per_layer += C.kvn * C.v_head_dim * C.D / C.tp  # type: ignore

        self.attn_params_num = self.attn_params_num_per_layer * C.LN
        # 1. 模型参数
        self.perlayer_attn_params += self.attn_params_num_per_layer * C.param_type
        # 2. 本地梯度
        self.perlayer_attn_grads += self.attn_params_num_per_layer * C.grad_type
        # 3. 优化器状态及因为混合精度训练而维护的master参数和梯度分片: Q_S=D*qD, K_S=D*kvD, V_S=D*kvD, O_S=D*qD
        self.attn_opt_states += 2 * self.attn_params_num_per_layer * C.LN_per_gpu * C.os_type / self.os_denom
        self.attn_master_params += self.attn_params_num_per_layer * C.LN_per_gpu * C.master_type / self.os_denom
        self.perlayer_attn_master_grads = self.attn_params_num_per_layer* C.master_grad_type / self.os_denom
        self.attn_master_grads += self.attn_params_num_per_layer * C.LN_per_gpu * C.master_grad_type / self.os_denom

        # 4. 激活值
        # QKV Linear的共同输入tensor: hidden_states [L,MBS,D]
        if C.recompute_norm:
            pass
        else:
            self.perlayer_attn_acts += C.L / C.tp * C.MBS * C.D * C.linear_act_type  # type: ignore    

        if C.attn_type == "mha":
            if C.qk_norm:
                # QKnorm的输入tensor: q [L, MBS, qn, H] 和 k [L, MBS, kvn, H], 其中 qn*H=qD, kvn*H=kvD
                self.perlayer_attn_acts += C.L * C.MBS * (C.qD + C.kvD) / C.tp * C.act_type  # type: ignore
            # rope的QK输入tensor: norm后或未norm的 q 和 k
            self.perlayer_attn_acts += C.L * C.MBS * (C.qD + C.kvD) / C.tp * C.act_type  # type: ignore
            # core attn的计算
            # Q@K^T的输入: q 和 k
            self.perlayer_attn_acts += C.L * C.MBS * (C.qD + C.kvD) / C.tp * C.act_type  # type: ignore
        else: #if C.attn_type == "mla":
            # q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(query_shape).transpose(1, 2)
            # q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # q_a_layernorm的输入: q_a_proj的输出 [L, MBS, q_lora_rank]
            self.perlayer_attn_acts += C.L / C.tp * C.MBS * C.q_lora_rank * C.act_type  # type: ignore
            # q_b_proj的输入： q_a_layernorm输出 
            self.perlayer_attn_acts += C.L / C.tp * C.MBS * C.q_lora_rank * C.act_type  # type: ignore

            # compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [L, MBS, kv_lora_rank + qk_rope_head_dim]
            # k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            # k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
            # k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # kv_a_layernorm 的输入：kv_a_proj_with_mqa 的输出 [L, MBS, kv_lora_rank + qk_rope_head_dim]
            self.perlayer_attn_acts += C.L / C.tp * C.MBS * C.kv_lora_rank * C.act_type  # type: ignore
            # kv_b_proj 的输入：kv_a_layernorm输出  [L, MBS, kv_lora_rank + qk_rope_head_dim]
            self.perlayer_attn_acts += C.L / C.tp * C.MBS * C.kv_lora_rank * C.act_type  # type: ignore

            # q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            # rope的输入： q_rot 为 [L, MBS, qn, qk_rope_head_dim], q_rot是多头的，有 C.qn 个
            self.perlayer_attn_acts += C.L / C.tp * C.MBS * C.qn * C.qk_rope_head_dim * C.act_type  # type: ignore
            # rope的输入： k_rot 为 [L, MBS, qk_rope_head_dim] ，注意k_rot只有1个，不是多头的
            self.perlayer_attn_acts += C.L / C.tp * C.MBS * C.qk_rope_head_dim * C.act_type  # type: ignore

            # Q@K^T的输入: q [L, MBS, qn, qk_nope_head_dim+qk_rope_head_dim] 和 k [L, MBS, kvn, qk_nope_head_dim+qk_rope_head_dim)]
            self.perlayer_attn_acts += C.L / C.tp * C.MBS * (C.qn * (C.qk_nope_head_dim + C.qk_rope_head_dim) + C.kvn * (C.qk_nope_head_dim + C.qk_rope_head_dim)) * C.act_type  # type: ignore

        if not C.flash_attn:
            # softmax的反向需要它自己的输出 softmax_out [MBS, qn, L, L]
            self.perlayer_attn_acts += C.MBS * C.qn / C.tp * C.L * C.L * C.act_type  # type: ignore
        else:
            # 如果开启flash attn v2，则将上面O(L^2)的空间优化到 O(L)。具体地，需要存储 logsumexp= [MBS, qn, L]
            # 参考 flash attn v1 https://zhuanlan.zhihu.com/p/669926191 的6.3节，以及 flash attn v2 https://zhuanlan.zhihu.com/p/691067658 的1.2 节
            self.perlayer_attn_acts += C.MBS * C.qn / C.tp * C.L * C.act_type  # type: ignore

        # attn over values需要softmax的输出（上面已经计算，不需要重复计算），以及V输入
        if C.attn_type == "mha":
            # v [L, MBS, kvn, H], 其中 kvn*H=kvD
            self.perlayer_attn_acts += C.MBS * C.L * C.kvD / C.tp * C.act_type  # type: ignore
        else: #if C.attn_type == "mla":
            # v [L, MBS, kvn, v_head_dim]
            self.perlayer_attn_acts += C.MBS * C.L * C.kvn * C.v_head_dim / C.tp * C.act_type  # type: ignore

        # Output Linear的输入: hidden_states [L, MBS, qn, H] = [L, MBS, qD]
        self.perlayer_attn_acts += C.MBS * C.L/C.tp * C.qD * C.act_type  # type: ignore
    
    def attn_residual(self):
        # residual反向时不需要保存激活值
        pass

    def post_attention_layernorm(self):
        # 参数量
        self.post_attn_ln_params_num_per_layer = C.D

        # 反向需要输入的激活值
        self.perlayer_mlp_acts += C.L / C.tp * C.MBS * C.D * C.act_type  # type: ignore

    def mlp(self):
        # 模型参数量: fc1_W=D*mH*le*mlp_act_dim, fc2_W=mH*D*le, 
        fc_params_num_per_layer = C.D * C.mH * C.le * C.mlp_act_dim + C.mH * C.D * C.le  # type: ignore

        # shared experts params: fc1=D*mH*n_shared_experts*mlp_act_dim, fc2=mH*D*n_shared_experts
        shared_params_num_per_layer = C.D * C.mH * C.n_shared_experts * C.mlp_act_dim + C.mH * C.D * C.n_shared_experts
        # gate_W=D*e
        router_params_num_per_layer = C.D * C.e  # type: ignore
        nonfc_params_num_per_layer = router_params_num_per_layer + self.post_attn_ln_params_num_per_layer + shared_params_num_per_layer

        self.mlp_params_num_per_layer = fc_params_num_per_layer + nonfc_params_num_per_layer
        self.mlp_params_num = (fc_params_num_per_layer * C.ep + nonfc_params_num_per_layer) * C.LN

        # 1. 模型参数: 
        self.perlayer_mlp_params += (fc_params_num_per_layer + nonfc_params_num_per_layer) * C.param_type  # type: ignore
        # 2. 本地梯度: fc1_G=D*mH*le*mlp_act_dim, fc2_G=mH*D*le, gate_G=D*e
        self.perlayer_mlp_grads += (fc_params_num_per_layer + nonfc_params_num_per_layer) * C.grad_type  # type: ignore
        # 3. 优化器状态及因为混合精度训练而维护的master参数和梯度分片
        # 2 for adam's momentum and variance
        self.mlp_opt_states += 2 * (fc_params_num_per_layer / self.mlp_os_denom + nonfc_params_num_per_layer / self.os_denom) * C.LN_per_gpu * C.os_type  # type: ignore
        self.mlp_master_params += (fc_params_num_per_layer / self.mlp_os_denom + nonfc_params_num_per_layer / self.os_denom) * C.LN_per_gpu * C.master_type  # type: ignore
        self.perlayer_mlp_master_grads = (fc_params_num_per_layer / self.mlp_os_denom + nonfc_params_num_per_layer / self.os_denom) * C.master_grad_type # type: ignore
        self.mlp_master_grads += (fc_params_num_per_layer / self.mlp_os_denom + nonfc_params_num_per_layer / self.os_denom) * C.LN_per_gpu * C.master_grad_type  # type: ignore

        # 4. 激活值
        # router的输入 hidden_states [L, MBS, D]
        if C.recompute_norm:
            pass
        else:
            self.perlayer_mlp_acts += C.L/C.tp * C.MBS * C.D * C.act_type  # type: ignore
        # permute 和 unpermute 在计算梯度时，需要保留 sorted_indices [bsk], 其中bsk=MBS*L*topk
        self.perlayer_mlp_acts += C.L/C.tp * C.MBS * C.topk * 1 * C.act_type  # type: ignore
        # router's permutated_probs [bsk, 1]
        self.perlayer_mlp_acts += C.capacity * C.L/C.tp * C.MBS * C.topk * 1 * C.linear_act_type  # type: ignore
        # 假设路由均衡，fc1中le个expert的总体输入 permuted_hidden_states [bsk, h]，其中bsk个token会被平均分到le个expert中
        self.perlayer_mlp_acts += C.capacity * C.L/C.tp * C.MBS * C.topk * C.D * C.linear_act_type  # type: ignore
        # swiglu的输入: le个expert的总体输出 gate [bsk, mH] 和 up_proj [bsk, mH]
        self.perlayer_mlp_acts += 2 * C.capacity * C.L/C.tp * C.MBS * C.topk * C.mH * C.linear_act_type  # type: ignore
        if C.recompute_swiglu:
            pass
        else:
            # fc2中le个expert的总体输入 intermediate_states [bsk, mH]
            self.perlayer_mlp_acts += C.capacity * C.L/C.tp * C.MBS * C.topk * C.mH * C.linear_act_type  # type: ignore
        if C.topk_weights_position == "after_fc2":
            # 如果topk_weights(probs)在fc2的输出之后相乘，则反向计算要求保存fc2的输出 [bsk, D]
            self.perlayer_mlp_acts += C.L/C.tp * C.MBS * C.topk * C.D * C.act_type  # type: ignore
        
        # shared experts
        # fc1 的输入和permute共享，不再重复计算
        # swiglu的输入: fc1的输出 gate [bs, mH*n_shared_experts] 和 up_proj [bs, mH*n_shared_experts]
        self.perlayer_mlp_acts += 2 * C.L/C.tp * C.MBS * C.mH * C.n_shared_experts * C.linear_act_type  # type: ignore
        # fc2 的输入：swiglu的输出 [bs, mH*n_shared_experts]
        if C.recompute_swiglu:
            pass
        else:
            self.perlayer_mlp_acts += C.L/C.tp * C.MBS * C.mH * C.n_shared_experts * C.linear_act_type  # type: ignore
    
    def mlp_residual(self):
        # 由于probs提前到fc2之前，residual反向时不需要保存激活值
        pass
    
    def _print_bars(self, title, parts_dict, total_bytes, width=120, char="█"):
        """在控制台打印简单的柱状图来表示占比。
        Args:
            title (str): 标题。
            parts_dict (dict[str, float]): 名称到字节数的映射。
            total_bytes (float): 总字节数。
            width (int): 柱状图宽度（字符数）。
            char (str): 柱字符。
        """
        print(title)
        if total_bytes <= 0:
            print("  (no data)")
            return
        for name, value in sorted(parts_dict.items(), key=lambda kv: kv[1], reverse=True):
            pct = value / total_bytes if total_bytes else 0.0
            bar_len = int(round(pct * width))
            bar = char * bar_len + ' ' * (width - bar_len)
            print(f"  {name:<20} |{bar}| {pct*100:5.1f}% ({value / 1024**3:.3f} GiB)")
        print("")
    

def load_config(config_path):
    config_dict = {}
    
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader) or {}
    else:
        print(f"WARN: config file '{config_path}' not found, use default config")
    
    return Config(**config_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to llm_calc_config.yaml")
    parser.add_argument("-o", "--output", type=str, help="path to df excel file")
    args = parser.parse_args()

    C = load_config(args.config)

    print("setting:".center(100, "-"))
    print(f"model:{C.model} zero_stage:{C.zero_stage} tp:{C.tp} pp:{C.pp} dp:{C.dp} ep:{C.ep} vpp:{C.vpp} nodes:{C.nodes} " 
          f"mbs:{C.MBS} gbs:{C.GBS} L:{C.L} recompute:{C.recompute} flash_attn:{C.flash_attn} capacity:{C.capacity} LN:{C.LN}")
    print("-" * 100)

    calc = Calculator()
    calc.run(args.output)
