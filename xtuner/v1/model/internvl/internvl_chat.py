from typing import List, cast
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from .configuration_internvl_chat import InternVLChatConfig
from ..moe.qwen3 import Qwen3MoE
from .modeling_intern_vit import InternVisionModel
from .configuration_intern_vit import InternVisionConfig
from xtuner.v1.config.base_model import MoEModelOutputs
from xtuner.v1.utils import HFCheckpointLoader, get_logger, get_padding_length
from xtuner.v1.ops.comm import split_for_sequence_parallel
import torch.distributed.nn.functional as distF
import torch.distributed as dist
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import GreedyRouterConfig
from xtuner.v1.model.moe.moe import MoEConfig, SequenceContext

logger = get_logger()


def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
               int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class InternVLChatModel(nn.Module):

    # TODO: No distinction between dense and moe models
    def __init__(self, config: InternVLChatConfig, ep_mesh: DeviceMesh | None = None, dispatcher: str = "deepep"):
        super(InternVLChatModel, self).__init__()

        self.select_layer = config.select_layer
        self.downsample_ratio = config.downsample_ratio

        vision_config: InternVisionConfig = config.vision_config
        llm_config = config.llm_config

        self.vision_model = InternVisionModel(vision_config)

        vit_hidden_size = vision_config.hidden_size
        llm_hidden_size = llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        if ep_mesh is not None and ep_mesh.size() == 1:
            dispatcher = "naive"

        replace_llm_config = self._replace_llm_config(llm_config, dispatcher)

        if llm_config.architectures[0] == 'Qwen3MoeForCausalLM':
            self.llm_model = Qwen3MoE(replace_llm_config, ep_mesh=ep_mesh)
        else:
            raise NotImplementedError

        self.img_context_token_id = None
        self._llm_prefix = 'llm_model.'
        self._hf_llm_prefix = 'language_model.'

    def _replace_llm_config(self, llm_config, dispatcher):
        router_config = GreedyRouterConfig(
            scoring_func="sigmoid",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        )
        attention_config = MHAConfig(
            num_attention_heads=llm_config.num_attention_heads,
            num_key_value_heads=llm_config.num_key_value_heads,
            head_dim=llm_config.head_dim,
            qk_norm=True
        )
        config = MoEConfig(
            vocab_size=llm_config.vocab_size,
            max_position_embeddings=llm_config.max_position_embeddings,
            padding_idx=0,  # TODO: 暂时不正确
            num_hidden_layers=llm_config.num_hidden_layers,
            hidden_size=llm_config.hidden_size,
            intermediate_size=llm_config.intermediate_size,
            rms_norm_eps=llm_config.rms_norm_eps,
            rope_theta=llm_config.rms_norm_eps,
            hidden_act=llm_config.hidden_act,
            attention=attention_config,
            tie_word_embeddings=llm_config.tie_word_embeddings,
            training_dtype="bf16",
            chunked_loss=False,
            n_routed_experts=llm_config.num_experts,
            n_shared_experts=0,
            num_experts_per_tok=llm_config.num_experts_per_tok,
            first_k_dense_replace=0,
            hidden_factor=1.0,
            moe_intermediate_size=llm_config.moe_intermediate_size,
            dispatcher=dispatcher,
            router=router_config,
        )
        return config

    def to_hf_key_list(self, key: str) -> str | List[str]:
        if key.startswith(self._llm_prefix):
            hf_keys = self.llm_model.to_hf_key_list(key[len(self._llm_prefix):])
            if not isinstance(hf_keys, list):
                hf_keys = [hf_keys]
            for i, hf_key in enumerate(hf_keys):
                hf_keys[i] = self._hf_llm_prefix + hf_key
            return hf_keys
        else:
            return key

    def from_hf(self, hf_path: str, device: torch.device | None = None, strict=True):
        # load moe weights from HF checkpoint
        self.llm_model.from_hf(hf_path, device=device, strict=False, prefix=self._hf_llm_prefix)

        # load other weights from HF checkpoint
        hf_loader = HFCheckpointLoader(hf_path)

        not_matched = []
        not_loaded = []
        loaded: List = []

        with torch.no_grad():
            for name, value in self.state_dict().items():
                if name.startswith(self._llm_prefix):
                    hf_keys = self.to_hf_key_list(name)
                    loaded.extend(hf_keys)
                    continue

                hf_keys = self.to_hf_key_list(name)
                hf_value = hf_loader.load(hf_keys)
                if hf_value is None:
                    not_loaded.append(f"{name}")
                    logger.warning(f"Parameter {f'{name}'} -> {hf_keys} not found in HF checkpoint.")
                    continue

                if hf_value.shape != value.shape:
                    not_matched.append(
                        f"Parameter {f'{name}'} -> {hf_keys}: {f'{name}'} {hf_value.shape} != {value.shape}"
                    )
                    logger.warning(
                        f"Parameter {f'{name}'} shape mismatch: expected {value.shape}, got {hf_value.shape}."
                    )
                value.copy_(hf_value)
                loaded.append(hf_keys)

        missing = set(hf_loader.weight_map) - set(loaded)

        if strict:
            if not_matched:
                raise RuntimeError(f"Some parameters from {hf_path} do not match the model: {not_matched}. ")
            if missing:
                raise RuntimeError(f"Missing parameters from {hf_path}: {list(missing)[0]}. ")

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def forward(
            self,
            seq_ctx: SequenceContext,
            labels: torch.LongTensor,
            return_router_results: bool = False,
            return_hidden_states: bool = False,
    ) -> MoEModelOutputs:

        input_ids = seq_ctx.input_ids
        pixel_values = seq_ctx.pixel_values
        image_flags = seq_ctx.image_flags
        sequence_parallel_mesh = seq_ctx.sequence_parallel_mesh

        inputs_embeds = self.llm_model.embed_tokens(input_ids)

        if pixel_values is not None and image_flags is not None:
            image_flags = cast(torch.LongTensor, image_flags.squeeze(-1))

            # in-place op on custom-function outputs will spoil autograd
            inputs_embeds = inputs_embeds.clone()

            if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
                vit_batch_size = pixel_values.shape[0]
                divisors = [sequence_parallel_mesh.size()]
                pad_size = get_padding_length(vit_batch_size, divisors)
                if pad_size != 0:
                    pixel_values = torch.cat([pixel_values, # type: ignore
                                              pixel_values[0:1].repeat(pad_size, *[1] * (pixel_values.dim() - 1))],
                                             dim=0)
                pixel_values = pixel_values.chunk(sequence_parallel_mesh.size(), dim=0)[ # type: ignore
                    sequence_parallel_mesh.get_local_rank()
                ]

            vit_embeds = self.extract_feature(pixel_values)

            if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
                vit_embeds_list = distF.all_gather(vit_embeds, group=sequence_parallel_mesh.get_group())
                vit_embeds = torch.cat(vit_embeds_list, dim=0)[:vit_batch_size]

            vit_embeds = vit_embeds[image_flags == 1]

            if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
                inputs_embeds_list = distF.all_gather(inputs_embeds, group=sequence_parallel_mesh.get_group())
                inputs_embeds = torch.cat(inputs_embeds_list, dim=1)

                input_ids_list = [torch.empty_like(input_ids) for _ in range(sequence_parallel_mesh.size())]
                dist.all_gather(input_ids_list, input_ids, group=sequence_parallel_mesh.get_group())
                input_ids = torch.cat(input_ids_list, dim=1) # type: ignore

            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)

            input_ids = cast(torch.LongTensor, input_ids.reshape(B * N))

            selected = input_ids == self.img_context_token_id

            try:
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, inputs_embeds[selected].shape={inputs_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.sum() * 0

            inputs_embeds = inputs_embeds.reshape(B, N, C)
        else:
            # in-place op on custom-function outputs will spoil autograd
            inputs_embeds = inputs_embeds.clone()

        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            # TODO: 如果外面传过来的没有 pad，那么这个地方会比较麻烦，因为不仅仅要 input_embeds 需要 pad，
            #  position_ids,cu_seq_lens_k 等都要重算？
            inputs_embeds = split_for_sequence_parallel(inputs_embeds, dim=1, sp_mesh=sequence_parallel_mesh)

        seq_ctx.image_flags = None
        seq_ctx.pixel_values = None
        seq_ctx.input_ids = None  # type: ignore
        seq_ctx.inputs_embeds = inputs_embeds

        outputs = self.llm_model(
            seq_ctx,
            labels=labels,
            return_router_results=return_router_results,
            return_hidden_states=return_hidden_states
        )
        return outputs
