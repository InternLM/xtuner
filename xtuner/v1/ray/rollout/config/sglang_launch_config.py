import inspect
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.server_args import ServerArgs

from xtuner.v1.utils import get_logger


logger = get_logger()


class SGLangDefaultServerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Model and tokenizer
    model_path: str = ""
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    tokenizer_worker_num: int = 1
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    model_loader_extra_config: str = "{}"
    trust_remote_code: bool = False
    modelopt_quant: Optional[Union[str, Dict]] = None
    modelopt_checkpoint_restore_path: Optional[str] = None
    modelopt_checkpoint_save_path: Optional[str] = None
    context_length: Optional[int] = None
    is_embedding: bool = False
    enable_multimodal: Optional[bool] = None
    revision: Optional[str] = None
    model_impl: str = "auto"

    # HTTP server
    host: str = "127.0.0.1"
    port: int = 30000
    skip_server_warmup: bool = False
    warmups: Optional[str] = None
    nccl_port: Optional[int] = None

    # Quantization and data type
    dtype: str = "auto"
    quantization: Optional[str] = None
    quantization_param_path: Optional[str] = None
    kv_cache_dtype: str = "auto"
    enable_fp32_lm_head: bool = False

    # Memory and scheduling
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = None
    max_queued_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    chunked_prefill_size: Optional[int] = None
    max_prefill_tokens: int = 16384
    schedule_policy: str = "fcfs"
    enable_priority_scheduling: bool = False
    schedule_low_priority_values_first: bool = False
    priority_scheduling_preemption_threshold: int = 10
    schedule_conservativeness: float = 1.0
    page_size: Optional[int] = None
    hybrid_kvcache_ratio: Optional[float] = None
    swa_full_tokens_ratio: float = 0.8
    disable_hybrid_swa_memory: bool = False
    radix_eviction_policy: str = "lru"

    # Runtime options
    device: Optional[str] = None
    tp_size: int = 1
    pp_size: int = 1
    pp_max_micro_batch_size: Optional[int] = None
    stream_interval: int = 1
    stream_output: bool = False
    random_seed: Optional[int] = None
    constrained_json_whitespace_pattern: Optional[str] = None
    watchdog_timeout: float = 300
    dist_timeout: Optional[int] = None  # timeout for torch.distributed
    download_dir: Optional[str] = None
    base_gpu_id: int = 0
    gpu_id_step: int = 1
    sleep_on_idle: bool = False

    # Logging
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    log_requests_level: int = 2
    crash_dump_folder: Optional[str] = None
    crash_on_nan: bool = False
    show_time_cost: bool = False
    enable_metrics: bool = False
    enable_metrics_for_all_schedulers: bool = False
    tokenizer_metrics_custom_labels_header: str = "x-custom-labels"
    tokenizer_metrics_allowed_custom_labels: Optional[List[str]] = None
    bucket_time_to_first_token: Optional[List[float]] = None
    bucket_inter_token_latency: Optional[List[float]] = None
    bucket_e2e_request_latency: Optional[List[float]] = None
    collect_tokens_histogram: bool = False
    prompt_tokens_buckets: Optional[List[str]] = None
    generation_tokens_buckets: Optional[List[str]] = None
    decode_log_interval: int = 40
    enable_request_time_stats_logging: bool = False
    kv_events_config: Optional[str] = None
    gc_warning_threshold_secs: float = 0.0
    enable_trace: bool = False
    oltp_traces_endpoint: str = "localhost:4317"

    # API related
    api_key: Optional[str] = None
    served_model_name: Optional[str] = None
    weight_version: str = "default"
    chat_template: Optional[str] = None
    completion_template: Optional[str] = None
    file_storage_path: str = "sglang_storage"
    enable_cache_report: bool = False
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None
    tool_server: Optional[str] = None
    sampling_defaults: str = "model"

    # Data parallelism
    dp_size: int = 1
    load_balance_method: str = "round_robin"
    load_watch_interval: float = 0.1
    # FIXME: remove this after dp rank scheduling is fully supported with PD-Disaggregation
    prefill_round_robin_balance: bool = False

    # Multi-node distributed serving
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    # Model override args in JSON
    json_model_override_args: str = "{}"
    preferred_sampling_params: Optional[str] = None

    # LoRA
    enable_lora: Optional[bool] = None
    max_lora_rank: Optional[int] = None
    lora_target_modules: Optional[Union[set[str], List[str]]] = None
    lora_paths: Optional[Union[dict[str, str], List[dict[str, str]], List[str], List[LoRARef]]] = None
    max_loaded_loras: Optional[int] = None
    max_loras_per_batch: int = 8
    lora_backend: str = "triton"
    max_lora_chunk_size: Optional[int] = 16

    # Kernel backend
    attention_backend: Optional[str] = None
    decode_attention_backend: Optional[str] = None
    prefill_attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    grammar_backend: Optional[str] = None
    mm_attention_backend: Optional[str] = None
    nsa_prefill: str = "flashmla_prefill"
    nsa_decode: str = "fa3"

    # Speculative decoding
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_draft_model_revision: Optional[str] = None
    speculative_num_steps: Optional[int] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_accept_threshold_single: float = 1.0
    speculative_accept_threshold_acc: float = 1.0
    speculative_token_map: Optional[str] = None
    speculative_attention_mode: str = "prefill"
    # For ngram only
    speculative_ngram_min_match_window_size: int = 1
    speculative_ngram_max_match_window_size: int = 12
    speculative_ngram_min_bfs_breadth: int = 1
    speculative_ngram_max_bfs_breadth: int = 10
    speculative_ngram_match_type: Literal["BFS", "PROB"] = "BFS"
    speculative_ngram_branch_length: int = 18
    speculative_ngram_capacity: int = 10 * 1000 * 1000

    # Expert parallelism
    ep_size: int = 1
    moe_a2a_backend: Literal["none", "deepep"] = "none"
    moe_runner_backend: str = "auto"
    flashinfer_mxfp4_moe_precision: Literal["default", "bf16"] = "default"
    enable_flashinfer_allreduce_fusion: bool = False
    deepep_mode: Literal["auto", "normal", "low_latency"] = "auto"
    ep_num_redundant_experts: int = 0
    ep_dispatch_algorithm: Optional[Literal["static", "dynamic", "fake"]] = None
    init_expert_location: str = "trivial"
    enable_eplb: bool = False
    eplb_algorithm: str = "auto"
    eplb_rebalance_num_iterations: int = 1000
    eplb_rebalance_layers_per_chunk: Optional[int] = None
    eplb_min_rebalancing_utilization_threshold: float = 1.0
    expert_distribution_recorder_mode: Optional[Literal["stat", "stat_approx", "per_pass", "per_token"]] = None
    expert_distribution_recorder_buffer_size: Optional[int] = None
    enable_expert_distribution_metrics: bool = False
    deepep_config: Optional[str] = None
    moe_dense_tp_size: Optional[int] = None

    # Mamba cache
    max_mamba_cache_size: Optional[int] = None
    mamba_ssm_dtype: str = "float32"

    # Hierarchical cache
    enable_hierarchical_cache: bool = False
    hicache_ratio: float = 2.0
    hicache_size: int = 0
    hicache_write_policy: str = "write_through"
    hicache_io_backend: str = "kernel"
    hicache_mem_layout: str = "layer_first"
    hicache_storage_backend: Optional[str] = None
    hicache_storage_prefetch_policy: str = "best_effort"
    hicache_storage_backend_extra_config: Optional[str] = None
    # LMCache
    enable_lmcache: bool = False

    # Double Sparsity
    enable_double_sparsity: bool = False
    ds_channel_config_path: Optional[str] = None
    ds_heavy_channel_num: int = 32
    ds_heavy_token_num: int = 256
    ds_heavy_channel_type: str = "qk"
    ds_sparse_decode_threshold: int = 4096

    # Offloading
    cpu_offload_gb: int = 0
    offload_group_size: int = -1
    offload_num_in_group: int = 1
    offload_prefetch_step: int = 1
    offload_mode: str = "cpu"

    # Scoring configuration
    # Delimiter token ID used to combine Query and Items into a single sequence for multi-item scoring.
    # Format: Query<delimiter>Item1<delimiter>Item2<delimiter>...
    # This enables efficient batch processing of multiple items against a single query.
    multi_item_scoring_delimiter: Optional[Union[int]] = None

    # Optimization/debug options
    disable_radix_cache: bool = False
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    enable_profile_cuda_graph: bool = False
    enable_cudagraph_gc: bool = False
    enable_nccl_nvls: bool = False
    enable_symm_mem: bool = False
    disable_flashinfer_cutlass_moe_fp4_allgather: bool = False
    enable_tokenizer_batch_encode: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    enable_mscclpp: bool = False
    enable_torch_symm_mem: bool = False
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_dp_lm_head: bool = False
    enable_two_batch_overlap: bool = False
    enable_single_batch_overlap: bool = False
    tbo_token_distribution_threshold: float = 0.48
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    triton_attention_split_tile_size: Optional[int] = None
    num_continuous_decode_steps: int = 1
    delete_ckpt_after_loading: bool = False
    enable_memory_saver: bool = False
    enable_weights_cpu_backup: bool = False
    allow_auto_truncate: bool = False
    enable_custom_logit_processor: bool = False
    flashinfer_mla_disable_ragged: bool = False
    disable_shared_experts_fusion: bool = False
    disable_chunked_prefix_cache: bool = False
    disable_fast_image_processor: bool = False
    keep_mm_feature_on_device: bool = False
    enable_return_hidden_states: bool = False
    scheduler_recv_interval: int = 1
    numa_node: Optional[List[int]] = None
    enable_deterministic_inference: bool = False

    # Dynamic batch tokenizer
    enable_dynamic_batch_tokenizer: bool = False
    dynamic_batch_tokenizer_batch_size: int = 32
    dynamic_batch_tokenizer_batch_timeout: float = 0.002

    # Debug tensor dumps
    debug_tensor_dump_output_folder: Optional[str] = None
    debug_tensor_dump_input_file: Optional[str] = None
    debug_tensor_dump_inject: bool = False
    debug_tensor_dump_prefill_only: bool = False

    # PD disaggregation: can be "null" (not disaggregated), "prefill" (prefill-only), or "decode" (decode-only)
    disaggregation_mode: Literal["null", "prefill", "decode"] = "null"
    disaggregation_transfer_backend: str = "mooncake"
    disaggregation_bootstrap_port: int = 8998
    disaggregation_decode_tp: Optional[int] = None
    disaggregation_decode_dp: Optional[int] = None
    disaggregation_prefill_pp: Optional[int] = 1
    disaggregation_ib_device: Optional[str] = None
    disaggregation_decode_enable_offload_kvcache: bool = False
    num_reserved_decode_tokens: int = 512  # used for decode kv cache offload in PD
    # FIXME: hack to reduce ITL when decode bs is small
    disaggregation_decode_polling_interval: int = 1

    # For model weight update and weight loading
    custom_weight_loader: Optional[List[str]] = None
    weight_loader_disable_mmap: bool = False
    remote_instance_weight_loader_seed_instance_ip: Optional[str] = None
    remote_instance_weight_loader_seed_instance_service_port: Optional[int] = None
    remote_instance_weight_loader_send_weights_group_ports: Optional[List[int]] = None

    # For PD-Multiplexing
    enable_pdmux: bool = False
    sm_group_num: int = 3

    def to_sglang_server_args(self) -> ServerArgs:
        server_args_params = set(inspect.signature(ServerArgs).parameters.keys())
        default_server_args_fields = set(self.model_fields.keys())
        missing_params = server_args_params - default_server_args_fields
        if missing_params:
            logger.info("Parameters in SGLang ServerArgs but not initialized in Xtuner DefaultServerArgs:")
            for param in sorted(missing_params):
                logger.info(f"- {param}")

        default_args_dict = self.model_dump()
        filtered_args = {key: value for key, value in default_args_dict.items() if key in server_args_params}
        return ServerArgs(**filtered_args)
