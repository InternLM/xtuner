import ray
import os
from uuid import uuid4
from copy import deepcopy

os.environ["XTUNER_USE_LMDEPLOY"] = "1"

from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v2.rollout_controller import RolloutController
from xtuner.v2.simple_env_runner import SimpleEnvRunner
from xtuner.v2.rollout_state import ProcessorUtilState, RolloutState
from xtuner.v1.ray.rollout.controller import SampleParams
from xtuner.v1.ray.judger.gsm8k import compute_reward


TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": 32,  # Aggressive: 32 concurrent processes
    # Python interpreter settings
    "python_timeout": 120,  # 2 minutes for complex calculations
    "python_memory_limit": "4GB",  # 4GB per Python process
    "python_cpu_limit": 1,
    # Memory management settings
    "max_memory_usage": 12288,  # 12GB total (75% of 16GB)
    "cleanup_threshold": 6144,  # 6GB
    "aggressive_cleanup_threshold": 3072,  # 3GB
    "force_cleanup_threshold": 9216,  # 9GB
}

# 一旦自定义函数，partial rollout 不太好做。
async def gsm8k_with_tools_generate(rollout_state: RolloutState, 
                                    processor_utils_state: ProcessorUtilState, 
                                    rollout_controller:RolloutController, 
                                    judger) -> RolloutState:
    # 可以自己用 tokens，也可以用 message + tools
    prompt = processor_utils_state.tokenizernizer.apply_chat_template(
         rollout_state.messages,
         tools=rollout_state.tools if hasattr(rollout_state, "tools") else None,
         tokenize=False,
         add_generation_prompt=True,
        )
    prompt_tokens_ids = processor_utils_state.tokenizer.tokenize(prompt)["input_ids"]

    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    rollout_log_probs=[]
    init_rollout_state = deepcopy(rollout_state)
    for turn in range(TOOL_CONFIGS["max_turns"]):
        current_token_ids = prompt_tokens_ids + response_token_ids
        rollout_state.tokens = current_token_ids
        init_rollout_state.tokens = rollout_state.tokens

        # TODO: 需要注意 ray actor 数据传输数据量，原则上他不需要的不应该传入
        # 发送给 rollout_controller 的对象只需要追加 tokens 即可，其余内容他不考虑

        # 相比于直接发送 post 请求，确实会麻烦一些，学习成本高一些
        # 但是由于有一层管理，少掉了给 n 个 url 然后自己路由的麻烦
        # TODO: 是否要换成 url post 方式？
        rollout_state = await rollout_controller.rollout.remote(rollout_state)

        init_rollout_state.state = rollout_state.state

        response_token_ids += rollout_state.response_ids
        rollout_log_probs += rollout_state.logprobs
        loss_masks += [1] * len(rollout_state.response_ids)

        if rollout_state.state == RolloutState.State.ABORTED:
            # rollout_state 内容转移到 init_rollout_state 中，他是全局保存的
            init_rollout_state.state = rollout_state.state
            init_rollout_state.response_ids = response_token_ids
            init_rollout_state.logprobs = rollout_log_probs
            init_rollout_state.loss_mask = loss_masks
            return init_rollout_state
        
        # 执行工具
        next_obs, done = await execute_predictions(rollout_state.response)
        if done:
            break

        obs_tokens_ids = processor_utils_state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        rollout_log_probs += [0.0] * len(obs_tokens_ids)
        loss_masks += [0] * len(obs_tokens_ids)

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break
    
    # rollout_state 内容转移到 init_rollout_state 中，他是全局保存的
    init_rollout_state.response_ids = response_token_ids
    init_rollout_state.logprobs = rollout_log_probs
    init_rollout_state.loss_mask = loss_masks
    return init_rollout_state


if __name__ == '__main__':

    ray.init(num_cpus=80, ignore_reinit_error=True)

    # model_path = '/mnt/shared-storage-user/llmrazor-share/model/intern-s1-mini-hha-fix_tokenizer'
    model_path ='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'

    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_workers=1,  # 1 or 8
        num_cpus_per_worker=12,
        cpu_memory_per_worker=16 * 1024 ** 3,  # 16 GB
    )
    pg = AutoAcceleratorWorkers.build_placement_group(resources)

    # 2. rollout
    rollout_config = RolloutConfig(
        device=resources.accelerator,
        model_path=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        expert_parallel_size=1,
        gpu_memory_utilization=0.75
    )
    
    rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg) 
    
    processor_utils_state = ProcessorUtilState(hf_checkpoint=model_path, sample_params=SampleParams())
    simple_env_runner = ray.remote(SimpleEnvRunner).remote(rollout_controller, 
                                                           processor_utils_state=processor_utils_state,
                                                           generate_external=gsm8k_with_tools_generate)
    
    prompt = [{"role": "user", "content": 'Calculate 13+24=', "type": "text"}]
    data_item = {'prompt': prompt}

    input_ids = processor_utils_state.tokenizer.apply_chat_template(data_item["prompt"], return_tensors="pt")["input_ids"][0].tolist()
    rollout_state = RolloutState(uid=uuid4().int, tokens=input_ids)
    
    # 生成单条
    res1 = ray.get(simple_env_runner.generate.remote(rollout_state))
    print("Response from infer:", res1)

    # 生成一组
    res1 = ray.get(simple_env_runner.generate_group.remote(rollout_state))
    print("Response from infer:", res1)

    # 生成多条
    res1 = ray.get(simple_env_runner.generate_batch.remote(batch_size=64))
    print("Response from infer:", res1)

    ray.get(rollout_controller.shutdown.remote(), timeout=300)


