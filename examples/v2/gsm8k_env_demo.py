import ray
import os
from uuid import uuid4

os.environ["XTUNER_USE_LMDEPLOY"] = "1"

from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v2.rollout_controller import RolloutController
from xtuner.v2.base_env_runner import BaseEnvRunner
from xtuner.v2.rollout_state import ProcessorUtilState, RolloutState
from xtuner.v1.ray.rollout.controller import SampleParams
from xtuner.v1.ray.judger.gsm8k import compute_reward


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
    
    async def gsm8k_generate(data_item, processor_utils_state: ProcessorUtilState, rollout_controller: RolloutController, judger):
        input_ids = processor_utils_state.tokenizer.apply_chat_template(data_item["prompt"], return_tensors="pt")["input_ids"][0].tolist()
        rollout_state = RolloutState(uid=uuid4().int, input_ids=input_ids)
        rollout_state = await rollout_controller.generate.remote(rollout_state)
        
        # reward = compute_reward(processor_utils_state, data_item, rollout_state)
        # rollout_state.rewards = reward
        return rollout_state
    
    processor_utils_state = ProcessorUtilState(hf_checkpoint=model_path, sample_params=SampleParams())
    gsm8k_env_runner = ray.remote(BaseEnvRunner).remote(rollout_controller, processor_utils_state=processor_utils_state, generate_external=gsm8k_generate)
    
    # prompt = [{"content": [{"image_url": {"image_wh": [404, 162],
    #                                       "url": "images/test_2.jpg"},
    #                         "type": "image_url"},
    #                        {
    #                            "text": "<IMG_CONTEXT>Find the area of the figure to the nearest tenth. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.",
    #                            "type": "text"}],
    #            "role": "user"}]
    # extra_info = {'media_root': '/mnt/shared-storage-user/llmrazor-share/data/geometry3k/'}

    prompt = [{"role": "user", "content": 'Calculate 13+24=', "type": "text"}]
    data_item = {'prompt': prompt}
    
    res1 = ray.get(gsm8k_env_runner.generate.remote(data_item))
    print("Response from SGLang infer:", res1)
    ray.get(rollout_controller.shutdown.remote(), timeout=300)
