import asyncio
import importlib

import ray

from xtuner.v1.ray.accelerator import AutoAcceleratorWorkers
from xtuner.v1.ray.judger.controller import JudgerController
from xtuner.v1.ray.rollout.controller import RolloutController


@ray.remote
class EnvController:
    def __init__(self, environment: str, placement_group, rollout_cfg, judger_cfg):
        self.environment = environment
        self.init_rollout_controller(placement_group, rollout_cfg)
        self.init_judger_controller(placement_group, judger_cfg)

    def init_rollout_controller(self, placement_group, rollout_cfg):
        if rollout_cfg.backend == "lmdeploy":
            from xtuner.v1.ray.rollout import LMDeployWorker

            rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(
                LMDeployWorker, rollout_cfg, placement_group
            )
            self.rollout_controller = RolloutController.remote(rollout_cfg, rollout_workers_map)
        else:
            raise NotImplementedError(f"Rollout backend '{rollout_cfg.backend}' is not supported.")

    def init_judger_controller(self, placement_group, judger_cfg):
        judger_class_path = judger_cfg.get("judger_type")
        if judger_class_path:
            try:
                module_path, class_name = judger_class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                judger_worker_class = getattr(module, class_name)
                judger_workers_map = AutoAcceleratorWorkers.from_placement_group(
                    judger_worker_class, judger_cfg, placement_group
                )
                self.judger_controller = JudgerController.remote(judger_workers_map, judger_cfg)
            except (ImportError, AttributeError, ValueError) as e:
                raise ImportError(f"Failed to import judger worker class '{judger_class_path}': {e}")

    async def run(self, enable_batch_reward, enable_partial_rollout, sample_params, group_samples):
        assert not enable_batch_reward

        response_future = [
            self.rollout_controller.rollout.remote(sample["prompt_str"], sample_params.dict())
            for sample in group_samples
        ]
        response = await asyncio.gather(*response_future)
        reward_future = [
            self.judger_controller.judge.remote(res[0], sample["reward_model"]["ground_truth"])
            for res, sample in zip(response, group_samples)
        ]
        reward = await asyncio.gather(*reward_future)

        for i in range(len(group_samples)):
            group_samples[i]["response_str"] = response[i][0]
            group_samples[i]["reward"] = reward[i]
            group_samples[i]["state"] = response[i][1]

        return group_samples

    def pause(self):
        return ray.get(self.rollout_controller.pause.remote())

    def shutdown(self):
        return ray.get(self.rollout_controller.shutdown.remote())

    def restart(self):
        return ray.get(self.rollout_controller.restart.remote())

    async def rollout(self, prompt):
        return await self.rollout_controller.rollout.remote(prompt)

    def get_rollout_info(self):
        return ray.get(self.rollout_controller.get_rollout_info.remote())

    def onload(self, *args, **kwargs):
        return ray.get(self.rollout_controller.onload.remote(*args, **kwargs))

    def offload(self, *args, **kwargs):
        return ray.get(self.rollout_controller.offload.remote(*args, **kwargs))
