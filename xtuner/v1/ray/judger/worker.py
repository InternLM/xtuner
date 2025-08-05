import asyncio

import ray

from xtuner.v1.ray import SingleAcceleratorWorker


class JudgerWorker(SingleAcceleratorWorker):
    def __init__(
        self, config: dict, rank: int, master_addr: str, master_port: int, world_size: int, accelerator: str = "CPU"
    ):
        self.config = config
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.server_addr: str = ""
        self.server_port: str = ""
        self.world_size = world_size
        self.launch_method: str = "function"
        self.paused = False

    def init(self, config, launch_method: str = "function"):
        # todo(@duanyanhui): Wrap the judge_function as an OpenAI API server
        pass

    async def judge_task(self, inqueue, outqueue):
        while not self.paused:
            if inqueue.empty() or outqueue.full():
                print("JudgerWorker judge_task waiting for data or outqueue full")
                await asyncio.sleep(0.1)
                continue
            data = inqueue.get()
            reward = self.judge_function(data)
            outqueue.put(ray.put((data, reward)))
            await asyncio.sleep(0.1)
        return

    async def judge(self, inqueue, outqueue):
        self.paused = False
        await self.judge_task(inqueue, outqueue)

    def judge_function(self, data: ray.ObjectRef):
        raise NotImplementedError("judge must be implemented in subclass")

    def pause(self):
        self.paused = True
