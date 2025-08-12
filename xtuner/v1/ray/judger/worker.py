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

    def judge(self, response, label):
        return self.judge_function(response, label)

    def judge_function(self, response, label):
        raise NotImplementedError("judge_function must be implemented in subclass")
