import ray


class DataFlow:
    def __init__(self, config, data_loader, environment):
        self.config = config
        self.environment = environment
        self.data_loader = data_loader
        self.outqueue = ray.util.queue.Queue()

    def dataflow_task(self):
        data = next(self.data_loader)
        response = self.environment.run(data)
        self.outqueue.put(response)

    def run(self, target_num):
        while self.outqueue.size() < target_num:
            self.dataflow_task()

        return [self.outqueue.get() for _ in range(target_num)]
