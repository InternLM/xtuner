

class Agent:
    def preprocess(self, sample: Sample) -> Sample:
        pass
    
    def generate_sample(self, sample: Sample) -> Sample:
        pass
    
    def postprocess(self, sample: Sample) -> Sample:
        pass


class SingleTurnAgent(Agent):
    def __init__(self, rollout_ctl: RolloutController):
        self._rollout_ctl = rollout_ctl
        self._judge = Judge()

    def generate_sample(self, sample: Sample) -> Sample:
        output = self._rollout_ctl.generate_sample(sample)
        output = self._judge.judge(output)
        return output

class MultiTurnAgent(Agent):
    ...


class MultiTurnToolAgent(Agent):
    ...


# TODO: 补充 Scheduler(包括同步和异步Scheduler), Roller(Agent?)
# TODO: 补充 API 接口
class Env:
    def __init__(self, rollout_ctl: RolloutController):
        self._agents: List[Agent(rollout_ctl)]
        self._agent_router: AgentRouter(self._agents)
    
    def produce_batch(self, data_mgr: DataManager):

        while finish_num < batch_size:
            for sample in data_mgr.next_data():  # TODO: current batch如何保持这个状态？
                self.generate_sample(sample)
            
    def generate_sample(self, sample: Sample):
        agent = self._agent_router(sample)
        out = agent.generate_sample(sample)
        return out

    def produce_loop(self, data_mgr: DataManager):
        pass


def main_colocate():
    data_mgr: DataManager
    pg: PlacementGroup
    rollout_ctl: RolloutController(pg)
    env: Env(rollout_ctl)
    train_ctl: TrainController(pg)

    eval_data_mgr: DataManager
    evaluator: Evaluator

    for i in range(total_rollouts):
        env.produce_batch(data_mgr)

        metrics = train_ctl.fit(data_mgr.get_batch())
        log_metrics(metrics)

        train_ctl.sync_weights(rollout_ctl)

        env.produce_batch(eval_data_mgr)
        eval_metrics = evaluator.evaluate(eval_data_mgr.get_batch())
        log_metrics(eval_metrics)
        

def main_colocate_lowlevel():
    data_mgr: DataManager
    pg: PlacementGroup
    rollout_ctl: RolloutController(pg)
    env: Env(rollout_ctl)
    train_ctl: TrainController(pg)

    eval_data_mgr: DataManager
    evaluator: Evaluator

    for i in range(total_rollouts):
        env.produce_batch(data_mgr)

        batch = data_mgr.get_batch()

        # below is equivalent to train_ctl.fit(batch)
        batch = pack_pad_dispatch(batch)
        batch = train_ctl.compute_old_logprobs(batch)
        batch = train_ctl.compute_ref_logprobs(batch)
        batch = train_ctl.compute_values(batch)
        batch = train_ctl.compute_advantages(batch)
        metrics = train_ctl.train(batch)

        log_metrics(metrics)

        train_ctl.sync_weights(rollout_ctl)

        env.produce_batch(eval_data_mgr)
        eval_metrics = evaluator.evaluate(eval_data_mgr.get_batch())
        log_metrics(eval_metrics)
        

def main_separate():
    data_mgr: DataManager
    pg1: PlacementGroup
    rollout_ctl: RolloutController(pg1)
    pg1_2: PlacementGroup
    rollout_ctl_2: RolloutController(pg1_2)
    env: Env(rollout_ctl, rollout_ctl_2)

    pg2: PlacementGroup
    train_ctl: TrainController(pg2)

    eval_data_mgr: DataManager
    evaluator: Evaluator

    producer_thread = threading.Thread(target=env.produce_loop, args=(data_mgr,))
    producer_thread.start()

    for i in range(total_rollouts):
        metrics = train_ctl.fit(data_mgr.get_batch())
        log_metrics(metrics)

        train_ctl.sync_weights(rollout_ctl)

        env.produce_batch(eval_data_mgr)  # 优先级高于env.produce_loop
        eval_metrics = evaluator.evaluate(eval_data_mgr.get_batch())
        log_metrics(eval_metrics)
