

# 非共卡，是否异步理论上是不区分的
class DisaggregatedRLTrainer:
    def __init__(
        self,
        config,
        env_runner, # 不允许有多少 env runner, 多 task 场景下，应该传入的是一个 composite env runner，由用户自己组织。这样这个类才能简洁通用
        update_weighter,

        rollout_controller,
        training_controller
    ):
        self.config = config
        self.env_runner = env_runner
        self.update_weighter = update_weighter
        self.training_controller = training_controller
        self.rollout_controller = rollout_controller
        self.replay_buffer = [] # 这个场景下，实际上 list 就足够了
        self.batch_size: list[int] = config.batch_size

        self.env_runner.set_controller(
            rollout_controller=self.rollout_controller,
            training_controller=self.training_controller)
        self.update_weighter.set_controllers(
            rollout_controller=self.rollout_controller,
            training_controller=self.training_controller)
        
        self.training_steps_per_epoch=1
        self.total_steps = 100
        self.require_batches = config.batch_size // 4

    def train_loop(self):
        for rollout_id in range(self.total_steps):
            self.train_step(rollout_id)
    
    def train_step(self, rollout_id):
        self.replay_buffer=[]
        # Collect rollouts
        for trajectory in self.env_runner.generate_batch(self.batch_size):
            self.replay_buffer.add(trajectory)
            
            # 达到指定的 batch 数量后，开始训练
            if len(self.replay_buffer)>= self.require_batches:
                # Train the model
                for _ in range(self.training_steps_per_epoch):
                    # 从 replay buffer 采样 batch 后，replay_buffer 内部已经训练的数据要清空
                    batch = self.replay_buffer.pop(self.require_batches)
                    train_batch= self.convert_rollout_batch_to_train_batch(batch)
                    self.training_controller.fit(train_batch)

        self.sync_weights()
        self.save_ckpt()

    def convert_rollout_batch_to_train_batch(self, batch):
        # 这里假设 rollout batch 和 train batch 是一样的
        return batch 
