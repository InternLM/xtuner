

# 共卡，是否异步应该不区分的
class ColocateRLTrainer:
    def __init__(
        self,
        config,
        env_runner, # 不允许有多少 env runner, 多 task 场景下，应该传入的是一个 composite env runner，由用户自己组织。这样这个类才能简洁通用
        weight_controller,

        rollout_controller,
        training_controller # ppo 算法的所有训练细节都是在这个类里面做，这个 trainer 也感知不到
    ):
        self.config = config
        self.env_runner = env_runner
        self.weight_controller = weight_controller
        self.training_controller = training_controller
        self.rollout_controller = rollout_controller
        self.replay_buffer = [] # 这个场景下，实际上 list 就足够了
        self.batch_size: list[int] = config.batch_size
        
        self.env_runner.set_controller(
            rollout_controller=self.rollout_controller,
            training_controller=self.training_controller)
        self.weight_controller.set_controllers(
            rollout_controller=self.rollout_controller,
            training_controller=self.training_controller)
        
        self.training_steps_per_epoch=1
        self.total_steps = 100

    def train_loop(self):
        for rollout_id in range(self.total_steps):
            self.train_step(rollout_id)
    
    def train_step(self, rollout_id):
        self.replay_buffer=[]

        # offload train controller to cpu
        self.training_controller.offload_to_cpu()
        # load rollout_controller
        self.rollout_controller.load_to_device()
        
        # Collect rollouts
        for trajectory in self.env_runner.generate_batch(self.batch_size):
            self.replay_buffer.add(trajectory)

        # offload rollout_controller to cpu
        self.rollout_controller.offload_to_cpu()
        # load train controller
        self.training_controller.load_to_device()    
        
        # Train the model
        for _ in range(self.training_steps_per_epoch):
            batch = self.replay_buffer
            train_batch= self.convert_rollout_batch_to_train_batch(batch)
            self.training_controller.fit(train_batch)
        
        # ipc 和 nccl 应该是两套不同的实现，但是接口一致
        self.weight_controller.update_weights()

    def convert_rollout_batch_to_train_batch(self, batch):
        # 这里假设 rollout batch 和 train batch 是一样的
        return batch 
