from cyclopts import App

from xtuner.v1.rl.config import GRPOTrainerConfig


app = App()


@app.default
def grpo(*, config: GRPOTrainerConfig):
    """Train a model using the GRPO trainer configuration.

    Args:
        config (GRPOTrainerConfig): Configuration for the GRPO trainer.
    """
    # 这里可以添加训练逻辑
    print(f"Training with config: {config}")


if __name__ == "__main__":
    # 测试使用预配置
    # db_config = DatabaseConfig(
    #     host="preconfigured.host",
    #     port=5432,
    #     database="preconfigured_db"
    # )
    # train_config = TrainConfig(config1=db_config)
    # train(train_config)

    # 或者从命令行运行
    app()
