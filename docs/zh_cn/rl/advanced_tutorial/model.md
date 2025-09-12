# 模型

新增一个模型的测试流程

1. 测试模型能否正常训练/保存和加载：pytest tests/ray/test_grpo_trainer.py

2. 测试rollout功能：pytest tests/ray/test_rollout.py，查看rollout输出是否正常，