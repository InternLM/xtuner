# 测试流程

1. 使用 xtuner 提供的镜像，自带全量的依赖库，包括 flash-attn，deepe-ep，deep-gemm 等
2. 准备环境变量，来指定测试时的模型路径和数据路径:
    - QWEN3_MOE_PATH: 指定 `models--Qwen--Qwen3-30B-A3B` 的模型路径
    - ALPACA_PATH: 指定 alpaca 数据集的路径，注意，该数据需要预处理成 xtuner 需要的格式
3. 运行测试：
    - 运行单元测试：`pytest tests`
    - 运行集成测试：
        - 测试 qwen3 在 alpaca 数据集上是否跑通：`torchrun --nproc-per-node 8  ci/scripts/test_sft_trainer.py` (精度验证待完成)
