# 可视化训练日志

XTuner 支持通过 [MMEngine](https://github.com/open-mmlab/mmengine) 使用 [TensorBoard](https://www.tensorflow.org/tensorboard?hl=zh-cn) 和 [Weights & Biases (WandB)](https://docs.wandb.ai/) 实验管理工具，你可以很方便地跟踪和可视化损失、显存占用等指标。

只需修改一行 config 文件即可配置上述实验管理工具。

## TensorBoard

设置 config 中的 `visualizer` 字段，并将 `vis_backends` 设置为 [TensorboardVisBackend](https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L514)：

```diff
+ from mmengine.visualization import Visualizer, TensorboardVisBackend
# set visualizer
- visualizer = None
+ visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])
```

启动实验后，tensorboard 产生的相关文件会存在 `vis_data` 中，通过 tensorboard 命令可以启动进行实时可视化：

![image](https://github.com/InternLM/xtuner/assets/67539920/abacb28f-5afd-46d0-91b2-acdd20887969)

```
tensorboard --logdir=$PATH_TO_VIS_DATA
```

## WandB

使用 WandB 前需安装依赖库 `wandb` 并登录至 wandb。

```bash
pip install wandb
wandb login
```

设置 config 中的 `visualizer` 字段，并将 `vis_backends` 设置为 [WandbVisBackend](https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L330)：

```diff
+ from mmengine.visualization import Visualizer, WandbVisBackend
# set visualizer
- visualizer = None
+ visualizer = dict(type=Visualizer, vis_backends=[dict(type=WandbVisBackend)])
```

启动实验后，可在 wandb 网页端 `https://wandb.ai` 上查看可视化结果：

![image](https://github.com/InternLM/xtuner/assets/41630003/fc16387a-3c83-4015-9235-8ec811077953)

可以点击 [WandbVisBackend API](https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L330) 查看 `WandbVisBackend` 可配置的参数。例如 `init_kwargs`，该参数会传给 [wandb.init](https://docs.wandb.ai/ref/python/init) 方法。

```diff
+ from mmengine.visualization import Visualizer, WandbVisBackend
# set visualizer
- visualizer = None
+ visualizer = dict(
+   type=Visualizer,
+   vis_backends=[
+       dict(type=WandbVisBackend, init_kwargs=dict(project='toy-example'))])
```
