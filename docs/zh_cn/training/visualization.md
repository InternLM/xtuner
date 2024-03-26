# 可视化训练日志

XTuner 支持通过 [MMEngine](https://github.com/open-mmlab/mmengine) 使用 [TensorBoard](https://www.tensorflow.org/tensorboard?hl=zh-cn)、[Weights & Biases (WandB)](https://docs.wandb.ai/)、[ClearML](https://clear.ml/docs/latest/docs)、[Neptune](https://docs.neptune.ai/)、[DVCLive](https://dvc.org/doc/dvclive) 和 [Aim](https://aimstack.readthedocs.io/en/latest/overview.html) 实验管理工具，你可以很方便地跟踪和可视化损失、显存占用等指标。

只需修改一行 config 文件即可配置上述实验管理工具。

## TensorBoard

设置 config 中的 `visualizer` 字段，并将 `vis_backends` 设置为 [TensorboardVisBackend](https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L514)：

```diff
+ from mmengine.visualization import Visualizer, TensorboardVisBackend
# set visualizer
- visualizer = None
+ visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])
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

## ClearML

使用 ClearML 前需安装依赖库 `clearml` 并参考 [Connect ClearML SDK to the Server](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#connect-clearml-sdk-to-the-server) 进行配置。

```bash
pip install clearml
clearml-init
```

设置 config 中的 `visualizer` 字段，并将 `vis_backends` 设置为 [ClearMLVisBackend](https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L859)：

```diff
+ from mmengine.visualization import Visualizer, ClearMLVisBackend
# set visualizer
- visualizer = None
+ visualizer = dict(type=Visualizer, vis_backends=[dict(type=ClearMLVisBackend)])
```

![image](https://github.com/InternLM/xtuner/assets/41630003/20961538-83eb-463e-91c5-1a359cb42112)

## Neptune

使用 Neptune 前需先安装依赖库 `neptune` 并登录 [Neptune.AI](https://docs.neptune.ai/) 进行配置。

```bash
pip install neptune
```

设置 config 中的 `visualizer` 字段，并将 `vis_backends` 设置为 [NeptuneVisBackend](https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L1000)：

```diff
+ from mmengine.visualization import Visualizer, NeptuneVisBackend
# set visualizer
- visualizer = None
+ visualizer = dict(type=Visualizer, vis_backends=[dict(type=NeptuneVisBackend)])
```

![image](https://github.com/InternLM/xtuner/assets/41630003/d346bead-68b3-48f1-ae0d-7ee7031189c8)

请注意：若未提供 `project` 和 `api_token` ，neptune 将被设置成离线模式，产生的文件将保存到本地 `.neptune` 文件下。
推荐在初始化时提供 `project` 和 `api_token` ，具体方法如下所示：

```diff
+ from mmengine.visualization import Visualizer, NeptuneVisBackend
# set visualizer
- visualizer = None
+ visualizer = dict(
+   type=Visualizer,
+   vis_backends=[
+       dict(type=NeptuneVisBackend,
+            init_kwargs=dict(
+               project='workspace-name/project-name',
+               api_token='your api token'))])
```

更多初始化配置参数可点击 [neptune.init_run API](https://docs.neptune.ai/api/neptune/#init_run) 查询。

## DVCLive

使用 DVCLive 前需先安装依赖库 `dvclive` 并参考 [iterative.ai](https://dvc.org/doc/start) 进行配置。常见的配置方式如下：

```bash
pip install dvclive
cd ${WORK_DIR}
git init
dvc init
git commit -m "DVC init"
```

设置 config 中的 `visualizer` 字段，并将 `vis_backends` 设置为 [DVCLiveVisBackend](https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L1144)：

```diff
+ from mmengine.visualization import Visualizer, DVCLiveVisBackend
# set visualizer
- visualizer = None
+ visualizer = dict(type=Visualizer, vis_backends=[dict(type=DVCLiveVisBackend)])
```

启动训练后，打开 `work_dir_dvc` 下面的 `report.html` 文件，即可看到如下图的可视化效果。

![image](https://github.com/InternLM/xtuner/assets/41630003/78f557a8-f1f1-429c-896f-1eb2132f3cec)

你还可以安装 VSCode 扩展 [DVC](https://marketplace.visualstudio.com/items?itemName=Iterative.dvc) 进行可视化。

更多初始化配置参数可点击 [DVCLive API Reference](https://dvc.org/doc/dvclive/live) 查询。

## Aim

使用 Aim 前需先安装依赖库 `aim`。

```bash
pip install aim
```

设置 config 中的 `visualizer` 字段，并将 `vis_backends` 设置为 [AimVisBackend](https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L1322)：

```diff
+ from mmengine.visualization import Visualizer, AimVisBackend
# set visualizer
- visualizer = None
+ visualizer = dict(type=Visualizer, vis_backends=[dict(type=AimVisBackend)])
```

设置 `Runner` 初始化参数中的 `visualizer`，并将 `vis_backends` 设置为 [AimVisBackend](mmengine.visualization.AimVisBackend)。

启动训练后，在终端中输入

```bash
aim up
```

或者在 Jupyter Notebook 中输入

```bash
%load_ext aim
%aim up
```

即可启动 Aim UI，界面如下图所示。

![image](https://github.com/InternLM/xtuner/assets/41630003/4b61089f-490a-46c1-8dd4-279bfa3f4bf4)

初始化配置参数可点击 [Aim SDK Reference](https://aimstack.readthedocs.io/en/latest/refs/sdk.html#module-aim.sdk.run) 查询。
