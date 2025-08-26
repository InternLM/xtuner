==============
可视化训练过程
==============

XTuner 支持通过 `MMEngine <https://github.com/open-mmlab/mmengine>`__
使用 `TensorBoard <https://www.tensorflow.org/tensorboard?hl=zh-cn>`__
和 `Weights & Biases (WandB) <https://docs.wandb.ai/>`__
实验管理工具，只需在 config 中添加一行代码，就可以跟踪和可视化损失、显存占用等指标。

TensorBoard
============

1. 设置 config 中的 ``visualizer`` 字段，并将 ``vis_backends`` 设置为 `TensorboardVisBackend <https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L514>`__\ ：

.. code:: diff

   # set visualizer
   - visualizer = None
   + from mmengine.visualization import Visualizer, TensorboardVisBackend
   + visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])

2. 启动实验后，tensorboard 产生的相关文件会存在 ``vis_data`` 中，通过 tensorboard 命令可以启动进行实时可视化：

|image1|

.. code::

   tensorboard --logdir=$PATH_TO_VIS_DATA

WandB
======

1. 使用 WandB 前需安装依赖库 ``wandb`` 并登录至 wandb。

.. code:: console

   $ pip install wandb
   $ wandb login

2. 设置 config 中的 ``visualizer`` 字段，并将 ``vis_backends`` 设置为 `WandbVisBackend <https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L330>`__\ ：

.. code:: diff

   # set visualizer
   + from mmengine.visualization import Visualizer, WandbVisBackend
   - visualizer = None
   + visualizer = dict(type=Visualizer, vis_backends=[dict(type=WandbVisBackend)])

.. tip::
   可以点击 `WandbVisBackend
   API <https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/mmengine/visualization/vis_backend.py#L330>`__
   查看 ``WandbVisBackend`` 可配置的参数。例如
   ``init_kwargs``\ ，该参数会传给
   `wandb.init <https://docs.wandb.ai/ref/python/init>`__ 方法。

   .. code:: diff

      # set visualizer
      - visualizer = None
      + from mmengine.visualization import Visualizer, WandbVisBackend
      + visualizer = dict(
      +   type=Visualizer,
      +   vis_backends=[
      +       dict(type=WandbVisBackend, init_kwargs=dict(project='toy-example'))])


3. 启动实验后，可在 wandb 网页端 ``https://wandb.ai`` 上查看可视化结果：

|image2|


.. |image1| image:: https://github.com/InternLM/xtuner/assets/67539920/abacb28f-5afd-46d0-91b2-acdd20887969
.. |image2| image:: https://github.com/InternLM/xtuner/assets/41630003/fc16387a-3c83-4015-9235-8ec811077953
