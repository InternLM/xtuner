.. xtuner documentation master file, created by
   sphinx-quickstart on Tue Jan  9 16:33:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到 XTuner V1 的中文文档
==================================

.. figure:: ./_static/image/logo.png
  :align: center
  :alt: xtuner
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>LLM 一站式工具箱
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/InternLM/xtuner" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/InternLM/xtuner/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/InternLM/xtuner/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: 开始使用

   get_started/index.rst


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: 预训练与微调

   pretrain_sft/tutorial/index.rst

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: 强化学习

   rl/tutorial/rl_grpo_trainer.md

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: 进阶教程

   pretrain_sft/advanced_tutorial/index.rst
   rl/advanced_tutorial/index.rst

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Benchmark

   benchmark/index.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: 旧版文档

   legacy_index.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   Pretrain & SFT Trainer  <api/trainer>
   Config <api/config>
   RL Trainer <api/rl_trainer>
   RL Config <api/rl_config>
   Loss Context <api/loss_ctx>


写在前面
==================================
1111



食用指南
==================================
1111


致谢
==================================
本项目 RL（强化学习）部分的设计与部分实现，充分参考和借鉴了业界优秀的开源强化学习框架，包括 verl（https://github.com/volcengine/verl/）、slime（https://github.com/THUDM/slime）、AReaL（https://github.com/inclusionAI/AReaL）等。这些项目在 RL 算法优化、工程实现和社区生态等方面为我们提供了宝贵的经验和灵感。在此，向这些开源项目的开发者和社区表示衷心感谢！