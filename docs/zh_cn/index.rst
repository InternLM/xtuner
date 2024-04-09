.. xtuner documentation master file, created by
   sphinx-quickstart on Tue Jan  9 16:33:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到 XTuner 的中文文档
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




文档
-------------
.. toctree::
   :maxdepth: 2
   :caption: 开始使用

   get_started/overview.rst
   get_started/installation.rst
   get_started/quickstart.rst

.. toctree::
   :maxdepth: 2
   :caption: 准备

   preparation/pretrained_model.rst
   preparation/custom_sft_dataset.md
   preparation/custom_pretrain_dataset.md
   preparation/custom_agent_dataset.md
   preparation/open_source_dataset.md

.. toctree::
   :maxdepth: 2
   :caption: 训练

   training/modify_settings.md
   training/custom_sft_dataset.md
   training/custom_pretrain_dataset.rst
   training/custom_agent_dataset.md
   training/multi_modal_dataset.rst
   training/open_source_dataset.md
   training/train_on_large_scale_dataset.rst
   training/accelerate.md

.. toctree::
   :maxdepth: 2
   :caption: 对话

   chat/llm.md
   chat/agent.md
   chat/vlm.md
   chat/lmdeploy.md

.. toctree::
   :maxdepth: 2
   :caption: 评测

   evaluation/hook.md
   evaluation/mmlu.md
   evaluation/mmbench.md
   evaluation/opencompass.md

.. toctree::
   :maxdepth: 2
   :caption: 模型

   models/supported.md

.. toctree::
   :maxdepth: 2
   :caption: InternEvo 迁移

   internevo_migration/internevo_migration.rst
   internevo_migration/ftdp_dataset/README.md
   internevo_migration/ftdp_dataset/Case1.rst
   internevo_migration/ftdp_dataset/Case2.rst
   internevo_migration/ftdp_dataset/Case3.rst
   internevo_migration/ftdp_dataset/Case4.rst
