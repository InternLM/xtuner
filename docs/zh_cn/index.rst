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

   get_started/installation.rst
   get_started/quickstart.rst

.. toctree::
   :maxdepth: 2
   :caption: 准备

   preparation/pretrained_model.rst
   preparation/prompt_template.rst

.. toctree::
   :maxdepth: 2
   :caption: 训练

   training/open_source_dataset.rst
   training/custom_sft_dataset.rst
   training/custom_pretrain_dataset.rst
   training/multi_modal_dataset.rst
   acceleration/train_large_scale_dataset.rst
   training/modify_settings.rst
   training/visualization.rst

.. toctree::
   :maxdepth: 2
   :caption: DPO

   dpo/overview.md
   dpo/quick_start.md
   dpo/modify_settings.md

.. toctree::
   :maxdepth: 2
   :caption: Reward Model

   reward_model/overview.md
   reward_model/quick_start.md
   reward_model/modify_settings.md
   reward_model/preference_data.md

.. toctree::
   :maxdepth: 2
   :caption: 加速训练

   acceleration/deepspeed.rst
   acceleration/flash_attn.rst
   acceleration/varlen_flash_attn.rst
   acceleration/pack_to_max_length.rst
   acceleration/length_grouped_sampler.rst
   acceleration/train_extreme_long_sequence.rst
   acceleration/hyper_parameters.rst
   acceleration/benchmark.rst


.. toctree::
   :maxdepth: 1
   :caption: InternEvo 迁移

   internevo_migration/differences.rst
   internevo_migration/ftdp_dataset/tokenized_and_internlm2.rst
   internevo_migration/ftdp_dataset/processed_and_internlm2.rst
   internevo_migration/ftdp_dataset/processed_and_others.rst
   internevo_migration/ftdp_dataset/processed_normal_chat.rst
