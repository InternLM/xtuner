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

   legacy/get_started/installation.rst
   legacy/get_started/quickstart.rst

.. toctree::
   :maxdepth: 2
   :caption: 准备

   legacy/preparation/pretrained_model.rst
   legacy/preparation/prompt_template.rst

.. toctree::
   :maxdepth: 2
   :caption: 训练

   legacy/training/open_source_dataset.rst
   legacy/training/custom_sft_dataset.rst
   legacy/training/custom_pretrain_dataset.rst
   legacy/training/multi_modal_dataset.rst
   legacy/acceleration/train_large_scale_dataset.rst
   legacy/training/modify_settings.rst
   legacy/training/visualization.rst

.. toctree::
   :maxdepth: 2
   :caption: DPO

   legacy/dpo/overview.md
   legacy/dpo/quick_start.md
   legacy/dpo/modify_settings.md

.. toctree::
   :maxdepth: 2
   :caption: Reward Model

   legacy/reward_model/overview.md
   legacy/reward_model/quick_start.md
   legacy/reward_model/modify_settings.md
   legacy/reward_model/preference_data.md

.. toctree::
   :maxdepth: 2
   :caption: 加速训练

   legacy/acceleration/deepspeed.rst
   legacy/acceleration/flash_attn.rst
   legacy/acceleration/varlen_flash_attn.rst
   legacy/acceleration/pack_to_max_length.rst
   legacy/acceleration/length_grouped_sampler.rst
   legacy/acceleration/train_extreme_long_sequence.rst
   legacy/acceleration/hyper_parameters.rst
   legacy/acceleration/benchmark.rst


.. toctree::
   :maxdepth: 1
   :caption: InternEvo 迁移

   legacy/internevo_migration/differences.rst
   legacy/internevo_migration/ftdp_dataset/tokenized_and_internlm2.rst
   legacy/internevo_migration/ftdp_dataset/processed_and_internlm2.rst
   legacy/internevo_migration/ftdp_dataset/processed_and_others.rst
   legacy/internevo_migration/ftdp_dataset/processed_normal_chat.rst
