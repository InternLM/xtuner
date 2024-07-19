.. xtuner documentation master file, created by
   sphinx-quickstart on Tue Jan  9 16:33:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XTuner's documentation!
==================================

.. figure:: ./_static/image/logo.png
  :align: center
  :alt: xtuner
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>All-IN-ONE toolbox for LLM
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/InternLM/xtuner" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/InternLM/xtuner/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/InternLM/xtuner/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>



Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Get Started

   get_started/overview.md
   get_started/installation.md
   get_started/quickstart.md

.. toctree::
   :maxdepth: 2
   :caption: Preparation

   preparation/pretrained_model.rst
   preparation/prompt_template.rst

.. toctree::
   :maxdepth: 2
   :caption: Training

   training/modify_settings.rst
   training/custom_sft_dataset.rst
   training/custom_pretrain_dataset.rst
   training/custom_agent_dataset.rst
   training/multi_modal_dataset.rst
   training/open_source_dataset.rst
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
   :caption: Acceleration

   acceleration/deepspeed.rst
   acceleration/pack_to_max_length.rst
   acceleration/flash_attn.rst
   acceleration/varlen_flash_attn.rst
   acceleration/hyper_parameters.rst
   acceleration/length_grouped_sampler.rst
   acceleration/train_large_scale_dataset.rst
   acceleration/train_extreme_long_sequence.rst
   acceleration/benchmark.rst

.. toctree::
   :maxdepth: 2
   :caption: Chat

   chat/llm.md
   chat/agent.md
   chat/vlm.md
   chat/lmdeploy.md

.. toctree::
   :maxdepth: 2
   :caption: Evaluation

   evaluation/hook.md
   evaluation/mmlu.md
   evaluation/mmbench.md
   evaluation/opencompass.md

.. toctree::
   :maxdepth: 2
   :caption: Models

   models/supported.md

.. toctree::
   :maxdepth: 2
   :caption: InternEvo Migration

   internevo_migration/internevo_migration.rst
   internevo_migration/ftdp_dataset/ftdp.rst
   internevo_migration/ftdp_dataset/Case1.rst
   internevo_migration/ftdp_dataset/Case2.rst
   internevo_migration/ftdp_dataset/Case3.rst
   internevo_migration/ftdp_dataset/Case4.rst
