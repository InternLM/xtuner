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

   get_started/installation.md
   get_started/quickstart.md

.. toctree::
   :maxdepth: 2
   :caption: Preparation

   preparation/pretrained_model.md
   preparation/prompt_template.md

.. toctree::
   :maxdepth: 2
   :caption: Training

   training/modify_settings.md
   training/custom_sft_dataset.md
   training/custom_pretrain_dataset.md
   training/custom_agent_dataset.md
   training/multi_modal_dataset.md
   training/open_source_dataset.md
   training/accelerate.md

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
