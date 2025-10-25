.. xtuner documentation master file, created by
   sphinx-quickstart on Tue Jan  9 16:33:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |checked| unicode:: U+2713
.. |unchecked| unicode:: U+2717

Welcome to XTuner V1 English Documentation
==========================================

.. figure:: ./_static/image/logo.png
  :align: center
  :alt: xtuner
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>LLM One-Stop Toolbox
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
   :caption: Getting Started

   get_started/index.rst

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Pretraining & Fine-tuning

   pretrain_sft/tutorial/index.rst

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Reinforcement Learning

   rl/tutorial/rl_grpo_trainer.md

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Advanced Tutorial

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
   :caption: Legacy Documentation

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


XTuner V1 is a new generation large model training engine specifically designed for ultra-large-scale MoE models. Compared with traditional 3D parallel training architectures, XTuner V1 has been deeply optimized for the current mainstream MoE training scenarios in academia.

🚀 Speed Benchmark
==================================

.. figure:: ../assets/images/benchmark/benchmark.png
   :align: center
   :width: 90%

Core Features
=============
**📊 Dropless Training**

- **Flexible Scaling, No Complex Configuration:** 200B scale MoE without expert parallelism; 600B MoE only requires intra-node expert parallelism
- **Optimized Parallel Strategy:** Compared with traditional 3D parallel solutions, smaller expert parallel dimensions enable more efficient Dropless training

**📝 Long Sequence Support**

- **Memory Efficient Design:** Through advanced memory optimization technology combinations, 200B MoE models can train 64k sequence length without sequence parallelism
- **Flexible Extension Capability:** Full support for DeepSpeed Ulysses sequence parallelism, maximum sequence length can be linearly extended
- **Stable and Reliable:** Insensitive to expert load imbalance during long sequence training, maintaining stable performance

**⚡ Excellent Efficiency**

- **Ultra-Large Scale Support:** Supports MoE model training up to 1T parameters
- **Breakthrough Performance Bottleneck:** First time achieving FSDP training throughput surpassing traditional 3D parallel solutions on MoE models above 200B scale
- **Hardware Optimization:** Training efficiency surpasses NVIDIA H800 on Ascend A3 NPU supernodes


.. figure:: ../assets/images/benchmark/structure.png
   :align: center
   :width: 90%
   :alt: Performance comparison


🔥 Roadmap
==========

XTuner V1 is committed to continuously improving the pretraining, instruction fine-tuning, and reinforcement learning training efficiency of ultra-large-scale MoE models, with a focus on optimizing Ascend NPU support.

🚀 Training Engine
-----------

Our vision is to build XTuner V1 into a universal training backend that seamlessly integrates into a broader open-source ecosystem.

+------------+-----------+----------+-----------+
|   Model    |  GPU(FP8) | GPU(BF16)| NPU(BF16) |
+============+===========+==========+===========+
| Intern S1  |    ✅     |    ✅    |    ✅     |
+------------+-----------+----------+-----------+
| Intern VL  |    ✅     |    ✅    |    ✅     |
+------------+-----------+----------+-----------+
| Qwen3 Dense|    ✅     |    ✅    |    ✅     |
+------------+-----------+----------+-----------+
| Qwen3 MoE  |    ✅     |    ✅    |    ✅     |
+------------+-----------+----------+-----------+
| GPT OSS    |    ✅     |    ✅    |    ❌     |
+------------+-----------+----------+-----------+
| Deepseek V3|    ✅     |    ✅    |    ❌     |
+------------+-----------+----------+-----------+
| KIMI K2    |    ✅     |    ✅    |    ❌     |
+------------+-----------+----------+-----------+


🧠 Algorithm Suite
-----------

Algorithm components are rapidly iterating. Community contributions are welcome - use XTuner V1 to scale your algorithms to unprecedented scales!

**Implemented**

- ✅ **Multimodal Pretraining** - Full support for vision-language model training
- ✅ **Multimodal Supervised Fine-tuning** - Optimized for instruction following
- ✅ `GRPO <https://arxiv.org/pdf/2402.03300>`_ - Group Relative Policy Optimization

**Coming Soon**

- 🔄 `MPO <https://arxiv.org/pdf/2411.10442>`_ - Mixed Preference Optimization
- 🔄 `DAPO <https://arxiv.org/pdf/2503.14476>`_ - Dynamic Sampling Policy Optimization
- 🔄 **Multi-round Agent Reinforcement Learning** - Advanced agent training capabilities


⚡ Inference Engine Integration
---------------

Seamless integration with mainstream inference frameworks

* |checked| LMDeploy
* |unchecked| vLLM
* |unchecked| SGLang



🤝 Contribution Guidelines
-----------

We thank all contributors for their efforts to improve and enhance XTuner. Please refer to the `Contribution Guidelines <.github/CONTRIBUTING.md>`_ to understand the relevant guidelines for participating in the project.

🙏 Acknowledgments
-----------

The development of XTuner V1 is deeply inspired and supported by excellent projects in the open-source community. We extend our sincere gratitude to the following pioneering projects:

**Training Engines:**

- [Torchtitan](https://github.com/pytorch/torchtitan) - PyTorch native distributed training framework
- [Deepspeed](https://github.com/deepspeedai/DeepSpeed) - Microsoft deep learning optimization library
- [MindSpeed](https://gitee.com/ascend/MindSpeed) - Ascend high-performance training acceleration library
- [Megatron](https://github.com/NVIDIA/Megatron-LM) - NVIDIA large-scale Transformer training framework


**Reinforcement Learning:**

XTuner V1's reinforcement learning capabilities draw on the excellent practices and experience of the following projects

- [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
- [SLIME](https://github.com/THUDM/slime) - THU's scalable RLHF implementation
- [AReal](https://github.com/inclusionAI/AReaL) - Ant Reasoning Reinforcement Learning for LLMs
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray

We sincerely thank all contributors and maintainers of these projects for their continuous advancement of the large-scale model training field.


🖊️ Citation
-----------

.. code-block:: bibtex

   @misc{2023xtuner,
       title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
       author={XTuner Contributors},
       howpublished = {\url{https://github.com/InternLM/xtuner}},
       year={2023}
   }

Open Source License
==========

This project adopts the `Apache License 2.0 Open Source License <LICENSE>`_. At the same time, please comply with the licenses of the models and datasets used.