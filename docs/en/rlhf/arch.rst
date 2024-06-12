.. _xtuner_rlhf_arch:

System Architecture
-------------------

The architecture of XTuner-RLHF is shown as follows:

.. image:: images/arch_en.svg
   :alt: XTuner-RLHF Architecture

Algorithm Layer
~~~~~~~~~~~~~~~

The algorithm layer implements various reinforcement learning algorithms and environments, i.e., specific training strategies and application scenarios. This includes various reinforcement learning algorithms such as PPO and KTO, as well as different task environments such as Q&A (question and answer) and LR (logical reasoning).

Coordination Layer
~~~~~~~~~~~~~~~~~~~~

The coordination layer provides model-level operation interfaces to the upper algorithm layer, simplifying interactions with the underlying engines. It also adapts to different training and inference frameworks and models, managing and scheduling multiple model resources to ensure efficient system operation.

Engine Layer
~~~~~~~~~~~~

The engine layer decouples training, inference, and generation, allowing users to choose different engines for these processes. For example, transformers can be used for training and inference, while the vLLM can be used for generation. The advantages of a multi-engine design include:

**Flexibility and Adaptability**: Different projects may have different requirements and constraints. Integrating multiple frameworks allows users to choose the most suitable tool for their specific situation, enhancing development efficiency and effectiveness.

**Performance Optimization**: Different frameworks may perform differently on various tasks. Users can choose the framework that performs best for a specific task to achieve optimal performance.

**Cross-Platform Compatibility**: Some frameworks perform better on specific platforms or support specific hardware. Providing multiple framework options ensures compatibility and optimization across different platforms and hardware.

**Ease of Use**: Some frameworks may be more user-friendly and suitable for rapid prototyping, while others may be better suited for large-scale deployment. Users can choose the appropriate framework based on the development stage.

Distributed Computing Framework: Ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a project incubated within InternLM, the RLHF system has adopted Ray as the distributed framework for efficient training, inference, and generation since the project began in May 2023, empowering the system with the following features:

**Abstraction of Underlying Cluster Differences**: Ray provides an abstraction layer, allowing users to ignore the details of the underlying hardware. Whether using local clusters or cloud resources, Ray can uniformly manage and schedule tasks, simplifying the development and deployment process.

**Efficient Resource Management**: Resources such as CPU and GPU can be dynamically allocated according to task requirements, ensuring efficient use of computational resources and improving overall system performance.

**Scalability**: Ray can easily scale to large clusters, supporting hundreds or even thousands of nodes. This allows the system to scale horizontally to meet the demands of large-scale data processing and computation.

**Flexible Task Scheduling**: Ray can schedule tasks flexibly based on priority and resource requirements, optimizing task execution order, reducing task waiting time, and improving system throughput.

**Automated Fault Recovery**: Ray has built-in fault tolerance mechanisms that can detect and attempt to recover failed tasks, enhancing the stability and reliability of the system while reducing the need for manual intervention.

Acknowledgements
~~~~~~~~~~~~~~~~~

In our journey of exploring and implementing the RLHF system, we have been fortunate to witness many outstanding open-source projects that shine like brilliant stars, illuminating our path forward. For example:

- `ColossalChat <https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalChat/coati/ray#detach-experience-makers-and-trainers>`_: Ingeniously utilizes Ray to implement distributed PPO, distributing trainers and experience makers across different nodes, enhancing computational efficiency.
- `ATorch <https://github.com/intelligent-machine-learning/dlrover/tree/master/atorch>`_: Adopts an innovative design of "training-decoding decoupling + high-performance inference backend," compatible with the open-source vLLM engine as the inference backend, supporting efficient fine-tuning of trillion-scale models.
- `OpenRLHF <https://github.com/OpenLLMAI/OpenRLHF>`_: A concise, easy-to-use and open-source spirited RLHF training framework that leverages open-source projects such as Ray, DeepSpeed, vLLM, and HF Transformers to implement high-performance PPO and other algorithms.

We hold deep respect and gratitude for the developers in the open-source community. They have not only shared valuable knowledge and experience but also fostered the prosperity and development of the large model RLHF system ecosystem with an open mindset. We believe that it is this selfless spirit of sharing that makes our community stronger and technological progress faster. We thank every contributor once again; it is your efforts that make this world a better place.