RL Config
===================================

.. currentmodule:: xtuner.v1

Resource Config
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   rl.utils.AcceleratorResourcesConfig
   rl.utils.CPUResourcesConfig

Rollout Config
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   rl.rollout.worker.RolloutConfig

Agent Loop Config
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   rl.agent_loop.SingleTurnAgentLoopConfig
   rl.agent_loop_manager.AgentLoopManagerConfig
   rl.agent_loop_manager.TaskSpecConfig
   rl.agent_loop_manager.SamplerConfig
   rl.agent_loop_manager.SyncProduceStrategyConfig
   rl.agent_loop_manager.AsyncProduceStrategyConfig

Judger Config
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   rl.judger.JudgerConfig
   rl.judger.GSM8KJudgerConfig
   rl.judger.ComposedJudgerConfig

Replay and Evaluation Config
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   rl.replay_buffer.SyncReplayBufferConfig
   rl.replay_buffer.AsyncReplayBufferConfig
   rl.evaluator.EvaluatorConfig

Training and Loss Config
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   rl.trainer.WorkerConfig
   rl.loss.BaseRLLossConfig
   rl.loss.GRPOLossConfig
   rl.loss.OrealLossConfig
   rl.rollout_is.RolloutImportanceSampling
