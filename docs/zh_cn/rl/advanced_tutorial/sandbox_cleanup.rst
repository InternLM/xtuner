沙盒回收与异常退出恢复
======================

``SandboxPool`` 把阶段释放与 Trial 最终关闭分开。``release_all()`` 按依赖逆序
释放当前资源，保留 provider，允许 DeepSWE 等任务随后创建 verifier；
``Runner`` 在最终退出时调用 ``aclose()``。直接使用 Pool 的调用方也应在最终
``finally`` 中调用 ``aclose()``。原来依靠 ``release_all()`` 关闭独占 provider
的调用方需要迁移。覆写 ``release_all()`` 的子类仍会被最终关闭入口调用。

删除确认与失败交接
------------------

有 ``get(env_id)`` 的 provider 默认开启删除确认。DELETE 被接受后，只有 GET
返回 ``{"env": {"status": "deleted"}}``、``{"status": "deleted"}``，或携带
准确 HTTP 404 状态的异常才算完成。``failed``、``stopping``、``expired`` 等
状态不被自动解释为已释放；平台映射由 provider 负责。没有 GET 的旧 provider
保留原先 DELETE 返回即成功的语义，不能启用持久回收。

前台清理预算默认 120 秒。清理锁、取消保护和最终关闭另有少量宽限；
``aclose()`` 的外层上限为 ``cleanup_timeout_sec * 1.2 + 6`` 秒，最终化器
还可能使用最多一秒的取消宽限。单个 ID 卡住不会耗尽其他资源的全部预算。

若前台回收失败，dict 形式的 provider 配置会被用于创建新的管理连接，
默认再重试 60 秒；同一事件循环最多 8 个回收 worker。独立 provider 实例
可以通过 ``cleanup_provider=dict(type=..., ...)`` 提供等价的重建配置。
重建工厂必须返回新实例，不能返回仍被其他 Pool 使用的共享实例。
共享 provider 不会被关闭，其其他资源也不会被纳入回收。

没有重建配置时，失败的原管理连接不会被提前关闭；调用方仍可显式重试。
已移交持久账本或后台 worker 的 Pool 则拒绝再次操作，防止双重 owner。
后台成功只更新 ``sandbox_cleanup.deferred_result``，不会改变原 Trial
失败状态或恢复其训练资格。原始业务异常保留，清理结果单独记录。

启用持久账本与退出兜底
----------------------

为每次运行提供独立的、可信且所有相关进程可见的目录。不能共用其他运行的
目录，也不能把账本放进工作负载可写的沙盒挂载。文件系统必须支持跨节点
``flock``、原子 rename 和 fsync。部署前需确认实际挂载的锁语义。

Pool 配置示例：

.. code-block:: python

   pool = dict(
       type=SandboxPool,
       provider=dict(type=GatewayProvider, gateway_url=gateway_url),
       specs=specs,
       cleanup_dir=f"{run_dir}/sandbox-cleanup",
       cleanup_provider_key="gateway-prod",
       cleanup_retry_timeout_sec=60,
   )

使用训练级自动恢复时，必须在构建 trainer 前配置下面的环境变量。标准
AgentLoop actors 自动接收目录和 key；独立启动的自定义 workers 需要自行传递：

.. code-block:: bash

   export XTUNER_SANDBOX_CLEANUP_DIR="${RUN_DIR}/sandbox-cleanup"
   export XTUNER_SANDBOX_PROVIDER_KEY="gateway-prod"
   export XTUNER_SANDBOX_CLEANUP_FACTORY="sandbox_recovery:resolve_provider"
   export XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT=180

混合 backend/namespace 的任务应分别配置不同的 ``cleanup_provider_key``，
不能使用同一个全局 key 指向不同平台。工厂 key 与创建时的 endpoint、namespace
必须严格匹配。可信配置模块 ``sandbox_recovery.py`` 示例：

.. code-block:: python

   import os
   from lagent.serving.sandbox.providers.gateway import GatewayProvider

   def resolve_provider(key):
       if key != "gateway-prod":
           raise ValueError("Unknown cleanup backend")
       return GatewayProvider(os.environ["SANDBOX_GATEWAY_URL"])

工厂模块和 provider 依赖必须在恢复进程的 ``PYTHONPATH`` / Python 环境中可用。
需要凭据的平台应由工厂读取受信凭据文件或环境变量；账本不会保存 provider
配置、凭据、任务正文或响应内容。账本只记录 provider key、Trial UID、
资源 ID、创建歧义与删除确认状态。

两种 RL trainer 在公共初始化入口自动启动独立恢复进程并注册信号/退出钩子，
因此标准 CLI、CyberGym 自定义 ``cfg.build(); trainer.fit()`` 均可使用同一配置。
配置不完整或恢复工厂导入失败会在创建训练资源前报错。AgentLoop 的远程 actor
自动继承 driver 的账本目录和默认 backend key；不同 backend 仍须在 Pool 中显式
覆盖 key。恢复工厂仅在 driver / 恢复节点上加载，不向 workers 传输凭据。

恢复进程默认每 30 秒扫描已失去 owner 的账本。driver 正常退出或被 SIGKILL 后，
内核关闭监控管道，恢复进程再用最多 180 秒补扫，等待 worker 退出释放锁。
可用 ``XTUNER_SANDBOX_CLEANUP_INTERVAL`` 和
``XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT`` 调整。每轮使用有总时间限制的新解释器，
避免 SDK 卡住长期阻塞恢复。``supervisor.log`` 记录启动/运行错误；
``cleanup-report.json`` 记录各 ID 的确认情况。预算耗尽仍保留失败记录。

``fit()`` 无论成功、异常还是取消都会在原清理流程后补扫；退出钩子先
``ray.shutdown()`` 再补扫。默认不执行集群级 ``ray stop --force``，避免影响共享
Ray 集群中的其他作业。自定义无 trainer 入口可调用
``start_cleanup_supervisor()``，并在停止 workers 后调用
``cleanup_from_environment()`` 或执行下面的命令。

``ray.init()`` 会覆盖 SIGTERM handler，标准 CLI 和 trainer 在初始化后
重新安装处理器；重复 TERM/INT 不会中断已经开始的退出清理。

.. code-block:: bash

   python xtuner/v1/rl/agent_loop/sandbox_agent_loop/cleanup.py \
       --directory "${RUN_DIR}/sandbox-cleanup" \
       --factory sandbox_recovery:resolve_provider \
       --timeout 30 --concurrency 8

命令只处理该目录中 UUID 命名的账本，返回 ``cleanup-report.json``；
退出码 0 表示本轮全部确认，1 表示仍有活跃 owner、未知创建、损坏账本或
回收失败。不会按模板、名称前缀或全项目列表删除资源。重复执行是幂等的。
CLI 的 ``--timeout`` 是每份账本的预算；大量账本或 SDK/文件系统卡住时，
调用它的 supervisor 应像退出钩子一样设置总进程预算。

正常 Trial 从首次获取资源到最终关闭始终持有账本锁。独立回收器遇到锁
被占用就跳过，因此跨更新、保存 checkpoint、阶段间切换都不会触发误删。
进程被 SIGKILL 后，内核释放锁，新进程才可接手。完整 JSON 快照经 fsync
和原子替换写入，避免 JSONL 尾部损坏影响整轮恢复；单份坏账本不会阻断其他
账本。取消不会让尚未完成的写入越过解锁操作。

任务接入与边界
--------------

* 普通 SWE/TB2/ClawGym：所有 Pool 创建的已知 ID 使用同一流程。
* CyberGym main + task-server：保留依赖组的逆序回收；两个 ID 分别确认。
* infer + eval：同一 Pool 中各阶段均登记；部分创建失败仍释放已知成员。
* DeepSWE main → verifier：阶段之间调用 ``release_all()``，Trial 结束调用
  ``aclose()``，不要在阶段释放时关闭共享 provider。
* 外部/localhost 服务：只有 Pool 实际创建且拥有的资源属于回收范围。

这是已知 ID 的恢复机制。创建请求发出前先记录 intent；在返回 ID 前崩溃
会保留 ``creating`` / ``ambiguous_creates``，不会伪报回收成功，也不会自动
重新 POST。SDK 在内部已获得 ID、但尚未返回给 Pool 就崩溃时，同样可能只
留下 unknown；彻底覆盖该窗口需要 provider 在分配时持久登记，或平台支持
按幂等创建 token 查询。TTL 是额外保护，不在本补丁中缩短。

恢复进程能接管单独的 driver SIGKILL；若调度器同时杀掉整个作业（包括恢复进程）
或整机失联，必须由作业外的 scheduler epilogue / 运维恢复服务执行上述命令。
独立 session 不等于独立 cgroup，不能宣称作业内进程能覆盖整作业强杀。
未配置持久目录时，进程退出后的自动恢复不可用。后台重试预算耗尽后会保留
账本，需退出补扫或显式重试；本实现不提供无限重试服务或共享创建配额背压。
删除沙盒不代表模型请求已经取消，模型请求排空与 trace 结算保持原有逻辑。

CPU 验证
--------

.. code-block:: bash

   pytest -q --confcutdir=tests/rl/sandbox_cleanup tests/rl/sandbox_cleanup

也可直接执行轻依赖测试脚本：

.. code-block:: bash

   python tests/rl/sandbox_cleanup_cpu.py

需要 Python 3.11+、pydantic v2，以及本机回环端口权限。该测试加载实际
Pool、Runner、schema、finalizer、reconciler；跳过训练包初始化和可选的
导入辅助模块。管理请求使用真实 loopback TCP HTTP，覆盖删除确认、取消、
独立管理连接、部分创建、活跃 owner 保护、SIGKILL 后恢复、独立 CLI、
损坏账本隔离和顺序 verifier。它不替代真实 provider、跨节点文件系统锁、
Ray 退出集成或真实平台资源回落验收。

训练结果与并发影响
------------------

清理失败会将原本成功的 Trial 标为 FAILED，后续 failed-attempt 过滤可能改变
``ProduceBatchResult.rollout_states``、``leftover_failed`` / ``leftover_filtered``、
有效样本/token/reward 计数；清理等待也可能增加生成/生产耗时。结果 schema、
奖励计算本身和 RoutedExperts tensor/reference 所有权不变。Ray 的 generate
concurrency group 和容量不变，新增的周期扫描运行在独立进程；不占用 actor RPC
并发槽，也不通过 checkpoint / weight-update 暂停回收仍持锁的 live task。
