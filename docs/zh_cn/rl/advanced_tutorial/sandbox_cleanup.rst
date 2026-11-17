沙盒退出清理与失败回收
====================

退出主流程沿用 Miles 的思路：停止新资源申请，排空任务，由退出方使用新管理
连接在有限预算内清理本次运行的资源，并确认删除结果。未完成的清理再交给
受限并发、带退避的后台回收器。正常训练期间不周期扫描全部沙盒账本。

Ray actor 保留任务调度、模型服务与训练 trace 的所有权，每条沙盒任务的
Runner、推理和验证在独立 CPU 子进程中执行。子进程使用原 HTTP 模型代理，
不加载模型、不初始化 Ray、不传输模型张量或 routed-experts 对象。

单任务进程隔离
--------------

训练 ``AgentInSandboxLoop`` 与 DeepSWE、TB 的本地 runner 入口统一调用
``run_trial_process(item)``。pipeline 必须是可信配置 dict 或可导入路径，
不接受已构造的 Runner 或在配置中传入活跃 Pool。Python class、Path 和
bytes 使用可信本地序列化保留，任务内容和凭据不放入命令行。隔离入口要求
Pool 使用下述持久目录和恢复工厂，否则在申请资源前失败。

每份 Pool 账本在允许资源创建前，原子登记到该子进程的私有清单。任务取消、
运行停止或超时后，父进程先发送 TERM，超时仅 KILL 该任务的进程组并等待
根进程退出，再检查清单中的精确账本。未确认退出时不接管资源。未完成的
账本先批量持久交接到原有后台回收器，再尝试有预算的前台回收；不停止整个
run、不扫描或删除其他任务的资源。原业务错误和取消保持优先，后台删除成功
不恢复失败样本的训练资格。

子进程通过 Linux parent-death signal 在其父 actor 被杀后退出；driver 死亡
则继续由运行级观察器停止准入并接管。仅根进程能直接 wait，其他本地辅助
进程会收到进程组 KILL；显式继承的存活账本租约仍阻止接管。全部作业内
恢复进程一起消失时，依然需要作业外恢复。

任务执行总期限 ``XTUNER_SANDBOX_TRIAL_TIMEOUT`` 默认 14400 秒，包含启动；
TERM 宽限 ``XTUNER_SANDBOX_TRIAL_STOP_TIMEOUT`` 默认 120 秒，随后等待 KILL
退出最多 5 秒；前台恢复及交接 ``XTUNER_SANDBOX_TRIAL_RECOVERY_TIMEOUT``
默认 30 秒。最终化外层预算为 TERM 宽限加恢复预算加 10 秒，另有原有取消
宽限。既有 stage 期限、Pool 删除期限和后台重试参数不变。创建许可仍向父
actor 的共享限流器申请，避免每个子进程重新获得 burst；不同 actor 的限流
作用域与此前一致。子进程继承运行环境、cwd、Python 导入路径及原 session
身份，并初始化自己的 span 日志；冷启动需要导入 recipe/Runner 依赖。

直接调用 ``Runner.run`` / ``run_with_pool`` 是进程内低层接口，供子进程、
测试及自行管理生命周期的调用方使用，不自动再套一层进程。

阶段释放与最终关闭
------------------

``SandboxPool.release_all()`` 按依赖逆序释放当前资源，保留 provider，允许
后续 verifier 阶段继续申请资源。公共 ``Runner`` 在最终 ``finally`` 中调用
``aclose()``。直接使用 Pool 的调用方也应在最终退出时调用 ``aclose()``。
覆写 ``release_all()`` 的子类仍会被最终关闭入口调用；原来在阶段释放中关闭
provider 的子类必须适配，不能直接假定兼容。

声明 ``supports_deletion_confirmation = True`` 的 provider 默认开启删除确认；
该能力要求 ``get(env_id)`` 提供规范化的删除证据，仅存在普通状态查询方法不会
自动开启。已符合该契约但尚未声明能力的 adapter 可显式设置
``verify_deletion=True``。DELETE 被接受后，必须 GET
返回 ``{"env": {"status": "deleted"}}``、``{"status": "deleted"}``，或得到
准确 HTTP 404，才算回收完成。``failed``、``stopping``、``expired`` 不自动
视为资源已释放；平台状态映射由 provider 负责。未声明能力的旧 provider
默认保留 DELETE 返回即成功的旧语义，不能启用持久回收。显式设置
``verify_deletion=False`` 同样只支持非持久模式；其进程内后台重试沿用前台策略，
结果标记 ``deletion_confirmed=False``，不会写成持久删除确认。

Pool 前台清理预算默认 120 秒，单个 ID 卡住不会耗尽其他成员的全部预算。
``aclose()`` 外层上限为 ``cleanup_timeout_sec * 1.2 + 6`` 秒，最终化器另有
最多一秒取消宽限。失败会保留原始业务异常，并使原本成功的 Trial 标为失败。
后台回收成功不能恢复该样本的训练资格。

持久账本配置
------------

每次运行使用独立、可信且所有相关进程可见的目录，包括恢复 checkpoint 的
新训练调用。文件系统必须支持 Linux 64 位 OFD 记录锁的跨节点互斥、原子
rename/hardlink 和 fsync；部署前应验证实际挂载，不能根据文件系统名称推断。
BSD ``flock`` 在当前 GPFS 挂载上实测只在节点内有效，因此不再用于清理。
不要把目录放入工作负载可写的沙盒挂载中。

锁文件先写入协议标记，再原子发布，已存在的锁 inode 永不替换。缺少 OFD
能力、无法观察记录锁，或锁文件缺少当前协议时，均保留资源并报错，不回退到
较弱锁。旧版空锁文件的目录不能与新版混用，也不能自动认定已无 owner；
须先在作业外确认所有持有者停止，再用匹配旧版本的工具恢复。局部能力探测
不是跨节点验收；共享锁探测允许返回其他合法读锁持有者的 PID。

在构建 trainer 前配置：

.. code-block:: bash

   export XTUNER_SANDBOX_CLEANUP_DIR="${RUN_DIR}/sandbox-cleanup"
   export XTUNER_SANDBOX_PROVIDER_KEY="gateway-prod"
   export XTUNER_SANDBOX_CLEANUP_FACTORY="sandbox_recovery:resolve_provider"
   export XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT=180

Pool 也可以显式设置 ``cleanup_provider_key``；混合 backend/namespace 的任务
必须使用能区分这些配置的 key。已配置运行级目录时，Pool 不能另设不同目录。
可信配置模块示例：

.. code-block:: python

   import os
   from lagent.serving.sandbox.providers.gateway import GatewayProvider

   def resolve_provider(key):
       if key != "gateway-prod":
           raise ValueError("Unknown cleanup backend")
       return GatewayProvider(os.environ["SANDBOX_GATEWAY_URL"])

工厂必须返回新的管理连接，并严格匹配创建时的 endpoint、namespace 和身份。
工厂模块及依赖必须在 driver 和可能交接失败任务的 worker Python 环境中可用；
凭据由工厂从受信配置读取。标准 AgentLoop 自动传递目录、key、工厂路径和
重试参数，但不序列化传输凭据。自定义 workers 需要自行传递配置。

账本只记录 provider key、Trial UID、资源 ID、创建 intent 和删除确认状态，
不保存凭据、provider 配置或任务正文。完整 JSON 经 fsync 和原子替换持久化。
Pool 从创建到最终关闭始终持有账本锁，阶段间切换、checkpoint 和权重更新
都不会使其成为可回收对象。恢复进程跳过仍持锁的 Pool；进程死亡后内核释放
锁，恢复方才能接管。fork 辅助进程不会继续持有原 owner 的锁。

首次账本登记在线程中执行，避免共享存储锁等待或 fsync 阻塞 Ray actor 事件
循环。只有调用方接收已完成的账本后才允许 POST；登记取消或超时时，由原
线程在初始化完成后关闭未使用的空账本，晚到的锁不会遗失。

参数校验与创建限流等待在创建 intent 之前完成；这两个阶段取消不会留下未知
创建记录。通过准入后才调用 provider，已发出请求的未知结果仍保守保留。

配套 OpenSandbox provider 使用线程执行 POST。创建期间收到取消后，由独立的
补偿任务继续接收响应、登记 ID 并执行回滚；调用方重复取消不会取消该补偿任务。
补偿结束后继续向上传播原始取消，实际删除确认仍由 Pool 的最终清理检查。

支持早期登记的 provider 显式声明 ``supports_allocation_callback = True``，
并接受单次 ``create(..., on_allocated=...)`` 异步回调。取得有效资源 ID 后、
等待平台就绪前必须等待回调完成；回调将 ID 登记与清除 ``creating`` 一次原子
持久化。回调失败应回滚已分配资源，不能继续初始化。DBox 和配套 OpenSandbox
provider 已接入，因此 owner 在 readiness 等待期间死亡也能按 ID 恢复。
旧 provider 不会收到额外参数，仍在返回 client/ID 后登记；未接入回调的 provider
不能获得 readiness 期间崩溃的同等保证。不要在 TypeError 后重试创建 POST。

退出主流程
----------

两种 RL trainer 的 ``fit()`` 收尾以及 SIGINT/SIGTERM 处理使用同一顺序：

#. 在 ``admission.lock`` 下写入永久停止标记，关闭新 Pool 和创建请求的准入。
#. 执行原有 producer / Ray 退出流程；Runner 观察停止标记并协作取消任务。
#. 启动新解释器，获取 ``run-cleanup.lock``，用新管理连接进行一次退出补扫。
#. 全部确认则记录完成；失败、超时或仍有活跃 owner，则持久交接后台回收器。

信号和 atexit 清理仅由注册它们的进程执行；fork 辅助进程收到 SIGTERM/SIGINT
仍会退出，但不会写父训练的停止标记或修改父 run 的结果。普通非沙盒 RL
导入这些生命周期入口不要求安装 Lagent；沙盒相关类按需加载。

信号/atexit 路径在同步关闭 Ray 前启动独立退出守护。Ray 正常返回时通过管道
取消守护；超过 ``XTUNER_SANDBOX_RAY_SHUTDOWN_TIMEOUT``（默认 30 秒）或
管道因 driver 退出而关闭时，守护执行同一退出补扫并交接失败项。等待阶段
不扫描账本，也不读写共享存储；即使 driver 卡在 Ray 关闭中仍可回收无 owner
的资源。活跃 Pool 仍受存活锁保护，不强制终止 Ray 或共享 actor。
守护启动失败时，先尝试有预算的补扫与失败交接，再进入 Ray 关闭。
此机制保证回收可独立推进，不保证阻塞的 Ray 调用本身返回。

退出补扫及失败交接共用默认 180 秒的总预算，预留最多 5 秒（总预算的 20% 上限）
用于失败交接。driver、存活 peer、后台回收器和手工 CLI
使用同一个运行级清理锁，避免多方同时补扫；每个 Pool 内仍按依赖逆序处理。
已获准的创建请求可能在停止后返回，其 ID 仍须登记并回滚。创建 intent 与
停止标记的登记共用准入锁，避免停止后漏记新资源。只要还有活跃 Pool 或未知
创建，退出补扫就不能宣布清理完成。停止目录不能复用于下一次训练。

Runner 的观察器不占 actor RPC 并发槽；即使 hook 吞掉取消后返回，也不会把
结果当作成功样本。隔离入口会在宽限耗尽后终止该任务子进程；进程内低层
Runner 若持续阻塞，仍须由调用方终止其持有者。存活 Pool 的锁始终阻止后台
接管，不强杀共享 actor，也不执行集群级 ``ray stop --force``。

driver 意外死亡的接管
--------------------

trainer 初始化会启动一个轻量退出观察进程，检查 driver 的内核 lease。
它在 driver 存活时不调用沙盒管理 API、不扫描资源；正常退出由 trainer
执行上述主流程。driver 被 SIGKILL 后，存活观察进程关闭准入并执行同一
退出补扫，失败同样交接后台回收器。配置不完整、重试参数无效或工厂不可导入
会在初始化阶段报错。

观察进程检测到 driver 退出后，直接进入有总预算的清理入口；关闭创建准入、
等待准入锁和读写退出结果均在可终止的子进程内完成。存活 worker 长时间占有
准入锁时，不会让观察进程在预算外无限等待；补扫超时后仍尝试交接失败请求。

driver 租约探测发生 ``OSError`` 时，观察进程将状态视为未知，按 0.2 秒起、
最多 5 秒的间隔退避重试；恢复读取前不触发删除。重复错误仅在次数为 2 的
整数次幂时记录，避免日志持续增长。正常读取恢复后重置退避。

设置 ``XTUNER_SANDBOX_CLEANUP_PEERS=1`` 后，标准 AgentLoop actors 也启动
独立观察进程，供 head 失效后接管。它们观察的是 driver lease，actor 自己
退出不触发全局停止。多方退出补扫由 ``run-cleanup.lock`` 串行化，默认不开启
peer，以免在没有恢复环境和凭据的节点启动额外进程。

具有同一共享目录和恢复环境的作业外节点，也可以在 driver 注册后执行：

.. code-block:: bash

   python xtuner/v1/rl/agent_loop/sandbox_agent_loop/supervisor.py

独立 session 不等于独立 cgroup。如果调度器杀掉整个作业及其观察/回收进程，
或者所有节点失联，仍需作业外 peer、scheduler epilogue 或运维服务接管。
本补丁不部署常驻的作业外服务。

失败后的后台回收
----------------

前台 Pool 清理失败交接对应账本；退出补扫失败交接本次停止运行。请求以不可变
文件持久写入 ``recovery-requests/`` 后，启动或通知本 run 唯一的后台回收器。
``recovery-worker.lock`` 保证单实例。入队与回收器退休共用短锁；启动时在
``recovery-batches/`` 固定本代请求及已经接管过的请求清单。正常成功路径
不启动回收器。

本代结束或达到硬超时后，守护先终止并回收旧 worker 进程组，再由有预算的
退休子进程检查新请求。新到请求交给继任守护，继承同一内核租约，并使用独立
进程组；无新请求时在队列锁内释放租约。自动继任只处理尚未接管的新请求，
不会使之前耗尽的请求无限重试。整个执行链空闲后，新的显式交接仍可重启
保留的失败请求。正常退出和预算耗尽采用同一交接协议。

每代同时冻结实际可执行删除的账本集合及祖先的预算归属。新退出请求即使面向
整个 run，也不能重新删除祖先代已接管的账本。退出完成判定另外只读观察全 run：
存活 owner、未决创建、无效账本或未确认资源都会保留未完成状态，观察本身不
触发额外 DELETE。旧格式批次仅在仍有完整 journal 请求证明时升级；缺失祖先
请求或旧 exit 请求无法还原历史范围时保留队列，等待空闲后的显式恢复。

退出代入队时只登记待封存范围；有硬预算的恢复子进程先成功关闭创建准入，
再在运行级清理锁内一次性封存当前账本，排除祖先已经接管的范围。封存前失败
可以在本代剩余轮次重试，封存后不再扩大删除范围。无法证明先停止准入的旧
exit 批次不自动继续删除，保留队列等待显式恢复。这样原退出补扫未能建立
停止标记时，后台仍能纳入停止前新登记的账本。

失败交接在可终止的子进程中完成，覆盖队列锁等待、持久写入和回收器启动；
队列锁使用非阻塞重试及截止时间。Pool 单项交接默认最多 5 秒，退出交接使用
退出总预算的剩余时间。交接超时或启动失败保留原账本；已写入的请求也保留。
调用方记录交接未确认，不把它当作已删除，也不无限等待共享存储恢复。

获取沙盒时，如果健康检查或准备失败，只有上一组资源的回滚删除及连接释放
完成后才允许内部重试。删除失败或未确认时保留原始获取异常，Runner 最终
关闭继续重试和交接，避免在旧资源未回收时继续创建下一组。

损坏的队列请求逐项隔离：保留原文件，在本轮结果的 ``invalid_requests`` 中
记录文件名及错误类型，其他有效请求继续执行并单独确认。即使退出补扫成功，
也不删除无效请求；该轮返回部分失败，重试仍受总轮数和预算限制。

回收器只处理显式交接的请求，不在正常训练中扫描未入队的孤儿账本。每轮使用
有硬超时的新解释器和新 provider，成功记录立即确认，剩余工作退避后重试。
请求按唯一文件名确认删除，不会误删稍后入队的同类请求。默认参数：

.. list-table:: 后台恢复参数
   :header-rows: 1

   * - 环境变量
     - 默认值
     - 含义
   * - ``XTUNER_SANDBOX_RECOVERY_CONCURRENCY``
     - 8
     - 同时恢复的账本数，每份账本内串行删除
   * - ``XTUNER_SANDBOX_RECOVERY_ATTEMPTS``
     - 6
     - 每代固定请求的最大轮数
   * - ``XTUNER_SANDBOX_RECOVERY_BACKOFF``
     - 2
     - 首次重试等待秒数，之后指数增长
   * - ``XTUNER_SANDBOX_RECOVERY_MAX_BACKOFF``
     - 60
     - 退避上限秒数；实际等待乘 0.8～1.0 随机抖动
   * - ``XTUNER_SANDBOX_RECOVERY_ATTEMPT_TIMEOUT``
     - 30
     - 每轮恢复子进程的硬超时秒数
   * - ``XTUNER_SANDBOX_RECOVERY_BUDGET``
     - 600
     - 每代固定请求及退休交接的总预算秒数

退出补扫也使用上述并发上限。此上限约束运行级补扫/后台恢复，不限制正常
任务自行释放资源的并发。回收器由独立守护进程约束整体寿命，预算同时覆盖
队列访问、状态持久化和每轮清理子进程。其中预留最多 5 秒、至多本代预算的一半
用于退休交接。守护进程不读写共享状态或日志；超时终止回收器的独立进程组，
再执行有预算的交接。该进程组不包含 Ray actor。若退休本身因共享存储卡住或
继任启动失败而不能完成，队列和账本仍保留，需要后续显式交接或外部恢复。

正常重试耗尽时保留请求和账本，记录 ``exhausted``。如果状态写入本身阻塞而
被守护进程终止，状态文件可能停留在上一轮，不能将该文件视为进程存活依据；
后续退出接管、显式入队或运维执行恢复命令可继续处理。本代运行期间的新请求
进入下一代，使用独立的有限预算；本代已耗尽的请求不会随之重新获得预算。
这不是无限重试服务，也未提供创建配额背压。

未配置持久恢复工厂时，Pool 保留兼容的进程内恢复：dict provider 配置或显式
``cleanup_provider`` 创建新连接，同一事件循环最多 8 个 worker，默认重试
60 秒。无重建配置时保留失败的原连接供显式重试。进程内模式不能覆盖进程死亡。

结果与运维恢复
--------------

``cleanup-report.json`` 保存一轮逐账本结果。``run-cleanup.json`` 保存运行级
状态：``training_outcome``、``cleanup_confirmed``、``fit_cleanup_confirmed``、
``finalization_failed``、``success``。只有训练成功且显式收尾确认清理，才算
成功；训练成功但清理未确认会抛 ``CleanupError``，已有训练异常则保留原异常。
后台后来回收成功只改变资源清理状态，不能将原先失败的训练改写为成功。

清理子进程只记录资源确认，不能提前发布 ``fit_cleanup_confirmed``。driver
只有观察到该子进程成功退出，才用退出总预算内的独立步骤记录 fit ACK；子进程
持久化资源确认后仍超时或异常退出时，fit 仍失败，持久报告也不会出现成功 ACK。
ACK 是对 driver 已确认事实的报告：报告步骤本身超时或失败只告警，不把已经
确认清理的 fit 改成失败。ACK 写入前超时可能保留保守的非成功报告，写入后超时
可以保留成功报告；这两种情况 fit 均按已确认的资源清理结果返回。

``supervisor.log`` 记录退出观察进程输出；``recovery.log`` 记录后台恢复输出；
``recovery-status.json`` 记录 PID、阶段、轮次历史、时间和退避信息。恢复失败
时保留队列和未确认资源 ID，以便后续继续处理。

手工执行一次补扫：

.. code-block:: bash

   python xtuner/v1/rl/agent_loop/sandbox_agent_loop/cleanup.py \
       --directory "${RUN_DIR}/sandbox-cleanup" \
       --factory sandbox_recovery:resolve_provider \
       --timeout 30 --concurrency 8

加 ``--finalize`` 会先关闭创建准入并更新运行状态；``--recover-queued`` 仅处理
已交接请求，成功后确认队列。手工 CLI 不自动启动后台回收器；训练退出 wrapper
负责失败交接。``--timeout`` 是每份账本的预算，``--budget`` 约束清理锁等待，
调用方仍须像训练 wrapper 一样设置总进程超时。运维恢复不要传入
``--training-outcome succeeded``，因为恢复本身不是训练成功证据。

命令只处理本 run 的 UUID 账本和其中的确切资源 ID，不按模板或项目全量删除。
退出码 0 表示此次全部确认，1 表示仍有未完成资源；损坏账本隔离处理。
自定义无 trainer 的入口可使用 ``start_cleanup_supervisor()``、
``request_stop_from_environment()`` 和 ``finish_training(outcome)``。

任务接入与边界
--------------

公共 Pool/Runner 路径覆盖普通 SWE/TB2/ClawGym，以及 infer/eval 多阶段任务。
CyberGym 的 main/task-server 依赖组分别登记、逆序回收；其配套 ``CyberGymPool``
必须使用基类阶段释放，不能提前关闭 provider。``CYBERGYM_CLEANUP_TIMEOUT``
映射到基类清理预算，结果继续从 ``pool.cleanup`` 写入 ``cybergym_cleanup``。

配套 DeepSWE 已将最终关闭委托公共 ``aclose()``，并验证阶段切换及重复关闭。
其他自定义关闭逻辑仍需逐项检查。``LocalhostRunner``、绕过 Pool 直接申请的资源和外部共享服务
不自动纳入账本；只回收本次 Pool 实际创建并拥有的资源。

创建请求已发出但尚未返回 ID 就崩溃时，账本保留创建歧义，不报告清理成功，
也不自动重复 POST。彻底回收这一窗口需 provider 在分配时持久登记，或平台
支持按幂等 token 查询。TTL 是额外保护，本补丁不缩短 TTL。

验证与训练影响
--------------

.. code-block:: bash

   python tests/rl/sandbox_cleanup_cpu.py
   pytest -q --confcutdir=tests/rl/sandbox_cleanup tests/rl/sandbox_cleanup

测试需要 Python 3.11+、pydantic v2 和回环端口权限；加载实际 Pool、Runner、
finalizer、reconciler，使用真实本机 HTTP 和子进程，绕过训练依赖初始化。
覆盖删除确认、重复取消、创建竞态、SIGKILL 后接管、存活 Pool 保护、退出
补扫串行化、并发限制、失败队列单实例、退避/耗尽及成功路径无后台扫描。
这些测试不替代真实 Ray、多机文件系统锁或平台资源回落验收。

清理失败影响 failed-attempt 过滤，可能改变 ``ProduceBatchResult.rollout_states``、
``leftover_failed`` / ``leftover_filtered``、有效样本/token/reward 计数，并增加
生成耗时。结果 schema、奖励计算、RoutedExperts tensor/reference 所有权、Ray
concurrency group 和容量不变。恢复在独立进程中进行，不占 actor RPC 槽。


代码职责与扩展接口
------------------

``SandboxPool`` 保留依赖组装、健康检查和 provisioner 调度；唯一的
``SandboxLifecycle`` 对象负责资源归属、阶段释放、最终关闭、账本及恢复策略。
Pool 和 Runner 不读取账本数据结构，任务 recipe 不读取 Pool 的私有 provider
或 client 容器。原有 Pool 构造参数、``release_all()`` 和 ``aclose()`` 保持兼容。

需要在执行前启动进度观察等工作的 recipe，可调用
``Runner.prepare_trial_pool(item)`` 校验输入并构造应用本任务 overrides 的 Pool，
再调用 ``Runner.run_with_pool(item, pool)`` 执行标准阶段和最终清理。构造过程
深拷贝模板及 overrides，不修改任务对象；执行过程不重复应用 overrides。
若 recipe 在进入执行前设置失败，recipe 自己的作用域负责关闭已构造的 Pool。
直接调用执行接口时，输入校验也在清理作用域内进行，校验失败仍会关闭 Pool。

``pool.task_scope(trial_id)`` 统一登记任务、观察停止请求并在退出时调用公开
``aclose()``，因此子类的最终关闭逻辑仍会执行。``pool.peek(name)`` 只读取已存在的
client，不申请新资源；``pool.pending_resource_ids`` 返回不可变的 ID 快照。
这些观察接口不延长资源的存活时间，也不授权调用方单独关闭共享 provider。
公开 ``release_all()`` 和 ``aclose()`` 始终报告本次清理失败，不依赖调用方是否
正在处理其他异常。``task_scope()`` 只保留从当前任务体实际传出的主异常；
如果任务体已正常返回，清理失败仍向上传播。Runner 已记录的业务错误保留
优先级，清理错误另外存入报告，后台补删不会把失败样本改写为完成。
``pool.cleanup`` 每次返回独立的深拷贝快照；修改报告及其嵌套内容不会改变生命周期
或后续报告。调用方需要在相关释放或恢复完成后重新读取快照。

阶段内所有资源组及排空触发的回滚共享本次固定删除期限；阶段释放结束后，
该期限随之清除。后续 verifier 获取失败时，独立回滚使用本次预算，
不会继承 infer 阶段已经过期的截止时间，也不会给同一阶段的每个资源组重新续时。

Runner 类通过一次 ``_run_sandbox_trial`` 模块操作接入清理。作用域入场、退出、
取消分类、清理失败映射及报告快照集中在该操作中；实际业务仍由线性的
``_run_stages`` 完成，最终日志在清理报告发布后记录。既有失败回调继续生效，
作用域入场失败不会启动业务。耗时数据仅保存在每次调用的纯数据记录中，
没有新增运行时管理类或 Runner 实例状态。

``CleanupJournal.persist(mutation, ...)`` 接收具名的可调用操作，内部负责串行化
写入及取消保护，不再用字符串反射选择方法。生命周期模块通过
``remember_created()`` 和 ``has_unresolved_creation`` 使用账本规则，不直接访问
``resources``、``creating`` 等存储字段。

训练及 Ray 接入统一调用 ``run_control`` 的配置和退出接口。环境变量白名单只有
一份，关闭创建准入仍先于任务排空；未配置恢复时在接口内部保持无操作行为。
