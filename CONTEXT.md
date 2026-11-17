# Sandbox cleanup domain

- **Isolated trial** keeps an entire infer/verify pipeline in one disposable
  CPU process. Ray owns scheduling, model services and training trace objects.
  The public `run_trial_process` interface takes declarative configuration,
  confirms process exit and recovers only pre-allocation registered journals.
  A private manifest atomically publishes one entry per journal; a partially
  published entry cannot have allocated resources. Run shutdown still owns the
  complete shared ledger scope if the parent disappears.
- **Creation permit channel** retains the actor's existing shared token bucket
  across isolated processes. Request identities separate concurrent/cancelled
  waiters; child disconnect settles actor-side permit requests. Model traffic
  continues through the HTTP proxy and never crosses this channel.
- **Kernel ownership** uses protocol-marked Linux OFD record locks. Fresh
  descriptors, fork closure and explicit guardian transfer preserve ownership.
  Unknown historical lock namespaces fail closed. Local capability observation
  and deployment-level cross-node exclusion are separate validation obligations.

- **SandboxPool** plans a trial's named dependency groups, checks health and runs
  provisioners. Public phase release keeps the management transport available
  for a later verifier; final close ends the trial's ownership.
- **Trial preparation** belongs to Runner: input validation and isolated Pool
  overrides precede execution. Recipes may observe the prepared Pool, then use
  Runner's standard execution without changing the task's overrides or copying
  its merge and cancellation rules. Setup scopes close a prepared Pool when
  execution has not started.
- **Trial execution** keeps stage orchestration in Runner and cleanup result
  policy in its module's `_run_sandbox_trial` operation. The latter owns scope
  entry/exit, cancellation, cleanup failures and the final receipt, preserving
  the Runner's existing failure callback. Runner calls it once and logs only
  after the receipt is published; timings are a per-call data record.
- **SandboxLifecycle** owns one Pool's resource-release obligations. It tracks
  allocated IDs and clients, confirms deletion, seals the journal and hands off
  unfinished cleanup. It also owns acquisition attempts, partial rollback and
  the decision to retry. Pool supplies dependency/readiness work through a
  per-attempt create capability. Release stops admission and drains acquisition
  completion before sealing ownership or closing transport; a resistant create
  retains the journal lease and transport for a later close attempt. Persistent
  and in-process-only operation share this module.
  A phase shares fixed deletion deadlines across its groups and drain-triggered
  rollback; the phase deadline expires with that release operation. Independent
  acquisition rollback uses its own existing budget, never an earlier phase's.
- **CleanupJournal** persists exact resource ownership and unresolved creation
  intents under a kernel lease. A DELETE response alone is not deletion evidence.
  Each concurrent request has an internal intent token; allocation/end resolve
  only that request. Their aggregate retains the existing persistent `creating`
  boolean, so older recovery readers still see an unresolved request.
- **Run cleanup** stops admission before task drain, then performs bounded exit
  recovery. `run_control` owns configuration propagation and exit integration;
  `recycler` freezes requests and ancestor budget ownership after an explicit
  handoff. An exit batch seals its resource scope once, after stopping admission;
  retries then use that immutable scope without renewing ancestor work.
