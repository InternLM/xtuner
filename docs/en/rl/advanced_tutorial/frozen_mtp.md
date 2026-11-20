# External frozen MTP with an actor that omits MTP

This opt-in route preserves dedicated Qwen3.5 MoE MTP weights while training an
actor whose text model has `mtp_config=None`. The actor's shared embedding and
output head still update normally. Dedicated `mtp.*` tensors come from an
explicit, content-verified CPU snapshot outside the actor and its optimizer.

The supported rollout is LMDeploy's PyTorch backend, TP1/EP1, using IPC or NCCL
weight transport and `qwen3_5_mtp` speculation. This route does not support MTP
training, other model families, Turbomind, vLLM, SGLang, or checkpoint-engine
transport. It does not change ordinary configurations with `frozen_mtp=None`.

## Configuration and source identity

Create a source manifest once, using a read-only Hugging Face model directory:

```bash
python -m xtuner.tools.fingerprint_frozen_mtp \
  --source /models/qwen35-original \
  --output ./frozen-mtp-manifest.json
```

Record the printed MTP content identity in the configuration. It is computed
from the sorted tensor names, shapes, dtypes, sizes and individual SHA256 values;
it is independent of mount paths and shard filenames. The manifest also binds
the source model config. Preparing a manifest reads every dedicated MTP tensor;
each actor verifies and caches its own CPU snapshot before building its engine.

Configure the existing worker and rollout fields:

```python
from xtuner.v1.rl.weight_update.frozen_mtp import FrozenMTPConfig

model_cfg.text_config.mtp_config = None
frozen_mtp = FrozenMTPConfig(
    source_dir="/models/qwen35-original",
    manifest_path="./frozen-mtp-manifest.json",
    content_sha256="<64-character MTP content identity from the manifest>",
)
# Pass frozen_mtp=frozen_mtp when constructing WorkerConfig.

rollout_config.extra_rollout_config.update(
    lmdeploy_speculative_algorithm="qwen3_5_mtp",
    lmdeploy_speculative_num_draft_tokens=3,
)
```

The `lmdeploy_` prefix is required by the native server-config transformation.
The draft length is a rollout setting and does not select an MTP training depth.
The obsolete experimental `XT_FROZEN_MTP_SOURCE` environment variable is not
consumed: this explicit configuration is the sole source for frozen weights.
Backend-selection environment variables still have their native precedence;
selecting a different backend makes this adapter reject the configuration.

To disable inference speculation, omit both speculative fields and leave
`WorkerConfig.frozen_mtp=None`. This follows the native non-speculative route;
it does not inject an external MTP shard into ordinary actor HF exports. Native
MTP training configurations without this adapter retain their existing behavior.

## Update, HF export and DCP restore

The actor passes its verified source into the existing weight updater. The native
iterator exports target weights, sends the dedicated MTP weights in dtype-homogeneous
batches, and only then emits its final marker. Every full update repeats this
sequence, including after level-2 sleep discarded the receiver's weights. A tensor
larger than the bucket limit is sent whole. The default MTP bucket limit is 256 MiB;
the actor CPU snapshot and a device copy of the current bucket consume additional
memory. Shared embedding/head values always come from the current target.

`TrainingWorker.save_hf` exports each role into its own directory. For this adapter,
`ActorWorker.save_hf` requires an empty actor directory, lets the native exporter
finish, appends `frozen-mtp.safetensors`, updates `model.safetensors.index.json`,
writes `frozen_mtp_source.json`, and finally publishes an actor-directory `READY`
record. It never publishes a root-directory READY record. Shard, index or binding
failure prevents this marker; rank-zero filesystem failure is reported to all
participating training ranks. Native export failures retain their native semantics.

Repeated export is deliberately rejected before overwriting a populated actor
directory. A partial export is not resumable or implicitly repaired: retain it for
diagnosis, isolate that directory, and export again to a fresh destination. This is
not a filesystem durability guarantee across host or storage failure.

DCP checkpoints save the actor and optimizer through the native role layout and
record the external source identity in `<checkpoint>/actor/frozen_mtp_source.json`.
They do not contain the MTP tensors themselves. Restore requires the source and
manifest again, checks the recorded identity before loading the actor role, and
rejects a missing/changed source or disabling the adapter for a bound checkpoint.
An old checkpoint without this record cannot silently acquire a new frozen source.
HF warm-start is separate from full DCP/optimizer restore.

## Validation scope

Focused tests exercise source identity and architecture checks, invalid startup
combinations, the real native iterator's final-marker order, actual LMDeploy
speculation config construction, actor-role HF placement, failed/repeated exports,
and the DCP source gate. CPU fixtures do not prove device transfer, distributed
collective behavior, a usable exported model, optimizer correctness or online Agent
continuation. Those require integration validation on the intended model and
backend. The adapter does not change reward, advantage, masks, logprob or R3 semantics.

## Alternative native route

[XTuner #2292](https://github.com/InternLM/xtuner/pull/2292) keeps dedicated frozen MTP weights in the actor as an alternative. Do not enable both routes. `FrozenMTPConfig.validate_model` requires `actor.mtp_config=None` and rejects a native `MTPConfig`, including `frozen=True`; use exactly one parameter owner. This external adapter still requires its own receiver/GPU validation; the native route's GPU results do not substitute for that evidence.
