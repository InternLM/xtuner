# SGLang routed_experts patches

These patches make SGLang return `routed_experts` through Ray instead of
embedding the full tensor payload in the HTTP response.

After the patch is applied, `routed_experts` is stored in a Ray named
actor/object store and the HTTP response only returns the shared-store key.
XTuner can then fetch the real ndarray through Ray.

The expected transfer path is:

```text
SGLang routed_experts tensor
  -> SGLang _RoutedExpertsSharedStore.put()
  -> ray.put(data) stores ndarray in Ray object store
  -> HTTP meta_info["routed_experts"] returns shared-store key
  -> XTuner SGLang rollout worker gets shared_store actor in namespace "sglang"
  -> rollout worker actor.get(key) fetches real ndarray
  -> rollout worker ray.put(np.asarray(...))
  -> trainer worker ray.get(...) consumes rollout_routed_expert
```

This only changes the SGLang side. XTuner also needs matching logic in
`xtuner/v1/rl/rollout/sglang.py` to resolve the returned Ray key.

## sglang-0.5.13-routed-experts-ray.patch

Patch file:

```text
patch/sglang-0.5.13-routed-experts-ray.patch
```

This patch is generated against a clean `sglang==0.5.13` wheel.

It covers both SGLang routed experts encoding paths:

- `sglang/srt/managers/detokenizer_manager.py`: normal `BatchStrOutput` path,
  where routed experts are pre-encoded before reaching the tokenizer manager.
- `sglang/srt/managers/tokenizer_manager.py`: `BatchTokenIDOutput` path, used
  when detokenization is bypassed.
- `sglang/srt/managers/routed_experts_shared_store.py`: shared helper that owns
  the Ray named actor/object-store logic.

`indexer_topk` keeps the upstream base64 behavior. Only `routed_experts` is
moved to Ray key transport.

Check the installed SGLang version before applying:

```bash
python -c "import importlib.metadata; print(importlib.metadata.version('sglang'))"
```

Expected output:

```text
0.5.13
```

Apply from the XTuner repository checkout:

```bash
XTUNER_REPO=$(pwd)
PATCH_FILE="$XTUNER_REPO/patch/sglang-0.5.13-routed-experts-ray.patch"
SITE_PACKAGES=$(python -c "import importlib.util, pathlib; spec = importlib.util.find_spec('sglang'); print(pathlib.Path(spec.origin).resolve().parents[1])")

cd "$SITE_PACKAGES"
git apply --check -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
git apply -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
```

Validate the applied patch:

```bash
git apply --reverse --check -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
python -m py_compile \
  sglang/srt/managers/detokenizer_manager.py \
  sglang/srt/managers/tokenizer_manager.py \
  sglang/srt/managers/routed_experts_shared_store.py
grep -R "class _RoutedExpertsSharedStore" -n sglang/srt/managers
```

The final `grep` should show a single definition in
`sglang/srt/managers/routed_experts_shared_store.py`.

To restore SGLang to the pre-patch state:

```bash
cd "$SITE_PACKAGES"
git apply --reverse -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
```

## Runtime usage

Use a MoE model that can return routed experts, for example Qwen3-A3B MoE.
Use the SGLang rollout backend and enable routed experts return:

```bash
export ENABLE_RETURN_ROUTED_EXPERTS=1
```

Then run XTuner normally, for example:

```bash
bash examples/v1/scripts/run_rl.sh \
  examples/v1/config/rl_grpo_gsm8k_async.py \
  sglang \
  "$MODEL_PATH" "$DATA_PATH" "$EVAL_DATA_PATH"
```

Validation signals in logs:

- SGLang rollout config contains `"enable_return_routed_experts": true`.
- Ray actors contain `shared_store(_RoutedExpertsSharedStore)`.
- Training logs contain `rollout_routed_expert = torch.as_tensor(...)`.
- Mismatch metrics such as `mismatch/mismatch_kl` are printed during training.

To disable this path, unset the flag or set:

```bash
export ENABLE_RETURN_ROUTED_EXPERTS=0
```

## sglang-0.5.10-routed-experts-ray.patch

Patch file:

```text
patch/sglang-0.5.10-routed-experts-ray.patch
```

This patch is generated against a clean `sglang==0.5.10` wheel.

SGLang `0.5.10` only needs to patch
`sglang/srt/managers/tokenizer_manager.py`. The patch stores
`routed_experts` in the Ray named actor/object store and returns only the
shared-store key in the HTTP response.

Check the installed SGLang version before applying:

```bash
python -c "import importlib.metadata; print(importlib.metadata.version('sglang'))"
```

Expected output:

```text
0.5.10
```

Apply from the XTuner repository checkout:

```bash
XTUNER_REPO=$(pwd)
PATCH_FILE="$XTUNER_REPO/patch/sglang-0.5.10-routed-experts-ray.patch"
SITE_PACKAGES=$(python -c "import importlib.util, pathlib; spec = importlib.util.find_spec('sglang'); print(pathlib.Path(spec.origin).resolve().parents[1])")

cd "$SITE_PACKAGES"
git apply --check -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
git apply -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
```

Validate the applied patch:

```bash
git apply --reverse --check -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
python -m py_compile sglang/srt/managers/tokenizer_manager.py
```

To restore SGLang to the pre-patch state:

```bash
cd "$SITE_PACKAGES"
git apply --reverse -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
```
