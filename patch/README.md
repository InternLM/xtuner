# SGLang patches

## Routed experts via Ray

Patch: `patch/sglang-0.5.15.post1-routed-experts-ray.patch`

Target version: `sglang==0.5.15.post1`

### Principle

SGLang normally serializes the complete `routed_experts` tensor as base64 in
the HTTP response. This patch stores the tensor in a Ray named actor/object
store and returns only its key through HTTP. XTuner uses that key to fetch the
ndarray from Ray, avoiding the base64 expansion and large HTTP payload.

The patch covers both the normal detokenizer path and the
`skip_tokenizer_init` path. If the Ray store is unavailable, it falls back to
the original base64 encoding. `indexer_topk` is unchanged.

```text
routed_experts tensor -> Ray shared_store -> key in HTTP response
                      -> XTuner fetches ndarray from Ray
```

XTuner requires the matching decode logic in
`xtuner/v1/rl/rollout/sglang.py` and `ENABLE_RETURN_ROUTED_EXPERTS=1`.

### Apply

Run from the XTuner repository root:

```bash
python -c "import importlib.metadata; print(importlib.metadata.version('sglang'))"

XTUNER_REPO=$(pwd)
PATCH_FILE="$XTUNER_REPO/patch/sglang-0.5.15.post1-routed-experts-ray.patch"
SITE_PACKAGES=$(python -c "import importlib.util, pathlib; spec = importlib.util.find_spec('sglang'); print(pathlib.Path(spec.origin).resolve().parents[1])")

cd "$SITE_PACKAGES"
git apply --check -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
git apply -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
```
