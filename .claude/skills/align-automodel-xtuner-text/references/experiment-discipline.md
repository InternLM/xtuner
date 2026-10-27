# Comparison experiment discipline

Check every item before designing or executing a comparison experiment. Focus here is on experiment
design; execution-side traps overlap with pitfalls P5.

## Alignment and configuration

1. Align parameters as far as possible; strategies alignable → align; otherwise list item by item
   (✅/❌+reason). Never leave a difference silent.
2. Compute bit-width must match: never one side bf16 and the other fp32; gradient-reduce dtype too.
3. Fixed constant lr + identical warmup (P1.2/P2.2).
4. fp32 master is mandatory-checked (P1.1).
5. Align backend stacks before memory/throughput comparisons (P2.3).
6. When reproducing someone else's issue, stay as close to their original config as possible. A layered
   design (minimal pair for localization, faithful pair for reproduction) is fine, but state the tradeoff
   explicitly — never silently downgrade.
7. Short runs: both sides in normal config; never compensate one side (P2.8). The principle is **because
   aligned, therefore identical results** — not "align until the results match".
8. A/B arms share one source directory (cold Dynamo cache pollutes first steps); switch only env toggles
   or config.

## Verdict levels

9. Trend comparison is the floor; a "numerical paths are identical" verdict requires the same pack stream
   pointwise (Stage 4), never trend endpoints alone.
10. Across different total-step counts, only s1 is comparable (before any update, lr-independent);
    measure a switch's "direct single-step effect" as the s1 on/off grad-norm relative difference.
11. Long-horizon comparisons use statistical shape (spike counts, envelopes, transition windows), not
    pointwise pairing — implementation-level microdifferences amplify chaotically; short-horizon (~20
    steps) pointwise alignment remains valid and necessary.
12. When suspecting non-determinism, first diff the per-step lr curves of both arms: same-lr arms stay
    within a small bound over long runs — separation comes from hyperparameters, not noise.
13. To prove the data stream identical, compare pure data statistics (e.g. effective-attention ratios)
    per step — all-equal rules out the data side.

## Experiment management

14. Multi-arm queues: the criterion arm first (clean → done), auxiliary arms later; stop an arm once past
    its criterion step count.
15. When a paired arm is requested, run both arms; the criterion step count is a stopping floor, not a cap
    promise.
16. When an intermittent issue fails to reproduce in A/B: repeat rounds (quantifying the noise floor =
    detection limit) + sweep configuration dimensions (e.g. pack length ladders) with both arms always on
    the same stream and config.
17. Clean criterion arm → stop per criterion; only with an anomaly, bisect along the change ladder.
18. Long experiments: short diagnostic run first, full run after; give a time estimate (startup +
    s/step × steps) before launching.
19. Tabulate failure sequences per attempt (change vs last attempt → death point → progress depth →
    attribution), with a reference-baseline table.
20. Rerunning an old baseline needs justification (first run / recalibration / new-bottleneck isolation).
21. When direction changes, tear down running chains cleanly: kill the relay watcher first (prevent
    auto-respawn), then the driver, then training; archive partial logs; verify GPUs idle before starting
    a new chain.
22. Never block on waiting: one short check or a background task; do other work while waiting.
23. Present results as line plots (archived PNG + terminal ASCII when useful); trends are judged from
    curves, not tables of endpoints.
