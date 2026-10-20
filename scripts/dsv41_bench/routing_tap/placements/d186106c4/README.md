# Static expert placements for DeepSeek-V4.1-Flash, solved on `epic/dsv41` @ d186106c4

`placement_round1.json` is the per-layer expert permutation `TPU_MOE_EXPERT_PERM_PATH` consumes
(`src/vllm_torchtpu/expert_permutation.py`). It was solved by `solve_placement.py` from the routing counts of
40 steady C384 decode steps (384 x 98,304-token LongBench-v2 prompts, DSpark K5, TP1/DP8/EP8) tapped on
2026-10-18 with `routing_tap/` (forge run 0mpapc69). It balances the per-device expert-tile cost of the fused-EP
MoE kernel; `placement_identity.json` is the identity permutation in the same format, the control arm.

Measured with it (same host, CI image, huge-page Engram staging on in both arms): fused-EP kernel -56 us/layer
mean (encoder layers -46, decoder layers -66), MoE 19.66 -> 17.41 ms/step, device self time 60.06 -> 57.79 ms/step;
median host-observed step 65.90 / 65.99 ms (placement) vs 67.30 / 70.18 ms (identity), which does not pass the
four-start gate because of start-level variance. Layer output is byte-identical to the unpermuted model
(12/12 isolated checks, loader tests). The table is specific to this checkpoint and was solved on the CI prompts;
other traffic may route differently. Solve a new one with `solve_placement.py` from a fresh tap.
