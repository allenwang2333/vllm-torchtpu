#!/bin/bash
# The four shards of the composed-candidate job, one per TPU chip.
#
#   bash shards.sh <0|1|2|3>
#
# The spec lists live here rather than in `RUN_CMD` because `RUN_CMD` is a
# single-quoted YAML scalar carrying 115 specs; a file is easier to read, has
# no quoting hazard, and is committed alongside the results so the allocation
# is recoverable later.
#
# ## Allocation rule
#
# **Every arm of a configuration runs on the same shard.**  The arms of one
# config are compared to each other, and per-chip thermal or clock differences
# would land directly in that comparison if the arms were split across chips.
# Configs are what get spread; arms never are.
#
# Within a config the order is `inc ctl c dl c_dl`, and for the replicated
# headline the order is rep-major (all five arms at rep 1, then rep 2, then
# rep 3) so that any drift over the shard's lifetime moves all five arms
# together and shows up in the replicate spread rather than in one arm.
#
# ## Balance
#
#   shard 0 : 35 specs -- the headline (21 layers) x 3 replicates, + 4 x ctx4096
#   shard 1 : 30 specs -- the prefill chain and both fast-path chains, + ctx16384
#   shard 2 : 30 specs -- the six largest single-call decode configs
#   shard 3 : correctness (10 configs x 5 arms) + the 4 prefill single-calls
#
# Fan-out buys `sum - max`, so the wall clock is shard 0's or shard 3's; the
# 21-layer chains are the expensive specs and shard 0 has 15 of them.
set -u
CX_DIR=${CX_VARIANT:-/root/kw/variants/composed}
SHARD=${1:-${TPU_FANOUT_SHARD:-0}}

# The five arms, in report order, for one config.
arms() {
  local cfg=$1 rep=${2:-}
  local a
  for a in inc ctl c dl c_dl; do
    if [ -n "$rep" ]; then echo -n "$cfg:$a:$rep "; else echo -n "$cfg:$a "; fi
  done
}

SPECS=""
case "$SHARD" in
  0)
    # Item 1 (headline) and item 4 (within-job replicates, n=3) are the same
    # traces: the production decode point at 21 layers, measured three times.
    for rep in 1 2 3; do
      SPECS="$SPECS $(arms L21_dec_b64_ctx8704 $rep)"
    done
    # Item 3, part: the ctx-4096 column of the single-call grid, plus the
    # single-call twin of the headline shape.
    for cfg in L1_dec_b8_ctx4096 L1_dec_b32_ctx4096 L1_dec_b64_ctx4096 \
               L1_dec_b64_ctx8704_mml9216; do
      SPECS="$SPECS $(arms $cfg)"
    done
    ;;
  1)
    # Item 2: the 21-layer prefill chain.
    SPECS="$SPECS $(arms L21_pre_q1024_ctx16384)"
    # Item 5: the fast-path regime at 21 layers.
    SPECS="$SPECS $(arms L21_dec_b64_ctx1024)"
    SPECS="$SPECS $(arms L21_dec_b64_ctx2048)"
    # Item 3, part: the ctx-16384 decode column.
    for cfg in L1_dec_b8_ctx16384 L1_dec_b32_ctx16384 L1_dec_b64_ctx16384; do
      SPECS="$SPECS $(arms $cfg)"
    done
    ;;
  2)
    # Item 3, part: the two longest-context decode columns.  These allocate the
    # biggest KV caches (b64/ctx65536 is ~1 GiB), so they are kept together on
    # one shard rather than spread into shards that also hold 21-layer chains.
    for cfg in L1_dec_b8_ctx32768 L1_dec_b32_ctx32768 L1_dec_b64_ctx32768 \
               L1_dec_b8_ctx65536 L1_dec_b32_ctx65536 L1_dec_b64_ctx65536; do
      SPECS="$SPECS $(arms $cfg)"
    done
    ;;
  3)
    # Item 6 (correctness) plus item 3's four prefill single-calls.  The
    # prefill configs are the ones Option C actually runs on, so they share a
    # shard with the correctness check that exercises the same launch.
    for cfg in L1_pre_q1024_ctx4096 L1_pre_q1024_ctx16384 \
               L1_pre_q4096_ctx4096 L1_pre_q4096_ctx16384; do
      SPECS="$SPECS $(arms $cfg)"
    done
    ;;
  *)
    echo "shards.sh: unknown shard $SHARD" >&2
    exit 2
    ;;
esac

# shellcheck disable=SC2086
set -- $SPECS
echo "===== CX SHARD $SHARD : $# specs ====="
echo "$@" | tr " " "\n" | sed "s/^/  spec /"

if [ "$SHARD" = "3" ]; then
  exec bash "$CX_DIR/correctness_shard.sh" "$@"
fi
exec bash "$CX_DIR/perf_shard.sh" "$@"
