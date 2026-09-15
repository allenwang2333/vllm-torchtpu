"""Materialize validated experimental tiles and trace serving lookup decisions."""
import json
import sys
from pathlib import Path


def main():
    winners = json.loads(Path(sys.argv[1]).read_text())
    assert len(winners) == 18
    path = Path(__file__).resolve(
    ).parents[1] / "src/vllm_torchtpu/kernels/mla/v2/tuned_params.py"
    source = path.read_text()
    marker = "# PR898_C256_EXPERIMENTAL_TILES"
    assert marker not in source
    additions = ["", marker]
    for row in winners:
        key, (bq, pages, split) = row["key"], row["tile"]
        assert key["max_num_seqs"] == 256 and key["pages_per_seq"] == 256
        assert key["q_dtype"] == "bfloat16" and key[
            "kv_dtype"] == "float8_e4m3fn"
        assert key["actual_num_q_heads"] == 3
        params = dict(
            num_queries_per_block=bq,
            num_kv_pages_per_block=pages,
            vmem_limit_bytes=62914560,
            decode_batch_size=split if key["case"] == "batched_decode" else 1,
            q_split=split if key["case"] == "mixed" else 1)
        additions.append(
            f"tuned_params_mapping[TuningKey(**{key!r})] = TunableParams(**{params!r})"
        )
    old = "    if tuning_key in tuned_params_mapping:\n        return tuned_params_mapping[tuning_key]"
    new = (
        '    if tuning_key in tuned_params_mapping:\n'
        '        logger.info("MLA_SERVING_TUNING_HIT key=%s params=%s", tuning_key, tuned_params_mapping[tuning_key])\n'
        '        return tuned_params_mapping[tuning_key]')
    assert source.count(old) == 1
    # Generated parameter-table extension, kept in the experiment checkout only.
    path.write_text(source.replace(old, new) + "\n".join(additions) + "\n")
    print("EXPERIMENTAL_TUNING_INSTALLED", len(winners), flush=True)


if __name__ == "__main__":
    main()
