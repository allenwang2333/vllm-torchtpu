# Probe 02 — `decomp-128k`: does the stored `n=133120` number reproduce, in my job?

**Question.** Two numbers were carried in from earlier work at `n=133120`: **2138 GB/s** SparseCore
HBM read bandwidth, and **"5.78% of stage 1 is DMA"**. `PROBING.md` forbids inheriting either — a
baseline measured in a different job is not my baseline. Do they reproduce here, and is the shape
they were taken at one anybody runs?

**Why it needed hardware.** It is a re-measurement by definition. It rode along in probe 01's job at
no extra job cost.

## Arms

Identical to probe 01 (`--decomp`, same arm set, 9 interleaved rounds, `us_median`), at
`--ctx 132096` → `n=133120`, `b ∈ {16, 1024}`, `k=2048`. Both shapes take `P=4`.

## Config

Same job, same image `fa63d6df` + mapped `b11b6eb6` `src/`, same in-job md5
`44353b347c96e0eedbd742c710d4540a`.

## Job

`j-fa60ff2a-c793-4f32-9576` → `j-fa60ff2a-c793-4f32-9576.txt`, `decomp_128k.json`.

## Answer

**Both reproduce.** Stored 1.9924 µs/wave at slice 33280 came back at **2.001 µs/wave** — 0.4% out,
inside the 1.86% null spread. The 5.78% DMA fraction came back as 5.3% hideable.

| `b` | waves | `stage1` | DMA slope | hideable | ceiling `(w−1)/w` |
| --- | --- | --- | --- | --- | --- |
| 16 | 2 | 94.4 µs | 4.0 µs | 4.2% | −2.12% |
| 1024 | 128 | 4801.6 µs | 256.1 µs | 5.3% | −5.29% |

**So the stored numbers were never wrong — they were taken at a shape no checked-in config sets.**
`--ctx 132096` is this harness's *default*, and reaching `n=133120` needs `--max-model-len 132096`,
which neither `glm-5.2-fp8-dp16-ep.sh` nor `deepseek-v4-flash-dp8-ep.sh` sets (report §4). That is
the difference between a stale number and a stale *shape*, and only the second one was true here.

A second, subtler consequence: at `n=133120` every bucket takes `P=4`, so all the two-stage
machinery — `gatherA`, `stage2`, `gatherB` — executes. At the shipping `n=10240` it does not, for
any bucket except `b=16`. **Reasoning carried over from this shape describes a code path production
mostly does not run.**

## Decided

Every number in `report.md` is quoted at `--ctx 9216`. The `n=133120` rows are kept in §6's ceiling
table as the long-context case, labelled as such, and excluded from the ranking that drives step 2.
