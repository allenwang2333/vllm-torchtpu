# How DeepSeek-V4's KV cache gets from the prefill machine to the decode machine

Background for PR #684.

## 1. Some quick definitions

```
  LAYER        a block of the neural network. DSv4 layer 0, layer 1, ...
               one layer can own SEVERAL caches (attention, SWA, compressor state)

  ARRAY        one actual tensor in HBM. Has a shape and a byte size.
               several caches can be overlaid on ONE array.

  CACHE GROUP  vLLM's unit of paging. Layers whose caches page identically get
               bundled into one group, and that group has ONE block table
               shared by every layer in it.

  BLOCK TABLE  where a request's pages physically live. [17, 4, 92] means
               "my first page is in slot 17, my second in slot 4, my third
               in slot 92".

  POOL TAG     Raiden's name for a transferable set of bytes, e.g. "dsv4.swa.g1"
```

The relationship is many-to-many in both directions:

```
   layer 0  ---owns--->  attention cache  ---lives on--->  array A0  ---in--->  group 0
            ---owns--->  swa cache        ---lives on--->  array A0  ---in--->  group 1
            ---owns--->  compressor state ---lives on--->  array A0  ---in--->  group 4
```

## 2. Why DeepSeek-V4 breaks the old assumption

Stage 3 was built for one cache group with one page size, and it refused to
start on anything else, with the error *"requires exactly one full-attention KV
cache group"* (`tpu_connector.py:283-287`, base commit `293a4737`). DSv4 breaks
that twice over.

### 2a. Six groups, six page sizes, six windows

DSv4 mixes several kinds of attention in one model, so vLLM ends up with six
cache groups. Here is the real geometry, taken from block tables captured off a
running DP4/DP4 DeepSeek-V4-Flash pair
(`test_stage3_window_recycled_pages.py:88-102`):

```
  group  pool tag              page size    sliding window
  -----  --------------------  -----------  --------------
    0    dsv4.csa.*, hca, idx  1024 tokens  none (full attention)
    1    dsv4.swa.g1            128 tokens   128 tokens
    2    dsv4.swa.g2            128 tokens   128 tokens
    3    dsv4.state.idx.g3       32 tokens     8 tokens
    4    dsv4.state.csa.g4       16 tokens     8 tokens
    5    dsv4.state.hca.g5       32 tokens   128 tokens
```

Page size in tokens is just `group.kv_cache_spec.block_size`
(`tpu_runner.py:5936-5939`); 1024 comes from `tpu_platform.py:1141-1148`, 128
from `deepseek_v4_attention_op.py:281-289`.

Every group's page is the same size in bytes, and vLLM asserts it
(`kv_cache_utils.py:992-998`). What differs is how many tokens fit in those
bytes, because each group compresses at a different ratio. So a 2000-token
request needs 2 pages in group 0 and 125 in group 4, and a single
`self.block_size` cannot say both.

### 2b. The caches share memory

DSv4 does not give each cache its own memory. It overlays them, in three
separate places in the allocator: sliding-window caches are bound to the same
tensor object as a CSA NoPE array (`tpu_runner.py:5809-5812`), a compressor
state is bound to its own compressed-KV layer's entry (`:5826`), and HCA states
land on CSA NoPE arrays too (`:5828-5829`).

Dumping the arrays of a short DSv4 config makes it concrete:

```
   array A0, 16x256x4x128, 2 MiB
   +--------+--------+--------+--------+--------+
   | page 0 | page 1 | page 2 | page 3 | page 4 |  ...
   +--------+--------+--------+--------+--------+
       ^  all four of these begin at byte 0 of A0:
       |
       +-- layer 0  attention (CSA NoPE)   group 0
       +-- layer 0  swa_cache              group 1   <- overlaid
       +-- layer 0  compressor state       group 4   <- overlaid
       +-- layer 2  compressor state       group 4   <- overlaid, different layer!
```

Now say you hand Raiden "array A0, pages 3 and 7." **That instruction is
ambiguous.** Four caches begin at byte 0 of A0, and nothing in the bytes says
who owns page 3.

What saves it is that groups own disjoint pages: every page comes from one
shared pool (`kv_cache_coordinator.py:91-96`), so if group 1's SWA cache holds
page 3, no other group does. Raiden cannot see that, because all it sees is a
byte range and a list of integers.

## 3. The plumbing that was added

**Each cache group gets its own transfer record.** Stage 3 used to hold a single
page size for the whole model. It now builds one `_Stage3TransferGroup` per
group, carrying that group's index, page size and window
(`tpu_connector.py:125-137`, `:251-296`):

```python
@dataclass(frozen=True)
class _Stage3TransferGroup:
    cache_group_index: int
    page_tokens: int
    window_tokens: Optional[int] = None
```

**Every pool tag names its group,** via a `.g<N>` suffix (`tags.py:30-32`,
`pool_manifest.py:1168`). This is the fix for 2b: `dsv4.swa.g1` reads page
numbers out of group 1's block table and no other, so the ambiguous "A0, page 3"
becomes the unambiguous "group 1's page 3", which disjointness guarantees is
unique.

**A refusal at startup** if two layers of the *same* group ever land on the same
array (`pool_manifest.py:1156-1164`, with the runner keeping its own copy at
`tpu_runner.py:5857-5891`). Same group means one shared block table, and then
the pages really would collide. Better to die at startup than corrupt KV
quietly.

**Per-group block tables on the wire:** `group_block_ids: list[list[int]]`
threaded through the send metadata (`tpu_connector.py:598-613`). One list per
group instead of one flat list for the model:

```
  BEFORE: one group, so one of everything.

    page size 128     block table [17, 4, 92]     pool tag "kv.fa"

  AFTER: one row per cache group.

    group  page  window  block table        pool tag
    -----  ----  ------  -----------------  -----------------
      0    1024  none    [17, 4]            dsv4.csa.nope.g0
      1     128  128     [9, 31, 8]         dsv4.swa.g1
      2     128  128     [22, 6, 44]        dsv4.swa.g2
      3      32  8       [3, 19]            dsv4.state.idx.g3
      4      16  8       [12, 5, 7, 40]     dsv4.state.csa.g4
      5      32  128     [61, 2]            dsv4.state.hca.g5
```

One send now carries all six block tables at once. Raiden reads row N's table
only through row N's tag, so a page number is never interpreted against a group
that does not own it. The tables have different lengths because the groups have
different page sizes, all describing the same request.

That is where most of the PR's 2712 lines go: every place that handled one group
now loops over six, and each of those places exists twice, once where the
producer registers what it is sending and once where the consumer works out
where it lands.

## 4. The bug the real runs shook out

### The setup

Sliding-window attention only ever looks back W tokens. Anything older is
unreachable, so vLLM frees those pages and points their block-table slots at
page 0, the reserved null block meaning "nothing here"
(`single_type_kv_cache_manager.py:507-544`, `block_pool.py:188-192`). The count
freed is

```
  pages_freed = max(0, num_computed_tokens - window + 1) // page_size
```

at `:862`. Read that variable name closely: `num_computed_tokens` is **how far
the producer has actually run**.

The connector computes its own skip from a different number. At
`tpu_connector.py:1371-1379`:

```python
computed_tokens = int(request.num_computed_tokens)
num_tokens = min(computed_tokens, prompt_tokens - 1)
```

`num_tokens` is the **transfer prefix**: the prompt minus its last token. The
last token is deliberately withheld, because the decode replica recomputes it
locally to get its first logits.

So there are two numbers, and both are correct for their own purpose. The
allocator frees against where the producer *is*. The connector plans against
where the *handoff* is. They agree only if the producer stops exactly at the
handoff token.

### Normally they drift by exactly one token

The proxy pins the prefill leg to `max_tokens=1`, so the producer stops one
token past the prefix. Usually harmless:

```
  group 4: page = 16 tokens, window = 8 tokens
  prompt 114 tokens -> num_tokens = 113, producer stops at computed = 114

  allocator frees   (114 - 8 + 1) // 16 = 6 pages
  connector skips   (113 - 8 + 1) // 16 = 6 pages          same. fine.

  block table:  [ 0  0  0  0  0  0 | 286  58 ]
                 \___ 6 freed ____/  \_ 2 live pages, both shipped _/
```

That's `test_handoff_length_transfers_only_live_pages`, with the table captured
live.

But one token is enough to cross a page boundary, and then they differ by one:

```
  prompt 375 -> num_tokens = 374, producer stops at computed = 375

  allocator frees   (375 - 8 + 1) // 16 = 23 pages
  connector skips   (374 - 8 + 1) // 16 = 22 pages          off by one!

  block table:  [ 0 x23 | 13 ]
                 \_____/   ^ live
                    ^ connector skips only the first 22, so page 22 (a null)
                      goes over the wire as a real destination
```

One dead page. The decode replica attends over a hole in an 8-token window,
which rarely changes the answer, so this survives an accuracy sweep and corrupts
state anyway.

### The case that killed the engine

Send a request straight at the producer, bypassing the proxy's `max_tokens=1`
clamp, and it generates the whole answer before `request_finished` runs:

```
  prompt 114 -> num_tokens = 113, but producer ran on to computed = 161

  step 1: at 161 tokens the group has ceil(161/16) = 11 pages,
          and the allocator freed (161 - 8 + 1) // 16 = 9 of them

          [ 0  0  0  0  0  0  0  0  0 | 250  18 ]
            \______ 9 freed __________/  \_live_/

  step 2: the connector normalizes to the 113-token transfer prefix,
          which is ceil(113/16) = 8 pages, discarding the two live ones

          [ 0  0  0  0  0  0  0  0 ]

  step 3: it skips (113 - 8 + 1) // 16 = 6

          [ 0  0  0  0  0  0 | 0  0 ]
            \_ skipped _____/  ^^^^^  two null blocks, shipped
```

That's `test_producer_that_kept_generating_is_caught`, again with the real
table.

**Why two is fatal and one isn't.** It isn't about how much data is lost. Both
nulls are page 0, so Raiden receives the same destination page twice in one
transfer: two sources writing to one slot. It rejects that outright, and the
rejection raises out of `execute_model` on the producer worker, which kills the
engine and every unrelated request riding on it. A single null is merely wrong
data going to a real slot, which nothing detects. That rejection is not visible
in this tree, since `tpu_sync`'s Python layer has no such check and the native
code is not here; it is inferred from the crash.

Notice also what step 2 tells you: by the time the producer reached token 161,
**the entire 113-token prefix had fallen out of an 8-token window and been
legitimately recycled**. There is nothing left to send for that group. The
allocator did nothing wrong; the data is genuinely gone. That's why the fix can
only be a guard and not a repair.

### The guard

It counts null blocks left in the range rather than re-deriving where the
allocator got to, because reading what the table says beats assuming two
formulas agree (`tpu_connector.py:208-235`).

- **More than one null:** drop the send. The decode replica prefills the request
  itself. Slow, but nothing dies.
- **Exactly one null:** warn and ship. A hole in one window is survivable, and
  dropping is worse, because the decode replica is provisioned to decode and
  falls over trying to prefill a long prompt.

The real repair is for the producer to publish the page count it actually
dropped, rather than having the consumer re-derive it. Until then the guard is
what keeps the engine alive.

### Why no other model has ever hit this

The skip is zero unless the group has a sliding window
(`tpu_connector.py:157`), and full attention never reports skipped tokens: the
base manager returns 0 and `FullAttentionManager` doesn't override it
(`single_type_kv_cache_manager.py:546-557`). Qwen3.5 and GLM have a single
full-attention group, so their block tables never contain a null, the two
numbers cannot drift, and the bug is unreachable. The last captured row in the
test makes the point directly: a 1024-token page with no window, at 4371 tokens,
yields zero dead pages.

You need a windowed group to see this, and before this PR no windowed model went
down the Raiden path at all.
