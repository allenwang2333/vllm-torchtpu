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

## 4. Sliding window

Three of DSv4's six groups attend over a sliding window instead of the whole
prompt, and that is the one piece of geometry the transfer cannot simply copy.
A windowed group's block table is mostly holes, and getting the holes wrong is
what killed the engine on the first real runs.

### 4a. vLLM hands the old pages back

A window of W tokens can never reach further back than W tokens, so every page
that falls behind it is unreachable and vLLM returns it to the shared pool. It
does not shorten the block table. It writes page 0, the reserved null block,
over the entry (`single_type_kv_cache_manager.py:507-544`,
`block_pool.py:188-192`). The count it frees is

```
  pages_freed = max(0, num_computed_tokens - W + 1) // page_size
```

at `:862`. Group 4 pages at 16 tokens under an 8-token window, so a 375-token
request ends up looking like this:

```
  [ 0  0  0  ...  0 | 13 ]
    \__ 23 nulls __/   ^ the one page still holding live KV

  24 entries, one page of real data. That is correct, not a bug: an 8-token
  window cannot reach back past a single 16-token page.
```

### 4b. So the transfer has to make the same cut

Sending those nulls is wrong twice over. They hold nothing the decode replica
can attend to, and they are all the same page number, so Raiden gets handed one
destination named 23 times. It rejects a duplicate destination, and the
rejection raises out of `execute_model` on the producer worker, which kills the
engine and every unrelated request riding on it.

The connector therefore drops the dead prefix before it builds the transfer,
using the same formula the allocator used (`_stage3_skipped_window_pages`,
`tpu_connector.py:144-163`, and the trim itself at `:165-205`). Producer and
consumer each run it against the same request length and the same page
geometry, which is what keeps their two page lists paired entry for entry.

For that to work every group has to carry its own window, so `window_tokens`
rides in `_Stage3TransferGroup` next to `page_tokens`. Reading it takes one
extra step, because DSv4 hands its groups over as `UniformTypeKVCacheSpecs`,
which has no `sliding_window` of its own; `_stage3_group_window_tokens`
(`:237-248`) unwraps the wrapper and asks one of the layers underneath.

### 4c. The two numbers have to be one number

Look at what each side counts from. The allocator frees against
`num_computed_tokens`, which is how far the producer has actually run. The
connector skips against `num_tokens`, the transfer prefix, and the transfer
prefix is one token shorter by design, because the last prompt token is
withheld so the decode replica can recompute it and get its first logits
(`tpu_connector.py:1393-1410`).

Two correct numbers, one token apart. One token is enough to cross a page
boundary:

```
  prompt 375, producer runs to computed = 375, num_tokens = 374

    allocator frees   (375 - 8 + 1) // 16 = 23 pages
    connector skips   (374 - 8 + 1) // 16 = 22 pages

  [ 0 x 23 | 13 ]
    \______/
      the connector skips only the first 22, so entry 22 is a null listed
      as a real destination, and it goes over the wire dead
```

The two agree whenever `computed - W + 1` and `num_tokens - W + 1` fall in the
same page, and drift by one whenever they do not, which for group 4 is one
request in sixteen.

### 4d. The fix is to stop the producer where the handoff stops

Rather than reconcile two numbers, make there be one. The producer drops the
final prompt token from its own request before prefill runs, so it computes
N-1 tokens and `num_computed_tokens` really is N-1
(`_maybe_truncate_prompt_for_handoff`, `tpu_connector.py:779-810`). Both
formulas now read the same input and cannot disagree:

```
  prompt 375, producer prefills 374, num_tokens = 374

    allocator frees   (374 - 8 + 1) // 16 = 22 pages
    connector skips   (374 - 8 + 1) // 16 = 22 pages
```

Nothing on the wire changes. `num_tokens` is 374 either way, so the byte spans,
the page-count checks and the whole consumer side are untouched; the only thing
that moves is the allocator's input.

The decode replica lands in the same place for the same reason. It still gets
the full 375-token prompt from the proxy and still recomputes token 374, and
its own allocator runs that same formula on that same 374, so it frees the 22
pages the producer skipped and expects exactly the pages the producer sent.
That is the part worth insisting on: the two sides match by construction, not
because their separate arithmetic happens to land on the same answer.

The machinery for this already existed, for a different model. Mamba state
advances with every token, so a producer that runs to N ships h(N) and the
decode replica then advances the recurrence a second time over token N;
truncating ships h(N-1) instead. The trigger was keyed to Mamba groups, and
`_stage3_truncates_prompt` (`tpu_connector.py:1085-1096`) widens it to any
group that has a window.

### 4e. The guard, for what truncation cannot reach

Truncation fixes the producer that stops at the prompt. It does nothing for a
client that hits the prefill port directly and generates a whole answer, which
is exactly what our accuracy oracle does when it queries prefill alone for a
non-disaggregated baseline. Then the producer runs far past the handoff and the
allocator keeps freeing:

```
  prompt 114, producer runs on to 161

  at 161 tokens the group holds ceil(161/16) = 11 pages, and the allocator
  has freed (161 - 8 + 1) // 16 = 9 of them:

    [ 0 x 9 | 250  18 ]

  the connector normalizes to the 113-token prefix, ceil(113/16) = 8 pages,
  which throws away both live ones:

    [ 0 x 8 ]

  and then skips 6, leaving two nulls. Two nulls is the duplicate destination.
```

By that point the entire 113-token prefix has genuinely fallen out of an
8-token window. The allocator did nothing wrong and the data is gone, so there
is nothing to repair and the only question is how to fail.

The guard counts the nulls left in the range rather than re-deriving where the
allocator got to, because reading what the table says beats assuming two
formulas agree (`_stage3_recycled_transfer_pages`, `:208-235`). More than one
null and it drops the send, so the decode replica prefills the request itself:
slow, but nothing dies. Exactly one and it warns and ships, because a hole in
one window is survivable and dropping is worse, given that the decode replica
is provisioned to decode and falls over trying to prefill a long prompt.

Raiden's rejection of a duplicate destination is not visible in this tree.
`tpu_sync`'s Python layer has no such check and the native code is not here, so
it is inferred from the crash.

### 4f. Why no other model has ever hit this

The skip is zero unless the group has a window (`tpu_connector.py:157`), and
full attention never reports skipped tokens: the base manager returns 0 and
`FullAttentionManager` does not override it
(`single_type_kv_cache_manager.py:546-557`). Qwen3.5 and GLM have a single
full-attention group, so their block tables never contain a null, the two
numbers cannot drift, and none of section 4 is reachable. The last captured row
in `test_stage3_window_recycled_pages.py` makes the point directly: a 1024-token
page with no window, at 4371 tokens, yields zero dead pages.

You need a windowed group to see any of this, and before this PR no windowed
model went down the Raiden path at all.
