# How DeepSeek-V4's KV cache gets from the prefill machine to the decode machine

Background for PR #684. Written for someone who has never seen this codebase.

## 1. Why split the work across two machines

When you send a prompt to a language model, the server does two jobs that look
nothing alike.

**Prefill** reads your whole prompt at once. Thousands of tokens in parallel,
enormous matrix multiplies, chips pinned at 100%.

**Decode** writes the answer one token at a time. Each step touches one token,
so the chips are mostly idle, waiting on memory.

Running both on the same chips is wasteful, because a long prefill stalls
everyone else's decode steps. So you put them on separate machines:

```
   "What is the capital of France?"   (131 tokens, say)
                 |
                 v
   +----------------------+                        +----------------------+
   |       PREFILL        |                        |        DECODE        |
   |   8 Ironwood chips   |  ---- KV cache ---->   |   8 Ironwood chips   |
   |  reads all 131 at    |                        |  emits one token at  |
   |  once, then stops    |                        |  a time              |
   +----------------------+                        +----------------------+
                                                              |
                                                              v
                                                           "Paris"
```

That's P/D disaggregation. Everything in this PR is about the arrow.

## 2. What actually has to move, and why it isn't a copy

While prefill reads your prompt it builds a **KV cache**: for each token, in
each layer, a pair of vectors the model must keep in order to attend back to
that token later. Decode cannot produce a single word without it.

The arrow is not a memcpy, and that's the first thing to understand. The two
machines lay the same data out differently.

```
  PREFILL splits ONE request across chips by token position (PCP):

     chip0: tokens 0-15    chip1: tokens 16-31   ...   chip7: tokens 112-127

  DECODE splits DIFFERENT requests across chips (DP):

     chip0: request A (all of it)   chip1: request B (all of it)  ...
```

Prefill wants every chip working on your one prompt, because that's how you
finish a big matrix multiply fast. Decode wants every chip on a different user,
because one user's decode step is too small to fill a chip. So the KV for your
request starts scattered across eight chips and must arrive gathered on one.

Rearranging it in flight is called **resharding**, and it's why this path uses
Raiden (`tpu_sync`) instead of the default ZMQ transport. The env var that picks
it is documented as "Controller-driven PCP->DP pool resharding"
(`envs.py:248-254`). You can see the assumption baked into the connector: it
converts between scheduler blocks and physical pages with
`scheduler_block_tokens = page_tokens * pcp_size` (`tpu_connector.py:1336`),
because one scheduler-level block is spread across `pcp_size` chips.

## 3. Pages and block tables

You can't hand each request one contiguous slab of memory, because you don't
know how long the answer will be and you'd shred memory into unusable gaps. So
vLLM does what an operating system does: fixed-size **pages**.

```
  physical KV memory, numbered slots:
  +--------+--------+--------+--------+--------+--------+
  | page 0 | page 1 | page 2 | page 3 | page 4 | page 5 |  ...
  +--------+--------+--------+--------+--------+--------+

  your request's BLOCK TABLE:  [17, 4, 92]
      "my first page lives in slot 17, my second in slot 4, my third in slot 92"
```

Three facts here carry the whole rest of the document.

Pages come from **one shared pool** for the entire engine
(`kv_cache_coordinator.py:91-96`), and a page is only handed out when nobody
holds it (`block_pool.py:542-573`). So a page number belongs to exactly one
owner at a time. That turns out to be the only thing keeping DSv4 from
corrupting itself.

Page 0 is reserved as the **null block**, meaning "nothing here"
(`block_pool.py:188-192`). Freeing a page means writing 0 into that slot of the
block table.

And Raiden's interface is exactly `(which memory region, which page numbers,
how many tokens)` (`byte_spans.py:391`). It has no idea what a model is. That
narrowness is what DSv4 breaks.

## 4. What the code assumed before this PR

One group of layers, one page size, one block table:

```
BEFORE  (base commit 293a4737)

  self._stage3_fa_group_index = 0        # which group? there's only one. an int.
  self.block_size = 128                  # one page size for the entire model
  request_finished(req, [17, 4, 92])     # one flat list of page numbers
```

`tpu_connector.py:276`, `:513`, `:386-389`. It didn't just assume this, it
refused to start otherwise, with the error *"requires exactly one full-attention
KV cache group"* (`:283-287`).

For Qwen3.5 and GLM that's simply true, and the code is correct as written.

## 5. Layer, array, cache group, pool tag

These four words sound like they nest, and they don't. Getting them straight is
most of the battle, which is why the PR puts their definitions in
`pool_manifest.py`'s docstrings.

```
  LAYER        a block of the neural network. DSv4 layer 0, layer 1, ...
               one layer can own SEVERAL caches (attention, SWA, compressor state)

  ARRAY        one actual tensor in HBM. Has a shape and a byte size.
               several caches can be overlaid on ONE array.

  CACHE GROUP  vLLM's unit of paging. Layers whose caches page identically get
               bundled into one group, and that group has ONE block table
               shared by every layer in it.

  POOL TAG     Raiden's name for a transferable set of bytes, e.g. "dsv4.swa.g1"
```

The relationship is many-to-many in both directions, which is the uncomfortable
part:

```
   layer 0  ---owns--->  attention cache  ---lives on--->  array A0  ---in--->  group 0
            ---owns--->  swa cache        ---lives on--->  array A0  ---in--->  group 1
            ---owns--->  compressor state ---lives on--->  array A0  ---in--->  group 4
```

One layer, three caches, one array, three different groups.

## 6. Why DeepSeek-V4 breaks the old assumption

### 6a. Six groups, six page sizes, six windows

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

There's a nice wrinkle. Every group uses the **same number of bytes** per page,
and vLLM asserts it (`kv_cache_utils.py:992-998`). What differs is how many
tokens fit in those bytes, because the groups compress at different ratios. So
"how many pages does this 2000-token request need?" has six simultaneously
correct answers, from 2 to 125. A single `self.block_size` cannot express that.

### 6b. The caches share memory

This is the harder half, and it's the one that makes the change genuinely
necessary rather than merely tedious.

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
ambiguous.** Four caches begin at byte 0 of A0. Nothing in the bytes says who
owns page 3.

So what prevents them from trampling each other during normal inference?
Exactly one thing, and it's the invariant from section 3: the allocator never
gives one page number to two groups at the same time. If group 1's SWA cache
holds page 3, nobody else holds page 3, so "A0, page 3" happens to be unique in
practice. That's real, and it's load-bearing, but Raiden cannot see it, because
Raiden sees a byte range and a list of integers.

## 7. The plumbing that was added

Four things, each answering one problem above.

**A transfer plan with one record per group,** replacing the single integer
(`tpu_connector.py:125-137`, `:251-296`):

```python
@dataclass(frozen=True)
class _Stage3TransferGroup:
    cache_group_index: int
    page_tokens: int
    window_tokens: Optional[int] = None
```

**Every pool tag names its group,** via a `.g<N>` suffix (`tags.py:30-32`,
`pool_manifest.py:1168`). This is the fix for 6b: `dsv4.swa.g1` reads page
numbers out of group 1's block table and no other, so the ambiguous "A0, page 3"
becomes the unambiguous "group 1's page 3", which the disjointness invariant
guarantees is unique.

**A refusal at startup** if two layers of the *same* group ever land on the same
array (`pool_manifest.py:1156-1164`, with the runner keeping its own copy at
`tpu_runner.py:5857-5891`). Same group means one shared block table, and then
the pages really would collide. Better to die at startup than corrupt KV
quietly.

**Per-group block tables on the wire:** `group_block_ids: list[list[int]]`
threaded through the send metadata (`tpu_connector.py:598-613`).

```
  BEFORE                             AFTER
  group index    : 0                 [(0, 1024, None), (1, 128, 128),
  page size      : 128                (2, 128, 128),   (3, 32, 8),
  block table    : [17, 4, 92]        (4, 16, 8),      (5, 32, 128)]
  tag            : "kv.fa"
                                     [[17,4], [9,31,8], [12,5,7], ...]
                                     "dsv4.csa.nope.g0", "dsv4.swa.g1", ...
```

Most of the PR's 2712 added lines are that six-way split repeated. It appears
twice over, once on the producer side where spans are registered and once on the
consumer side where destination blocks are built, because those are separate
code paths in `tpu_connector.py`. Then again in the byte-span lowering, the
manifest builder, and the tests.

## 8. The bug the real runs shook out

### The setup

Sliding-window attention only ever looks back W tokens. Anything older is
unreachable, so vLLM frees those pages and writes the null block over their
slots (`single_type_kv_cache_manager.py:507-544`). The count is

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
data going to a real slot, which nothing detects.

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

## 9. Vocabulary

**Token** a word-piece. **KV cache** the per-token vectors the model keeps in
order to attend back. **Page (block)** a fixed-size slice of KV memory. **Block
table** the list mapping a request's logical pages to physical slots. **Cache
group** layers that page identically and therefore share one block table. **Null
block** page 0, meaning freed. **Array** one physical tensor in HBM. **Pool tag**
Raiden's name for a transferable byte set. **Resharding** rearranging KV in
flight because the two machines shard differently. **PCP** prefill context
parallel, splitting one request's tokens across chips. **DP** data parallel,
splitting different requests across chips.

## 10. What is not verified from code in this tree

**Raiden's duplicate-destination rejection is not visible here.** The `tpu_sync`
Python layer has no such check, so it lives in the native code. The claim rests
on the observed crash, not a line anyone can cite.

**The nine-tags-over-six-groups mapping is observed, not derived.** The `.gN`
index is purely positional (`tags.py:30-32`), and which role lands where depends
on layer counts living in the checkpoint config rather than either repo. The
synthetic fixture in `test_raiden_pool_manifest.py:762-768` deliberately uses a
different mapping to make that point.

**The single-null case is knowingly left imperfect.** The real repair is for the
producer to publish the page count it actually dropped instead of making the
consumer re-derive it. The guard is how you stop dying while that's outstanding.

**"Stage 3" is project jargon** with no definition anywhere in the repo.
Functionally it's the Raiden-backed resharding path selected by
`TPU_KV_RESHARD_TRANSPORT=raiden`.
