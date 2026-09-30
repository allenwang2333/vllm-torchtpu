"""GLM-5.2 decode megakernel geometry. `FULL` is the served model; `TINY` keeps every
structural constraint (lane multiples, cache byte layout, head/rank split, the
dense/MoE and full/shared-indexer layer patterns) at a size interpret mode can run."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class MKConfig:
    R: int                 # ranks (= decode rows = DP = EP = TP for the sharded projections)
    L: int                 # decoder layers
    first_dense: int       # layers [0, first_dense) use the dense MLP
    H: int                 # hidden
    NH: int                # attention heads (NH % R == 0)
    QL: int                # q_lora
    KL: int = 512          # kv_lora (fixed by the 512 B nope cache token)
    DN: int = 192          # qk_nope
    DR: int = 64           # rope (fixed by the rope cache token)
    DV: int = 256          # v head
    E: int = 256           # routed experts (E % R == 0)
    TOPK: int = 8
    I: int = 2048          # moe intermediate
    IS: int = 2048         # shared expert intermediate (1 shared expert)
    ID: int = 12288        # dense intermediate
    IH: int = 32           # indexer heads
    IDIM: int = 128        # indexer head dim
    KT: int = 2048         # indexer top-k
    idx_freq: int = 4
    idx_skip: int = 3
    page: int = 1024
    rope_theta: float = 8e6
    eps: float = 1e-5
    routed_scale: float = 2.5
    KREP: int = 0          # decode replica slots per rank per MoE layer (0: off, program unchanged)
    SPEC: int = 0          # speculative first expert: at most SPEC tiles issued early (0: off)

    @property
    def HR(self):          # heads per rank
        return self.NH // self.R

    @property
    def ER(self):          # experts per rank
        return self.E // self.R

    @property
    def DQ(self):          # q head width
        return self.DN + self.DR

    @property
    def QKVA(self):
        return self.QL + self.KL + self.DR

    def full_indexer(self, l):
        """Layers that compute top-k; the others reuse the last computed set."""
        return l < self.idx_skip or (l - self.idx_skip) % self.idx_freq == self.idx_freq - 1

    def is_moe(self, l):
        return l >= self.first_dense


FULL = MKConfig(R=16, L=78, first_dense=3, H=6144, NH=64, QL=2048)

TINY = MKConfig(R=4, L=5, first_dense=1, H=512, NH=8, QL=256, DN=64, DV=128,
                E=8, TOPK=2, I=256, IS=256, ID=512, IH=4, KT=256, idx_freq=2,
                idx_skip=1, page=256)
