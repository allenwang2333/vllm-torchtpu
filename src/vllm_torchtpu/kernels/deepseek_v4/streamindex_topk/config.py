"""StreamIndex Top-K Configuration."""

import enum


class MlaCase(enum.Enum):
  """Represents the different cases for MLA.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """

  DECODE = 0
  PREFILL = 1
  MIXED = 2

  @property
  def symbol(self):
    return {
        MlaCase.DECODE: "d",
        MlaCase.PREFILL: "p",
        MlaCase.MIXED: "m",
    }[self]


# Default buffer_count for (DECODE, PREFILL, MIXED) cases.
# Determined from microbenchmarks for 8k/1k and 256k/1k.
DEFAULT_BUFFER_COUNT: tuple[int, int, int] = (4, 3, 3)
