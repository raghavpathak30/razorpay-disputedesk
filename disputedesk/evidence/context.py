"""The order-context fields the evidence assembler reads from a dispute row.
A plain dataclass, not an LLM schema (`schemas.py` is for LLM output only) -
this is deterministic input, built straight from `DisputeRecord`-shaped data
by `assembler.py`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DisputeContext:
    reason_code: str
    amount: float
    avs_match: bool
    cvv_match: bool
    device_fingerprint_known: bool
    delivery_confirmed: bool
    prior_order_count: int
