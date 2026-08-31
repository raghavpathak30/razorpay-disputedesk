"""The policy engine's tunable constants. One place, so a decision can always be
traced to the parameters that produced it, per CLAUDE.md's config-module
convention. Nothing here is derived from data - these are policy choices, not
measurements.
"""

from pydantic import BaseModel, ConfigDict

# INR, per contested dispute. This is the cost SPEC.md §4 subtracts in
# `expected_value` and the cost SPEC.md §6 asks to decompose for the
# false-positive metric (checklist item 6).
#
# ASSUMPTION, not a citation: no published Razorpay or card-network fee
# schedule for representment cost was found for this project. Modeled as
# three named components, each a stated guess:
#   representment / network resubmission fee:            INR 200
#   analyst time to assemble and submit the packet:       INR 150
#   excessive-representment exposure (marginal risk of    INR  50
#     tripping a card network's dispute-ratio program on
#     a representment that is later lost)
#   total:                                                 INR 400
# Replace with a real figure before this system is used with real money.
REPRESENTMENT_COST_INR: float = 400.0


class PolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    representment_cost_inr: float = REPRESENTMENT_COST_INR

    # The "I don't know" band (SPEC.md §4): P(win) this close to a coin flip
    # is treated as too uncertain to act on automatically, regardless of the
    # sign of expected_value, and is escalated to a human instead. Inclusive
    # at both ends. ASSUMPTION: centered on 0.5 (maximum model uncertainty)
    # rather than on the dispute's own breakeven point, so the band has a
    # fixed, auditable meaning independent of `amount`.
    low_confidence_band: tuple[float, float] = (0.45, 0.55)
