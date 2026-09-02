"""The generator fingerprint gate (stale-number audit item A, 2026-09-03):
any change to the generator's numeric output fails this test, unconditionally
and on every CI run, rather than surviving until a downstream number happens
to be checked.

This exists because the audit found a number - the seed-42 realized average
precision, recorded as 0.4335 - that never matched any commit in this
repository's history, and nothing had ever noticed. The mechanism here does
not depend on anyone remembering to re-check a specific figure: it hashes the
generator's entire output at one fixed seed and compares against a committed
constant, so a change to `disputedesk/generator/` of any kind - not just ones
a human predicts will matter - is caught here, immediately, by name.
"""

from eval.generator_fingerprint import (
    COMMITTED_GENERATOR_FINGERPRINT,
    compute_generator_fingerprint,
)


def test_the_generator_fingerprint_matches_the_committed_value():
    """If this fails, the generator's numeric output changed. That is not
    necessarily wrong - but every golden fixture pinned against generator
    output (grep `tests/` for `GOLDEN FIXTURE`) must be re-run and
    re-committed, and `COMMITTED_GENERATOR_FINGERPRINT`
    (`eval/generator_fingerprint.py`) updated to match, in the same change
    that touched the generator. Do not update the constant to make this pass
    without doing that - that is the exact failure this test exists to catch.
    """
    assert compute_generator_fingerprint() == COMMITTED_GENERATOR_FINGERPRINT


def test_the_fingerprint_is_deterministic_across_repeated_calls():
    """A prerequisite for the gate meaning anything: if the fingerprint itself
    were not reproducible, a failure here could not be trusted to mean the
    generator changed.
    """
    assert compute_generator_fingerprint() == compute_generator_fingerprint()


def test_the_fingerprint_is_sensitive_to_a_single_changed_value():
    """Proves the gate can actually fire, the same way the leakage guard's
    red-team fixtures do - a guard that only ever passes is worthless. Patches
    one latent value in an otherwise-identical run and confirms the hash
    moves.
    """
    import hashlib

    from disputedesk.generator.config import GeneratorConfig
    from disputedesk.generator.pipeline import generate_dataset
    from eval.generator_fingerprint import FINGERPRINT_N_ROWS, FINGERPRINT_SEED, _frame_bytes

    features_df, debug_df = generate_dataset(
        FINGERPRINT_N_ROWS, FINGERPRINT_SEED, GeneratorConfig()
    )
    genuine = hashlib.sha256(_frame_bytes(features_df) + _frame_bytes(debug_df)).hexdigest()
    assert genuine == COMMITTED_GENERATOR_FINGERPRINT

    tampered_debug = debug_df.copy()
    tampered_debug.loc[tampered_debug.index[0], "p"] += 1e-9
    tampered = hashlib.sha256(_frame_bytes(features_df) + _frame_bytes(tampered_debug)).hexdigest()

    assert tampered != genuine


def test_the_fingerprint_is_sensitive_to_a_relabelled_value_with_no_numeric_effect():
    """The case that actually happened: `b5770a1` renamed `VISA_83` to
    `VISA_10_4` in `config.reason_codes`. That rename turned out to have no
    numeric effect (`_reason_subtype_offsets` maps by position, not by the
    string's meaning) - but the fingerprint changes anyway, because the value
    in the `reason_code` column changed, even though no measured quantity did.
    That is intentional: this gate answers "did the code path change", not
    "did anything I currently measure change" - the second question is what
    let 0.4335 go unnoticed.
    """
    import pandas as pd

    from eval.generator_fingerprint import _frame_bytes

    original = pd.DataFrame({"reason_code": ["VISA_83", "MC_4837"], "amount": [100.0, 200.0]})
    renamed = pd.DataFrame({"reason_code": ["VISA_10_4", "MC_4837"], "amount": [100.0, 200.0]})

    assert _frame_bytes(original) != _frame_bytes(renamed)
