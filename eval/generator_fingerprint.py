"""A single hash over the generator's numeric output, so any change to it -
intentional or not - fails a fast, loud, always-run test instead of surviving
silently until someone happens to notice a downstream number drifted.

Why this module exists (2026-09-03, stale-number audit item A). The audit
found `DECISIONS.md`'s recorded seed-42 realized average precision, 0.4335,
does not reproduce from *any* commit in this repository's history - not
because the generator changed under it (it never did; every commit from the
first one produces 0.4304927827841146 at that seed), but because the number
was never computed from committed code at all. That is a worse failure mode
than drift: nothing would have caught it, because nothing was watching for
it. The mechanism here is what the audit's own report asked for - something
that makes the *next* case of this fail the suite rather than survive until
someone happens to split a test.

The fingerprint is a hash of `generate_dataset`'s full output - every value
of every column of both the feature frame and the debug/latent frame, at one
fixed, small, CI-cheap `(n_rows, seed)` - not a summary statistic. A summary
statistic can leave two different generators indistinguishable (two RNG call
orders that happen to average out); a full-value hash cannot.

This catches every generator change, including ones with no numeric effect at
all (e.g. a comment, or a rename that only relabels a value already covered
by `eval.leakage`'s provenance guard). That is intentional, not a false
positive to tune away: the point is not "did the *numbers* change", it is
"did the code path that produces every recorded number change" - the second
question is answerable cheaply and reliably; the first is not, without
re-running every downstream measurement, which is the whole problem.
"""

import hashlib

import pandas as pd

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset

FINGERPRINT_N_ROWS = 2000
FINGERPRINT_SEED = 0


def _frame_bytes(df: pd.DataFrame) -> bytes:
    """A canonical byte serialisation of a frame's values: columns in a fixed
    (sorted) order, full float precision, no index. Column *order* in the
    source frame is deliberately not part of the hash - `eval.leakage`'s
    provenance guard already owns "did the column set change"; this hash is
    about values.
    """
    ordered = df[sorted(df.columns)]
    return ordered.to_csv(index=False, float_format="%.17g").encode("utf-8")


def compute_generator_fingerprint(
    n_rows: int = FINGERPRINT_N_ROWS, seed: int = FINGERPRINT_SEED
) -> str:
    """SHA-256 over the full value content of both frames `generate_dataset`
    returns, at one fixed `(n_rows, seed, GeneratorConfig())`.
    """
    features_df, debug_df = generate_dataset(n_rows, seed, GeneratorConfig())
    digest = hashlib.sha256()
    digest.update(_frame_bytes(features_df))
    digest.update(_frame_bytes(debug_df))
    return digest.hexdigest()


# The committed fingerprint. Any change to the generator's numeric output -
# including one with no visible effect on any headline metric - changes this
# hash. `tests/test_generator_fingerprint.py` asserts equality against it on
# every run, CI included, with no seed count or scale flag to skip it.
#
# On a legitimate generator change: regenerate this constant with
#     python -c "from eval.generator_fingerprint import \
#         compute_generator_fingerprint as f; print(f())"
# then re-run and re-commit every golden fixture this repository pins
# against generator output before updating it - grep for `GOLDEN FIXTURE`
# across `tests/` for the current list. Updating this constant without doing
# that is the exact failure this mechanism exists to prevent.
COMMITTED_GENERATOR_FINGERPRINT = "337f7194132849520a32c04147c51c3fde65c4850434e12b7b3175f7976a7869"
