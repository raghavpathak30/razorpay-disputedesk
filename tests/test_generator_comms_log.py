"""Defect 1 (session 2): customer_communication_log must not be a low-noise path
into true_fraud. The old design used three fixed templates per branch, so every
string mapped to exactly one true_fraud value. The fix slot-fills text from
shared, overlapping phrase pools; no recurring string should be single-class.
"""

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset

MIN_OCCURRENCES = 20  # below this, single-class groups are expected by chance alone


def test_no_recurring_comms_string_maps_to_a_single_true_fraud_value():
    config = GeneratorConfig()
    features_df, debug_df = generate_dataset(15000, seed=42, config=config)
    merged = features_df.merge(debug_df[["id", "true_fraud"]], on="id")

    counts = merged["customer_communication_log"].value_counts()
    recurring = counts[counts >= MIN_OCCURRENCES].index
    assert len(recurring) > 0, (
        "no string recurred often enough to test - dataset too small or too diverse"
    )

    subset = merged[merged["customer_communication_log"].isin(recurring)]
    distinct_labels = subset.groupby("customer_communication_log")["true_fraud"].nunique()
    single_class = distinct_labels[distinct_labels == 1]
    assert single_class.empty, (
        f"{len(single_class)} recurring comms strings map to exactly one true_fraud "
        f"value, e.g. {single_class.index[0]!r}"
    )


def test_comms_log_has_large_output_space():
    config = GeneratorConfig()
    features_df, _ = generate_dataset(5000, seed=42, config=config)
    assert features_df["customer_communication_log"].nunique() > 500


def test_non_delivery_claim_removed():
    config = GeneratorConfig()
    features_df, _ = generate_dataset(5000, seed=42, config=config)
    logs = features_df["customer_communication_log"].str.lower()
    assert not logs.str.contains("never received", regex=False).any()
