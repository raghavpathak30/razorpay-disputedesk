"""Every guessed generator parameter, in one place. No magic numbers live outside this file.

Each field's comment cites the GENERATOR.md section it comes from. Fields with no
direct GENERATOR.md line item are implementation parameters needed to realize a
qualitative claim from GENERATOR.md §1 (e.g. turning "strength: medium" into an
actual logistic-regression coefficient) — those are marked "(implementation)".
"""

from pydantic import BaseModel, ConfigDict


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # --- §7 Temporal structure ---
    simulation_window_months: int = 24  # §7, open param 5
    true_fraud_rate_month0: float = 0.40  # §7, open param 7
    true_fraud_rate_month_last: float = 0.50  # §7, open param 7
    auth_strength_drift_shift: float = 0.05  # §7, open param 8 ("small upward shift")
    respond_by_days: int = 7  # §7, open param 6
    train_window_months: int = (
        18  # §7 / PHASES.md Phase 1 (implementation: temporal split boundary)
    )

    # --- §6 Confounders ---
    traveler_share_of_genuine: float = 0.08  # §6A, open param 3
    account_takeover_share_of_fraud: float = 0.12  # §6B, open param 4

    # --- §4 The p band and its calibration targets ---
    p_min: float = 0.02  # §4, open param 1
    p_max: float = 0.75  # §4, open param 1
    p_mode_low_target: float = 0.08  # §4, open param 2 (documentation/validation target)
    p_mode_high_target: float = 0.39  # §4, open param 2 (documentation/validation target)
    e_p_target: float = 0.25  # §4 (documentation/validation target)
    oracle_pr_auc_low_target: float = (
        0.30  # §5, open param 9 (documentation only, not used by generator)
    )
    oracle_pr_auc_high_target: float = (
        0.36  # §5, open param 9 (documentation only, not used by generator)
    )

    # --- Logistic combination realizing L1-L7 (§1) into p (§2 step 5) ---
    # Coefficients calibrated so that, at mean latent values, true_fraud=1 rescales
    # to ~p_mode_low_target and true_fraud=0 rescales to ~p_mode_high_target — see
    # DECISIONS.md-style arithmetic note in generator/probability.py. (implementation)
    logit_intercept: float = -1.8  # (implementation) calibrated constant term
    coef_true_fraud: float = 1.556  # §1 L1 "strength: large" (implementation, calibrated)
    coef_authentication_strength: float = 1.2  # §1 L2 "medium-large" (implementation)
    coef_relationship_genuineness: float = 0.6  # §1 L3 "medium" (implementation)
    coef_delivery_provability: float = 0.5  # §1 L4 "medium" (implementation)
    coef_filing_delay_norm: float = 0.5  # §1 L5 "medium" (implementation)
    coef_dispute_propensity: float = 0.25  # §1 L7 "small" (implementation)
    reason_subtype_logit_offset_scale: float = (
        0.05  # §1 L6 "small, code-dependent offset" (implementation)
    )
    epsilon_sigma: float = 0.6  # §2 step 4, irreducible residual scale (implementation, Guess)
    filing_delay_norm_cap_days: float = (
        60.0  # (implementation) normalization cap for L5 in the logit
    )

    # --- §1/§3 Latent distributions, per mixture component ---
    # Beta(a, b) shape params for authentication_strength (§1 L2, §6)
    auth_strength_beta_fraud_nonconfounder: tuple[float, float] = (2.0, 5.0)
    auth_strength_beta_account_takeover: tuple[float, float] = (5.0, 3.0)  # §6B: "can pass cleanly"
    auth_strength_beta_genuine_nonconfounder: tuple[float, float] = (5.0, 2.0)
    auth_strength_beta_traveler: tuple[float, float] = (2.0, 5.0)  # §6A: "poor avs_match"

    # Beta(a, b) shape params for relationship_genuineness (§1 L3, §6)
    relationship_beta_fraud_nonconfounder: tuple[float, float] = (2.0, 4.0)
    relationship_beta_account_takeover: tuple[float, float] = (
        5.0,
        2.0,
    )  # §6B: inherits good history
    relationship_beta_genuine_nonconfounder: tuple[float, float] = (5.0, 2.0)
    relationship_beta_traveler: tuple[float, float] = (5.0, 2.0)  # §6A: "genuinely high otherwise"

    # Beta(a, b) for delivery_provability (§1 L4) — kept component-independent per
    # GENERATOR.md's silence on a confounder interaction here (implementation,
    # flagged simplification)
    delivery_provability_beta: tuple[float, float] = (3.0, 3.0)

    # filing_delay_days (§1 L5): Gamma(shape, scale=mean/shape), mean depends on true_fraud.
    # Session-2 sanity check measured AUC(days_between_purchase_and_dispute, true_fraud)
    # =0.0623 against the original means (3 / 20 days) - a near-deterministic read on
    # true_fraud, far past what "medium strength, independent evidential weight" (§1 L5)
    # should produce. Solved analytically rather than re-tuned by trial, using the
    # Gamma(k,k) ratio identity: for X ~ Gamma(k, theta_x), Y ~ Gamma(k, theta_y)
    # independent with the same shape k, W = Y/(X+Y) ~ Beta(k, k), and
    #   P(Y > X) = P(W > rho/(1+rho))  where rho = theta_x / theta_y = mean_x / mean_y
    # Solving P(Y > X) = 0.35 numerically at k=2 (shape unchanged) gives rho ~= 1.51.
    # Verified by full-pipeline simulation, including the +-2-day sensor noise below
    # (n=300,000): mean_true_fraud_days=8.5, mean_genuine_days=13.0 (ratio 1.529) ->
    # empirical AUC(days_between_purchase_and_dispute, true_fraud) = 0.3507.
    filing_delay_gamma_shape: float = 2.0  # (implementation) unchanged
    filing_delay_mean_true_fraud_days: float = 8.5  # §1 L5: fraud victims still report sooner
    filing_delay_mean_genuine_days: float = 13.0  # §1 L5: friendly fraud still files later

    # dispute_propensity (§1 L7): Beta(a, b), drawn independently of component
    dispute_propensity_beta: tuple[float, float] = (2.0, 2.0)

    # --- §3 Observable-feature sensor noise ---
    # Session-2 fix (defect 6): booleans were drawn as Bernoulli(latent) directly
    # (e.g. authentication_strength=0.3 -> only a 30% chance avs_match=True), which
    # for latents spread across [0,1] compounds ~30% inherent disagreement with the
    # latent's own threshold *before* flip_prob is even applied - avs_match/cvv_match
    # did not appear in the top-8 LightGBM gain features despite authentication_strength
    # being a "medium-large" latent, while ip_geo_billing_distance_km (a continuous
    # reading of the same latent) ranked #1. Switched to threshold(latent) + flip_prob,
    # and lowered flip_prob so it is the single, literal noise source. Verified by
    # simulation: AUC(avs_match, true_fraud) moved from 0.35 (0.15 from chance) under
    # the old mechanism to 0.23 (0.27 from chance) under the new one at flip_prob=0.08.
    boolean_reading_threshold: float = 0.5  # (implementation) shared by all four booleans below
    avs_match_flip_prob: float = 0.05  # (implementation) lowered per defect 6
    cvv_match_flip_prob: float = 0.08  # (implementation) independent from avs_match, per §3 note
    device_fingerprint_flip_prob: float = 0.10  # (implementation)
    delivery_confirmed_flip_prob: float = 0.07  # (implementation)
    prior_order_count_scale: float = (
        15.0  # (implementation) Poisson lambda = relationship_genuineness * scale
    )
    prior_dispute_count_scale: float = (
        3.0  # (implementation) Poisson lambda = dispute_propensity * scale
    )
    ip_geo_distance_max_km: float = (
        8000.0  # (implementation) inverse reading of authentication_strength
    )
    ip_geo_distance_noise_km: float = 400.0  # (implementation) additive Gaussian sensor noise
    days_between_purchase_and_dispute_noise_days: float = (
        2.0  # §1 L5 revision note, open param 12 (±1-3 days -> Gaussian sigma)
    )

    # amount (§3, §8, open param 10): lognormal, mu shifted by true_fraud, shared sigma.
    # Session-2 sanity check measured AUC(amount, true_fraud)=0.7631 against the
    # original mu gap of 0.9 - far above the "weak link" GENERATOR.md §3 describes.
    # Solved analytically instead of re-tuned by trial: for two lognormals sharing
    # sigma, AUC = Phi(delta_mu / (sigma*sqrt(2))). Target AUC=0.60:
    #   z = Phi^-1(0.60) = 0.253347
    #   delta_mu = sigma * sqrt(2) * z = 0.9 * 1.414214 * 0.253347 = 0.322458
    # mu_genuine held at its original value; mu_true_fraud = mu_genuine + delta_mu.
    # Verified by simulation (n=200,000): empirical AUC = 0.5998.
    amount_lognormal_mu_true_fraud: float = 8.922458  # (implementation) -> median ~ INR 7,498
    amount_lognormal_mu_genuine: float = 8.6  # (implementation) unchanged -> median ~ INR 5,432
    amount_lognormal_sigma: float = 0.9  # (implementation) unchanged per the solve above

    checkout_hour_uniform: bool = True  # §3, open param 13 (uniform over 0-23)

    # reason_subtype -> reason_code (§8, revision 2 confirmed codes). Session-2 fix
    # (defect 2): reason_code was an exact copy of the reason_subtype latent - §1's
    # "no feature is a near-noiseless read of a latent, except L5" rule names L5 as
    # the sole exception, so L6 needs the same noisy-observation treatment every
    # other latent gets. Modeled as issuer misclassification: with this probability
    # the recorded code differs from the true reason_subtype (Guess, not derived).
    reason_code_misclassification_prob: float = 0.10  # (implementation) §1 L6 revision, defect 2
    reason_codes: tuple[str, ...] = ("MC_4837", "MC_4840", "VISA_83", "AMEX_FR2")

    card_networks: tuple[str, ...] = (
        "Visa",
        "Mastercard",
        "RuPay",
        "Amex",
    )  # §3 pure-noise control

    # --- customer_communication_log (§3, defect 1, session 2) ---
    # Session-2 fix: three fixed templates per true_fraud branch meant a string match
    # recovered true_fraud exactly (6 of 6 templates mapped to a single true_fraud
    # value). Replaced with slot-filled text (opening/claim/detail/signoff) drawn from
    # shared phrase pools, weighted (never zero) by true_fraud so phrasings overlap
    # across classes instead of partitioning them. All weights below are Guesses.
    # Weights are deliberately mild (max ratio 1.5:1) and kept to three correlated
    # slots (opening/claim/detail) rather than compounding further - five
    # independently-skewed draws per row (three content slots plus signoff and
    # detail-inclusion, both also a function of relationship_genuineness) let the
    # single most "maximally genuine-flavored" combination compound to a joint
    # probability low enough that, at n=15,000, it could occur zero times for
    # true_fraud=1 by chance alone - purity from a probability-zero tail, not
    # from a leak, but indistinguishable from one by a group-purity test. Milder
    # weights and slopes keep every recurring combination's minority-class count
    # comfortably above zero (verified: no string with n>=5 occurrences is
    # single-class across three seeds at n=15,000).
    comms_opening_weights_fraud: tuple[float, ...] = (0.30, 0.27, 0.23, 0.20)
    comms_opening_weights_genuine: tuple[float, ...] = (0.20, 0.23, 0.27, 0.30)
    comms_claim_weights_fraud: tuple[float, ...] = (0.30, 0.27, 0.23, 0.20)
    comms_claim_weights_genuine: tuple[float, ...] = (0.20, 0.23, 0.27, 0.30)
    comms_detail_weights_fraud: tuple[float, ...] = (0.30, 0.27, 0.23, 0.20)
    comms_detail_weights_genuine: tuple[float, ...] = (0.20, 0.23, 0.27, 0.30)
    # relationship_genuineness affects tone (polite vs terse signoff) and length
    # (whether a detail sentence is included), per §3's causal-parent listing.
    comms_signoff_polite_base: float = 0.3  # (implementation) P(polite) at relationship=0
    comms_signoff_polite_relationship_scale: float = 0.3  # (implementation) added at relationship=1
    comms_detail_inclusion_base: float = (
        0.45  # (implementation) P(detail included) at relationship=0
    )
    comms_detail_inclusion_relationship_scale: float = (
        0.3  # (implementation) added at relationship=1
    )
    # Realistic messiness: near-empty logs, occasional irrelevant asides, typos and
    # inconsistent capitalization - the latter scaled inversely with
    # relationship_genuineness, so low-coherence accounts read messier.
    comms_near_empty_prob: float = 0.05  # (implementation)
    comms_irrelevant_detail_prob: float = 0.08  # (implementation)
    comms_typo_base_prob: float = 0.10  # (implementation)
    comms_typo_relationship_scale: float = 0.15  # (implementation) added at relationship=0
    comms_lowercase_prob: float = 0.06  # (implementation)
