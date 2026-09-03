# NUMBERS.md

Every numeric claim in `README.md`, `ARCHITECTURE.md`, and `DECISIONS.md`,
mapped to the exact command that reproduces it. Built 2026-09-03 (Phase 3)
in direct response to the stale-number audit: 0.4335 was quoted for a full
day of DECISIONS.md entries with no command attached, and nothing caught
that it had never been computed from committed code. A number with no
command in this table is a number with no provenance — that is the failure
this file exists to make impossible to repeat.

Commands that need `LLM_API_KEY` are marked **[KEY]**. Everything else runs
from a clean clone with no secrets. `make verify` (see the end of this file)
regenerates every non-**[KEY]** row in one command.

Withdrawn numbers are listed at the end, not omitted — a number that used to
be here and no longer reproduces is exactly the class of fact this file
exists to keep visible.

---

## Calibration provenance

| Claim | Value | Command |
|---|---|---|
| Generator parameters sourced against published statistics (chargeback rate, friendly-fraud share, win rate, contest cost, analyst cost, VAMP threshold, reason-code mix) | See `CALIBRATION.md` — an external-citation index, not a number this repo computes, so it is not reproduced by a command here | `CALIBRATION.md` |

## Generator and model quality

| Claim | Value | Command |
|---|---|---|
| Model PR-AUC (average precision) | 0.3522 (IQR 0.3448–0.3696) | `python -m eval.run_harness --n-seeds 20 --n-rows 15000` |
| Prevalence baseline | 0.2377 (IQR 0.2331–0.2422) | same |
| Oracle Bayes ceiling | 0.4572 (IQR 0.4527–0.4585) | same |
| Calibration error (ECE) | 0.0270 (IQR 0.0208–0.0314) | same |
| Placeholder threshold precision/recall (superseded, kept for history) | 0.3537 / 0.6954 | same |
| Oracle closed-form vs. replicate mean, seed 42 | 0.4556 vs. 0.4569 (std 0.0173, SE 0.0008) | `pytest tests/test_eval_oracle_replicate_check.py -v` |
| Seed-42 realized AP (golden fixture) | 0.4304927827841146 | `pytest tests/test_eval_oracle_replicate_check.py::test_the_historical_seed_42_draw_reproduces` |
| `AUC(days_between_purchase_and_dispute, true_fraud)`, n=15,000, seed=42 | 0.3504 | `python -c "from sklearn.metrics import roc_auc_score; from disputedesk.generator.config import GeneratorConfig; from disputedesk.generator.pipeline import generate_dataset; f,d=generate_dataset(15000,42,GeneratorConfig()); print(roc_auc_score(d['true_fraud'],f['days_between_purchase_and_dispute']))"` |
| `AUC(amount, true_fraud)`, n=15,000, seed=42 | 0.6082 | same pattern, substitute `f['amount']` |
| `AUC(days_between_purchase_and_dispute, true_fraud)`, n=300,000, seed=42 | 0.3496 | same pattern, `n_rows=300000` |
| `AUC(amount, true_fraud)`, n=200,000, seed=42 | 0.5978 | same pattern, `n_rows=200000`, substitute `f['amount']` |
| `reason_code` misclassification rate, seed=42 | 0.0971 | `python -c "from disputedesk.generator.config import GeneratorConfig; from disputedesk.generator.pipeline import generate_dataset; f,d=generate_dataset(15000,42,GeneratorConfig()); print((f['reason_code'].to_numpy()!=d['reason_subtype'].to_numpy()).mean())"` |
| Generator fingerprint (n=2000, seed=0) | `337f7194...` | `pytest tests/test_generator_fingerprint.py` |

## Leakage guard margins

| Claim | Value | Command |
|---|---|---|
| Bayes AUC (n=5,000, seed=11) | 0.7397 (lift 0.2397) | `pytest tests/test_generator_leakage_guard.py -v` |
| Strongest legitimate feature's ceiling fraction (`ip_geo_billing_distance_km`) | 69.9% | same |
| Legitimate categoricals' NMI ceiling | ≤0.000463 vs. 0.30 threshold | same |

## Business metrics (configured cost, ₹400)

| Claim | Value | Command |
|---|---|---|
| Policy recovered (naive_contest), median | 1,714,015 (IQR 1,632,554–1,746,598) | `python -m eval.run_business_harness --n-seeds 20 --n-rows 15000` |
| Baseline A (contest everything), median | 1,701,092 (IQR 1,620,248–1,737,029) | same |
| Baseline B (accept everything) | 0 | same |
| Policy recovered (zero mode) | 1,559,504 | same |
| Policy recovered (oracle mode) | 1,727,152 | same |
| Escalated share of holdout rupees | 0.0433 | same |
| False positive count / cost per 1,000 | 2,204.5 / 244,384 (IQR 241,137–247,783) | same |
| False negative count / cost per 1,000 | 71.5 / 42,357 (IQR 38,127–51,175) | same |
| Paired advantage, configured cost | +11,210 (95% CI +8,508 to +13,633, 19/20 seeds) | `python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000` |
| Advantage as % of baseline A | ≈0.66% | derived: 11,210.3 / 1,701,092.5 |

## Phase 2 — EV threshold verification (no number moved)

Phase 2 found the contest/accept decision already derived from the cost
model (`decide()`'s `expected_value = p_win * amount - representment_cost`
has been SPEC.md §4's rule since Phase 3) — algebraically the Elkan (2001)
`p > cost/amount` threshold. Nothing was refactored; these rows are the
verification, not a change. Full detail: `DECISIONS.md` 2026-09-03 "Phase 2
— EV threshold verified, calibration and escalate band checked".

| Claim | Value | Command |
|---|---|---|
| Threshold equivalence pinned on real holdout predictions | `decide()`'s CONTEST decision == `p_win > cost/amount` on every non-escalated holdout row | `pytest tests/test_policy_ev_threshold_is_derived.py -v` |
| Brier score, model vs. always-predict-prevalence baseline | 0.1697 (IQR 0.1670–0.1715) vs. 0.1812 baseline | `python -m eval.run_calibration_report` |
| Reliability table (10 bins, pooled across 20 seeds, n=72,130) | see DECISIONS.md entry for the full table | same |
| Near-threshold reliability (±0.05 of each row's own `cost/amount`) | mean predicted 0.1061 vs. observed 0.1223 (gap −0.0162), n=15,028/72,130; overall median derived threshold 0.0628 | same |
| Escalate band fraction of holdout | 0.0562 (IQR 0.0485–0.0624) | `python -m eval.run_escalate_band_counterfactual` |
| Band-free EV-rule counterfactual advantage vs. baseline A, ₹400 | +11,478.0 (95% CI +8,746.4 to +13,936.5, 19/20 seeds) | same |
| Band cost, directly paired per seed (not a difference of two marginal estimates) | +267.7 INR/1,000 (95% CI +146.4 to +397.3, 16/20 seeds positive — excludes zero) | same |

## Cost sweep (paired estimator)

| Claim | Value | Command |
|---|---|---|
| Full 0→10,000 paired advantage table | see README "Policy precision/recall" table | `python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000` |
| No measurable advantage at/below ₹150; appears at ₹200 | CI excludes 0 from ₹200 up (except ₹50, which is negative) | same |
| ESCALATE rate (invariant to cost) | 0.0562 (IQR 0.0485–0.0624) | same |
| CONTEST rate | 0.8052 (IQR 0.7985–0.8186) | same |
| ACCEPT rate | 0.1363 (IQR 0.1316–0.1407) | same |
| Loss-tail table (₹50/₹100/₹400) | 7.65× / 2.52× / 0.45× loss÷gain | `python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000` (prints the loss-tail block automatically wherever mean/majority disagree) |
| Break-even human-review cost, configured point | ≈₹200 | same (`break_even_human_review_cost_inr` column) |
| Human-cost overstatement at configured point | 200.5% (22,475 vs. 11,210) | `pytest tests/test_eval_sweep_assumptions.py -v` plus the DECISIONS.md 2026-09-03 recomputation |

## Ablation

| Claim | Value | Command |
|---|---|---|
| Top-1 feature advantage at ₹400, % of full | 65.8% (+7,377 of +11,210) | `python -m eval.run_ablation --n-seeds 20 --n-rows 15000` |
| Top-3 features, % of full | 68.8% (+7,717 of +11,210) | same |
| Full advantage table across all swept costs | see DECISIONS.md 2026-09-03 ablation entry | same |
| Restricted models exceed full above ₹2,000 | 101.5%–101.9% | same |

## LLM extraction comparison

| Claim | Value | Command |
|---|---|---|
| n=60 paired difference (TF-IDF − LLM) | +0.1624 (95% CI −0.0648 to +0.3858, includes 0) | `python -m eval.run_extraction_comparison` (no key — committed recording) |
| TF-IDF per-fold AUC, n=60 | 0.5104 | same |
| TF-IDF per-fold AUC, n=3,000 | 0.6479 | same |
| LLM per-fold AUC, n=60 (recorded, reproduces exactly) | 0.4211 | same |
| Each arm vs. chance, n=60 | TF-IDF +0.0392 (n.d.), LLM −0.1232 (n.d.) | same |
| **[KEY]** Raised-n result | **NOT RUN** — blocked by the same daily token budget exhaustion as the grounding gate (DECISIONS.md 2026-09-03). The n=60 paired result above stands as the recorded finding; raising n addresses only the sample-size limit, not the separate closed-template eval-design limit (see README "Limits") | `python -m eval.run_llm_normalization_quality --n-rows <N> --seed 0` then commit the output and wire it into `eval/run_extraction_comparison.py`, once budget allows |

## Grounding gate

| Claim | Value | Command |
|---|---|---|
| Break-even review cost by false-flag rate (0%–20%) | ₹200 down to ₹52 | `pytest tests/test_eval_review_cost.py -v` |
| Required false-flag budget at ₹150/review | 2.3% | same |
| Baseline's Class B template coverage | 6/12 | `pytest tests/test_eval_grounding_corpus.py -v` |
| **[KEY]** False-flag rate, Wilson interval, budget_verdict() | **NOT MEASURED** — blocked by the account's daily token budget (200,000 TPD for `openai/gpt-oss-20b`, exhausted mid-run; see DECISIONS.md's 2026-09-03 "key run: blocked" entry for the full arithmetic — a full n=250 measurement needs 750–1,250 calls, ≈2.25–3.75 days of budget at the account's current rate) | Single session (if budget allows): `python -m eval.run_grounding_draft --n-letters 250 --seed 0` then `python -m eval.run_grounding_eval`. **Multi-day** (checkpointed, safe to re-run any number of times — already-drafted positions are skipped): run the same `run_grounding_draft` command once per day until `already_drafted_positions(Path("data/reference/grounding_letters_seed0.csv"))` has length 250, then run `run_grounding_eval` once. |

## Withdrawn — recorded so the gap stays visible, not omitted

| Claim | Old value | Why withdrawn | Dated entry |
|---|---|---|---|
| Seed-42 realized AP | 0.4335 | Never computed from any commit this repo has made, at any point in its history — verified by checking out the Phase 1 commit itself | DECISIONS.md 2026-09-02 "oracle single-draw test", corrected 2026-09-03 |
| TF-IDF baseline | 0.6371 | No code, no recorded n, no seed anywhere in the repo; turned out to be a large-sample (n≈3,000) measurement compared against a 60-item LLM run | DECISIONS.md 2026-09-02 "TF-IDF baseline had no code" |
| Cost-sweep advantage, configured point | +12,923 (≈0.75%) | Difference of medians on a paired design | DECISIONS.md 2026-09-02 "unpaired estimator" |
| Letter-drafting reliability (20/20, zero repairs) | — | Measured against `explanation_letter_v1` and the 4,000-char schema, both replaced by Phase 0 | DECISIONS.md 2026-09-02 "WITHDRAWN: letter-drafting reliability" |
| `AUC(days_between_purchase_and_dispute)`, n=300,000 | 0.3507 | No seed recorded; re-run at seed=42, now **0.3496** — not a withdrawal, a reproducibility fix | DECISIONS.md 2026-09-03 "unseeded large-n claims" |
| `AUC(amount, true_fraud)`, n=200,000 | 0.5998 | Same — re-run at seed=42, now **0.5978** | same |
| Human-cost overstatement at configured point | 174% → −9,553 | Computed from the pre-paired-estimator advantage figure; corrected value is 200.5% → −11,265 | DECISIONS.md 2026-09-03 stale-number audit |

---

## `make verify`

```
make verify
```

Regenerates every row above that does not need `LLM_API_KEY`: the full test
suite (leakage guard, generator fingerprint, golden fixtures, grounding-gate
structural tests), the model/business/cost-sensitivity/ablation harnesses at
headline scale (20 seeds × 15,000 rows), and the extraction comparison
against the committed n=60 recording. Takes a few minutes; makes no network
call.

`make verify-key` lists the **[KEY]**-marked commands above separately — it
does not run them, since they cost real API budget and this repository never
runs them unattended. Print them, read them, decide whether to spend the
calls.
