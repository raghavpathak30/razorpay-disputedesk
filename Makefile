.PHONY: verify verify-key lint test

# Regenerates every numeric claim in NUMBERS.md that does not need
# LLM_API_KEY. Built 2026-09-03 (Phase 3) so a judge - or anyone else - can
# re-derive every number in this submission with one command, from a clean
# clone, with no secrets. See NUMBERS.md for what each target maps to.
verify: lint test
	python -m eval.run_harness --n-seeds 20 --n-rows 15000
	python -m eval.run_business_harness --n-seeds 20 --n-rows 15000
	python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000
	python -m eval.run_ablation --n-seeds 20 --n-rows 15000
	python -m eval.run_extraction_comparison
	@echo ""
	@echo "verify complete. Everything above regenerated with no API key and"
	@echo "no network call. Numbers needing LLM_API_KEY are NOT included -"
	@echo "run 'make verify-key' to see which, and why they are separate."

lint:
	ruff check .
	ruff format --check .

test:
	pytest

# Lists the commands that need LLM_API_KEY instead of running them. This
# repository never spends API budget unattended - each command below makes
# real network calls and costs real (free-tier) usage, so it is printed for
# a human to read and decide, not executed by this target.
verify-key:
	@echo "The following need LLM_API_KEY (Groq) and are NOT run by 'make verify':"
	@echo ""
	@echo "As of 2026-09-03 both were attempted and blocked by the account's daily"
	@echo "token budget (200,000 TPD), not by a missing key - a full n=250 grounding-"
	@echo "gate run alone needs 2.25-3.75 days of that budget at this account's rate."
	@echo "See DECISIONS.md's 2026-09-03 'key run: blocked' entry before re-running."
	@echo ""
	@echo "  python -m eval.run_grounding_draft --n-letters 250 --seed 0"
	@echo "      drafts the grounding-gate corpus (~500 calls). Commits"
	@echo "      data/reference/grounding_letters_seed0.csv."
	@echo ""
	@echo "  python -m eval.run_grounding_eval"
	@echo "      grades the corpus against the gate (~700-750 calls)."
	@echo "      Needs the committed letters file above first."
	@echo ""
	@echo "  python -m eval.run_llm_normalization_quality --n-rows <N> --seed 0"
	@echo "      re-measures the LLM extraction arm at a chosen n. Commit the"
	@echo "      output CSV to data/reference/ and wire it into"
	@echo "      eval/run_extraction_comparison.py to make the paired result"
	@echo "      reproduce without a key afterward, as the n=60 recording does."
	@echo ""
	@echo "See NUMBERS.md for the values these commands last produced, and"
	@echo "DECISIONS.md for the dated entry recording when each was run."
