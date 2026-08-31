"""customer_communication_log: a noisy, class-overlapping reading of true_fraud
(via claim/opening phrasing rates) and relationship_genuineness (via tone,
inclusion of detail, and messiness) - see GENERATOR.md §3. Session-2 fix for
defect 1: the previous three-fixed-templates-per-branch design let a string
match recover true_fraud exactly. Every phrase pool below is shared across both
true_fraud values, just drawn at different rates, so no fixed string is
class-exclusive.
"""

import numpy as np

from disputedesk.generator.config import GeneratorConfig

_OPENINGS = [
    "I am writing to dispute a transaction on my account.",
    "I do not recognize this charge on my statement.",
    "I need to report an issue with a recent charge.",
    "This charge needs to be looked into.",
]
_CLAIMS = [
    "I did not authorize this transaction.",
    "Someone must have used my card without my permission.",
    "I don't recall approving this purchase.",
    "This does not match any purchase I remember making.",
]
_DETAILS = [
    "My card may have been compromised recently.",
    "A family member sometimes has access to my card.",
    "I have already contacted my bank about this.",
    "I travel frequently and my account activity looks unusual to me.",
]
_SIGNOFFS_POLITE = [
    "Please investigate and let me know what you find. Thank you.",
    "I appreciate your help resolving this quickly.",
    "Thank you for looking into this matter.",
]
_SIGNOFFS_TERSE = ["Fix this.", "Please refund me.", "Let me know."]
_IRRELEVANT = [
    "By the way, do you have any upcoming sales?",
    "Also, my delivery address recently changed.",
    "Unrelated, but your app keeps crashing on my phone.",
]
_NEAR_EMPTY = ["please refund", "?", "n/a", "see above"]


def _weighted_slot(
    rng: np.random.Generator,
    true_fraud: np.ndarray,
    phrases: list[str],
    weights_fraud: tuple[float, ...],
    weights_genuine: tuple[float, ...],
) -> np.ndarray:
    result = np.empty(true_fraud.shape[0], dtype=object)
    fraud_mask = true_fraud
    genuine_mask = ~true_fraud
    result[fraud_mask] = rng.choice(phrases, size=int(fraud_mask.sum()), p=weights_fraud)
    result[genuine_mask] = rng.choice(phrases, size=int(genuine_mask.sum()), p=weights_genuine)
    return result


def _content_slots(
    rng: np.random.Generator, true_fraud: np.ndarray, config: GeneratorConfig
) -> dict[str, np.ndarray]:
    return {
        "opening": _weighted_slot(
            rng,
            true_fraud,
            _OPENINGS,
            config.comms_opening_weights_fraud,
            config.comms_opening_weights_genuine,
        ),
        "claim": _weighted_slot(
            rng,
            true_fraud,
            _CLAIMS,
            config.comms_claim_weights_fraud,
            config.comms_claim_weights_genuine,
        ),
        "detail": _weighted_slot(
            rng,
            true_fraud,
            _DETAILS,
            config.comms_detail_weights_fraud,
            config.comms_detail_weights_genuine,
        ),
    }


def _tone_and_noise_slots(
    rng: np.random.Generator, relationship_genuineness: np.ndarray, config: GeneratorConfig
) -> dict[str, np.ndarray]:
    n = relationship_genuineness.shape[0]

    p_polite = config.comms_signoff_polite_base + config.comms_signoff_polite_relationship_scale * (
        relationship_genuineness
    )
    polite_choice = rng.random(n) < p_polite
    polite = np.array(_SIGNOFFS_POLITE)[rng.integers(0, len(_SIGNOFFS_POLITE), size=n)]
    terse = np.array(_SIGNOFFS_TERSE)[rng.integers(0, len(_SIGNOFFS_TERSE), size=n)]

    p_detail = (
        config.comms_detail_inclusion_base
        + config.comms_detail_inclusion_relationship_scale * (relationship_genuineness)
    )

    return {
        "signoff": np.where(polite_choice, polite, terse),
        "include_detail": rng.random(n) < p_detail,
        "near_empty": rng.random(n) < config.comms_near_empty_prob,
        "near_empty_choice": np.array(_NEAR_EMPTY)[rng.integers(0, len(_NEAR_EMPTY), size=n)],
        "irrelevant": rng.random(n) < config.comms_irrelevant_detail_prob,
        "irrelevant_choice": np.array(_IRRELEVANT)[rng.integers(0, len(_IRRELEVANT), size=n)],
    }


def _prepare_slots(
    rng: np.random.Generator,
    true_fraud: np.ndarray,
    relationship_genuineness: np.ndarray,
    config: GeneratorConfig,
) -> dict[str, np.ndarray]:
    return {
        **_content_slots(rng, true_fraud, config),
        **_tone_and_noise_slots(rng, relationship_genuineness, config),
    }


def _apply_messiness(
    rng: np.random.Generator,
    texts: np.ndarray,
    relationship_genuineness: np.ndarray,
    config: GeneratorConfig,
) -> np.ndarray:
    n = texts.shape[0]
    typo_prob = config.comms_typo_base_prob + config.comms_typo_relationship_scale * (
        1.0 - relationship_genuineness
    )
    typo_draw = rng.random(n) < typo_prob
    typo_fraction = rng.random(n)
    lowercase_draw = rng.random(n) < config.comms_lowercase_prob

    out = np.empty(n, dtype=object)
    for i in range(n):
        text = texts[i]
        if lowercase_draw[i]:
            text = text.lower()
        if typo_draw[i] and len(text) > 4:
            pos = int(typo_fraction[i] * (len(text) - 1))
            text = text[:pos] + text[pos + 1 :]
        out[i] = text
    return out


def generate_communication_log(
    rng: np.random.Generator,
    true_fraud: np.ndarray,
    relationship_genuineness: np.ndarray,
    config: GeneratorConfig,
) -> np.ndarray:
    slots = _prepare_slots(rng, true_fraud, relationship_genuineness, config)
    n = true_fraud.shape[0]
    texts = np.empty(n, dtype=object)
    for i in range(n):
        if slots["near_empty"][i]:
            texts[i] = slots["near_empty_choice"][i]
            continue
        parts = [slots["opening"][i], slots["claim"][i]]
        if slots["include_detail"][i]:
            parts.append(slots["detail"][i])
        parts.append(slots["signoff"][i])
        text = " ".join(parts)
        if slots["irrelevant"][i]:
            text = text + " " + slots["irrelevant_choice"][i]
        texts[i] = text
    return _apply_messiness(rng, texts, relationship_genuineness, config)
