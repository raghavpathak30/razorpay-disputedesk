"""MiniCheck: grounding-cascade stage 2 (CLAUDE.md Day-2). Scores one sentence
against a text rendering of the dispute record, independent of and never
calling the Groq-based generative gate in `grounding.py`.

Same shape as `llm.py`'s `LLMClient`/`FakeLLMClient` split, for the same
reason: everything in `grounding_cascade.py` that needs a MiniCheck score
goes through `MiniCheckScorer`, never a concrete model directly, so tests
substitute `FakeMiniCheckClient` and load no 615MB model and make no
inference call at all.

**Model and input/output contract**, pinned here from the Phase 1 recon
against `lytang/MiniCheck-Flan-T5-Large`'s own source
(`Liyan06/MiniCheck/minicheck/inference.py`) and verified live against the
`nvhf/MiniCheck-Flan-T5-Large-Q6_K-GGUF` conversion this client loads:

    input  = "predict: " + document + <eos> + claim   (fed to the ENCODER)
    decode = one step, starting from the model's decoder_start_token
    output = decoder step-1 logits at token ids [3, 209] - id 209 is the
             single-token encoding of "1" in this vocab, id 3 is the leading
             piece of the two-token encoding of "0" - softmax over just
             those two, P(supported) = softmax[1]

T5 is encoder-decoder. llama-cpp-python's high-level `Llama.eval`/
`create_completion` never calls `llama_encode` (checked directly against
`llama_cpp/llama.py` and `_internals.py` in this project's installed
version) - it only ever calls `llama_decode`, which for a real T5 GGUF means
the encoder cross-attention state is never populated. `_score_low_level`
below drives `llama_encode`/`llama_decode` directly instead of going through
`create_completion`.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

ZERO_TOKEN = 3
"""The leading piece of this vocab's two-token encoding of "0" - see the
module docstring. Not a standalone "0" token; comparing its logit against
`ONE_TOKEN`'s at decoder step 1 is what MiniCheck's own fine-tuning target
compares, per the source cited above."""

ONE_TOKEN = 209
"""The single-token encoding of "1" in this vocab - see the module
docstring."""


class MiniCheckUnavailableError(RuntimeError):
    """Raised when a real MiniCheck scorer cannot be constructed or cannot
    produce a score - missing model file, `llama-cpp-python` not installed,
    or an inference-time failure. The cascade's caller must treat this the
    same as any other stage-2 failure: withhold, never pass (SPEC.md §7 -
    the system degrades, it does not crash; `grounding_cascade.py` fails
    closed on this exception).
    """


class MiniCheckScorer(Protocol):
    def score(self, document: str, claim: str) -> float:
        """P(the record supports `claim`), in `[0, 1]`. Raises
        `MiniCheckUnavailableError` if no score could be produced - never
        returns a sentinel value for a failure, so a caller cannot mistake
        "could not score" for "scored low".
        """
        ...


class FakeMiniCheckClient:
    """Deterministic stand-in for tests: no model load, no inference call.
    Returns each entry of `responses` in order, one per call; repeats the
    last entry once exhausted (mirrors `llm.py`'s `FakeLLMClient`).
    """

    def __init__(self, responses: list[float]):
        if not responses:
            raise ValueError("FakeMiniCheckClient needs at least one response")
        self._responses = responses
        self._call_count = 0
        self.calls: list[tuple[str, str]] = []

    def score(self, document: str, claim: str) -> float:
        index = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        self.calls.append((document, claim))
        return self._responses[index]

    @property
    def call_count(self) -> int:
        return self._call_count


class NeverCallMiniCheckClient:
    """Spy that fails the test loudly if the cascade ever reaches stage 2 for
    a sentence stage 1 should have already resolved. Stronger than asserting
    `call_count == 0` after the fact: this raises at the call site, so a test
    failure points at exactly the cascade branch that wrongly fell through.
    """

    def score(self, document: str, claim: str) -> float:
        raise AssertionError(
            f"MiniCheck must not be called for a stage-1-resolved sentence: {claim!r}"
        )


def _softmax_pair(zero_logit: float, one_logit: float) -> float:
    import math

    hi = max(zero_logit, one_logit)
    e_zero = math.exp(zero_logit - hi)
    e_one = math.exp(one_logit - hi)
    return e_one / (e_zero + e_one)


class LlamaCppMiniCheckClient:
    """Real MiniCheck-Flan-T5-Large (770M, Q6_K GGUF) scorer via
    `llama-cpp-python`'s low-level encode/decode API. CPU-only on this
    project's hardware (no CUDA toolkit installed - see the Phase 1 recon);
    ~240ms/sentence measured on an RTX 4060 laptop's CPU path.

    Loads the model once, at construction, not per call - `score()` re-uses
    the same `Llama` context for every sentence in a letter.
    """

    def __init__(self, model_path: str, *, n_ctx: int = 2048):
        try:
            import llama_cpp
            from llama_cpp import Llama
            from llama_cpp._internals import LlamaBatch
        except ImportError as error:
            raise MiniCheckUnavailableError(
                "llama-cpp-python is not installed; MiniCheck cannot run"
            ) from error

        self._llama_cpp = llama_cpp
        self._LlamaBatch = LlamaBatch
        try:
            self._llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
        except Exception as error:  # noqa: BLE001 - any load failure must fail closed
            raise MiniCheckUnavailableError(
                f"could not load MiniCheck model at {model_path!r}: {error}"
            ) from error

        if not llama_cpp.llama_model_has_encoder(self._llm._model.model):
            raise MiniCheckUnavailableError(
                f"model at {model_path!r} has no encoder; is this really the T5 GGUF?"
            )

    def score(self, document: str, claim: str) -> float:
        try:
            return self._score_low_level(document, claim)
        except MiniCheckUnavailableError:
            raise
        except Exception as error:  # noqa: BLE001 - inference failure must fail closed
            raise MiniCheckUnavailableError(f"MiniCheck inference failed: {error}") from error

    def _score_low_level(self, document: str, claim: str) -> float:
        llama_cpp = self._llama_cpp
        llm = self._llm
        model_ptr = llm._model.model
        ctx_ptr = llm._ctx.ctx
        n_vocab = llm.n_vocab()

        eos_id = llm.token_eos()
        enc_ids = llm.tokenize(("predict: " + document).encode(), add_bos=False, special=False)
        enc_ids += [eos_id] + llm.tokenize(claim.encode(), add_bos=False, special=False)

        decoder_start = llama_cpp.llama_model_decoder_start_token(model_ptr)
        if decoder_start < 0:
            decoder_start = llama_cpp.llama_vocab_pad(llama_cpp.llama_model_get_vocab(model_ptr))

        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(ctx_ptr), True)

        enc_batch = self._LlamaBatch(
            n_tokens=max(len(enc_ids), 1), embd=0, n_seq_max=1, verbose=False
        )
        try:
            enc_batch.set_batch(enc_ids, n_past=0, logits_all=False)
            llm._ctx.encode(enc_batch)
        finally:
            enc_batch.close()

        dec_batch = self._LlamaBatch(n_tokens=1, embd=0, n_seq_max=1, verbose=False)
        try:
            dec_batch.set_batch([decoder_start], n_past=0, logits_all=True)
            llm._ctx.decode(dec_batch)
        finally:
            dec_batch.close()

        import numpy as np

        logits_ptr = llama_cpp.llama_get_logits_ith(ctx_ptr, 0)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
        return _softmax_pair(float(logits[ZERO_TOKEN]), float(logits[ONE_TOKEN]))
