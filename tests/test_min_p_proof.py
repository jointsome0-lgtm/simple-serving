"""The verdict of the card's proof: a wrong pass would put a filter that does nothing in front of the stories."""

import math

from simple_serving.min_p_proof import ENOUGH_BELOW, ENOUGH_TOKENS, below, filtered, follows


def test_a_token_counts_as_below_by_its_probability_after_the_temperature() -> None:
    def token(ratio: float, temperature: float = 0.8) -> dict[str, object]:
        return {"logprob": -1 + temperature * math.log(ratio), "top_logprobs": [{"logprob": -1}]}

    assert below([token(1), token(0.08), token(0.081)], 0.08, 0.8) == 0
    assert below([token(0.07), token(0.01), token(1)], 0.08, 0.8) == 2


def test_the_filter_passes_only_where_the_control_shows_what_it_removes_and_tokens_were_drafted() -> None:
    enough = {"tokens": ENOUGH_TOKENS, "below": 0, "accepted": 100}
    shown = enough | {"below": ENOUGH_BELOW}
    assert filtered(shown, enough) == "passed"
    assert filtered(shown, enough | {"below": 1}) == "failed"
    assert filtered(enough, enough) == "inconclusive"  # the control removed nothing to find
    assert filtered(shown, enough | {"accepted": 0}) == "inconclusive"  # nothing was drafted
    assert filtered(shown, enough | {"tokens": 0}) == "inconclusive"  # vLLM refused or answered nothing


def test_an_answer_follows_the_schema_or_does_not() -> None:
    assert follows('{"title": "t", "people": [{"name": "n", "age": 3}]}')
    assert not follows('{"title": "t"}') and not follows('{"title": "t", "people": [{"name": "n", "age": true}]}')
    assert not follows("not JSON")
