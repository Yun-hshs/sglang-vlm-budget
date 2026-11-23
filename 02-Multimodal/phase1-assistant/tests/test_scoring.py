from vl_assistant.scoring import relaxed_correct, summarize


def test_relaxed_numeric_and_text():
    assert relaxed_correct("105", ["100"])
    assert not relaxed_correct("105.1", ["100"])
    assert relaxed_correct("0.50", ["50%"])
    assert relaxed_correct("0.0", ["0"])
    assert not relaxed_correct("0.001", ["0"])
    assert relaxed_correct("BLUE", ["red", "blue"])
    assert not relaxed_correct("Answer: blue", ["blue"])


def test_failed_requests_count_in_denominator():
    scores = summarize([
        {"subset": "human", "status": "ok", "correct": True},
        {"subset": "human", "status": "error", "correct": False},
    ])
    assert scores["human"]["relaxed_accuracy"] == 0.5
    assert scores["overall"]["failed"] == 1
