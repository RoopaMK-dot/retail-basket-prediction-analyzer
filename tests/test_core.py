
from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5
)


def test_milestone_1():
    baskets = milestone_1(n=100)
    assert len(baskets) == 100
    assert all(isinstance(b, list) for b in baskets)


def test_milestone_2():
    baskets = milestone_1(n=100)
    model = milestone_2(baskets)
    assert len(model.counts) > 0


def test_milestone_3():
    baskets = milestone_1(n=100)
    model = milestone_3(baskets)
    assert len(model.vocab) > 0


def test_milestone_4():
    baskets = milestone_1(n=100)
    results = milestone_4(baskets)
    assert "N-Gram" in results
    assert "Small Language Model" in results


def test_milestone_5():
    results = {
        "N-Gram": {
            "accuracy": 0.8,
            "mrr": 0.9,
            "total_predictions": 20
        },
        "Small Language Model": {
            "accuracy": 0.85,
            "mrr": 0.92,
            "total_predictions": 20
        }
    }

    df = milestone_5(results)
    assert len(df) == 2
