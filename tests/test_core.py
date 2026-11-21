
from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    MostFrequentItemModel,
    NGramModel,
    SmallLanguageModel
)


# --------------------------------------------------
# TEST MILESTONE 1
# --------------------------------------------------

def test_milestone_1():
    baskets = milestone_1(num_baskets=100)

    assert len(baskets) == 100
    assert all(isinstance(basket, list) for basket in baskets)
    assert all(len(basket) > 0 for basket in baskets)


# --------------------------------------------------
# TEST N-GRAM MODEL
# --------------------------------------------------

def test_milestone_2():
    baskets = [
        ["Bread", "Milk", "Butter"],
        ["Bread", "Milk", "Eggs"],
        ["Bread", "Milk", "Cheese"]
    ]

    model = milestone_2(baskets)

    assert isinstance(model, NGramModel)

    predictions = model.predict(["Bread"])

    assert isinstance(predictions, list)


# --------------------------------------------------
# TEST SMALL LANGUAGE MODEL
# --------------------------------------------------

def test_milestone_3():
    baskets = [
        ["Bread", "Milk", "Butter"],
        ["Bread", "Milk", "Eggs"],
        ["Bread", "Milk", "Cheese"],
        ["Tea", "Biscuits", "Sugar"]
    ]

    model = milestone_3(baskets)

    assert isinstance(model, SmallLanguageModel)

    predictions = model.predict(["Bread"])

    assert isinstance(predictions, list)


# --------------------------------------------------
# TEST BASELINE MODEL
# --------------------------------------------------

def test_baseline_model():
    baskets = [
        ["Bread", "Milk"],
        ["Bread", "Butter"],
        ["Bread", "Eggs"],
        ["Milk", "Butter"]
    ]

    model = MostFrequentItemModel()
    model.train(baskets)

    predictions = model.predict(["Bread"])

    assert isinstance(predictions, list)
    assert "Bread" not in predictions
    assert predictions[0] == "Milk"


# --------------------------------------------------
# TEST MILESTONE 4
# --------------------------------------------------

def test_milestone_4():
    baskets = [
        ["Bread", "Milk", "Butter"],
        ["Bread", "Milk", "Eggs"],
        ["Tea", "Biscuits", "Sugar"],
        ["Rice", "Sugar", "Milk"],
        ["Bread", "Butter", "Cheese"],
        ["Tea", "Coffee", "Biscuits"],
        ["Rice", "Milk", "Eggs"],
        ["Bread", "Milk", "Cheese"],
        ["Tea", "Sugar", "Biscuits"],
        ["Rice", "Butter", "Milk"]
    ]

    results = milestone_4(baskets)

    assert "Most Frequent Item (Baseline)" in results
    assert "N-Gram" in results
    assert "Small Language Model" in results

    assert len(results) == 3


# --------------------------------------------------
# TEST MILESTONE 5
# --------------------------------------------------

def test_milestone_5():
    results = {
        "Most Frequent Item (Baseline)": {
            "accuracy": 0.70,
            "mrr": 0.80,
            "total_predictions": 20
        },

        "N-Gram": {
            "accuracy": 0.80,
            "mrr": 0.90,
            "total_predictions": 20
        },

        "Small Language Model": {
            "accuracy": 0.85,
            "mrr": 0.92,
            "total_predictions": 20
        }
    }

    df = milestone_5(results)

    assert len(df) == 3

    assert "accuracy" in df.columns
    assert "mrr" in df.columns
    assert "total_predictions" in df.columns
