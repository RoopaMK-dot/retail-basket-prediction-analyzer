
import random
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ITEMS = [
    "Milk", "Bread", "Butter", "Eggs", "Cheese",
    "Rice", "Dal", "Sugar", "Tea", "Coffee",
    "Chips", "Juice", "Biscuits", "Soap", "Shampoo"
]

PLANTED_PATTERNS = [
    ["Milk", "Bread", "Butter"],
    ["Rice", "Dal"],
    ["Tea", "Biscuits"],
    ["Bread", "Eggs"],
    ["Coffee", "Sugar"]
]


def milestone_1(n=1000, seed=42):
    """Generate synthetic retail shopping baskets."""
    rng = random.Random(seed)
    baskets = []

    for _ in range(n):
        basket = set()

        if rng.random() < 0.7:
            basket.update(rng.choice(PLANTED_PATTERNS))

        for item in ITEMS:
            if rng.random() < 0.12:
                basket.add(item)

        if not basket:
            basket.add(rng.choice(ITEMS))

        baskets.append(sorted(basket))

    Path("data").mkdir(exist_ok=True)
    pd.DataFrame({
        "basket_id": range(1, n + 1),
        "items": [", ".join(b) for b in baskets]
    }).to_csv("data/baskets.csv", index=False)

    return baskets


class NGramModel:
    """Frequency-based next-item prediction model."""

    def __init__(self):
        self.counts = defaultdict(Counter)

    def train(self, baskets):
        for basket in baskets:
            for i in range(1, len(basket)):
                context = tuple(basket[:i])
                self.counts[context][basket[i]] += 1

    def predict(self, context, top_k=5):
        context = tuple(context)

        while context:
            if context in self.counts:
                return [
                    item for item, _ in
                    self.counts[context].most_common(top_k)
                ]
            context = context[1:]

        return []


def milestone_2(baskets=None):
    """Train and return an N-gram model."""
    baskets = baskets or milestone_1()
    model = NGramModel()
    model.train(baskets)
    return model


class SmallLanguageModel:
    """A lightweight neural next-item language model."""

    def __init__(self, vocab, seed=42):
        self.vocab = list(vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.rng = np.random.default_rng(seed)

        size = len(self.vocab)
        self.embedding = self.rng.normal(0, 0.1, (size, 16))
        self.W = self.rng.normal(0, 0.1, (16, size))
        self.b = np.zeros(size)

    def train(self, baskets, epochs=35, lr=0.05):
        examples = []

        for basket in baskets:
            for i in range(1, len(basket)):
                context = basket[:i]
                target = self.index[basket[i]]
                x = np.mean(
                    [self.embedding[self.index[w]] for w in context],
                    axis=0
                )
                examples.append((x, target))

        for _ in range(epochs):
            for x, target in examples:
                logits = x @ self.W + self.b
                logits -= np.max(logits)
                probs = np.exp(logits)
                probs /= probs.sum()

                probs[target] -= 1

                grad_w = np.outer(x, probs)
                grad_b = probs

                self.W -= lr * grad_w
                self.b -= lr * grad_b

        return self

    def predict(self, context, top_k=5):
        valid = [w for w in context if w in self.index]

        if not valid:
            return self.vocab[:top_k]

        x = np.mean(
            [self.embedding[self.index[w]] for w in valid],
            axis=0
        )

        logits = x @ self.W + self.b
        order = np.argsort(logits)[::-1]

        return [self.vocab[i] for i in order[:top_k]]


def milestone_3(baskets=None):
    """Train the small neural language model."""
    baskets = baskets or milestone_1()
    vocab = sorted(set(item for b in baskets for item in b))

    model = SmallLanguageModel(vocab)
    model.train(baskets)
    return model


def evaluate(model, baskets):
    """Calculate next-item prediction metrics."""
    correct = 0
    reciprocal_rank = 0
    total = 0

    for basket in baskets:
        for i in range(1, len(basket)):
            context = basket[:i]
            actual = basket[i]

            predictions = model.predict(context, top_k=5)

            total += 1

            if predictions and predictions[0] == actual:
                correct += 1

            if actual in predictions:
                reciprocal_rank += 1 / (predictions.index(actual) + 1)

    return {
        "accuracy": correct / total if total else 0,
        "mrr": reciprocal_rank / total if total else 0,
        "total_predictions": total
    }



class MostFrequentItemModel:
    """Baseline: recommend the most frequently purchased items."""

    def __init__(self):
        self.frequencies = Counter()

    def train(self, baskets):
        for basket in baskets:
            self.frequencies.update(basket)
        return self

    def predict(self, context, top_k=5):
        already_selected = set(context)

        return [
            item for item, count in self.frequencies.most_common()
            if item not in already_selected
        ][:top_k]

def milestone_4(baskets=None):
    """Evaluate baseline, N-gram and neural models."""
    baskets = baskets or milestone_1()

    split = int(len(baskets) * 0.8)
    train_data = baskets[:split]
    test_data = baskets[split:]

    baseline = MostFrequentItemModel().train(train_data)
    ngram = milestone_2(train_data)
    neural = milestone_3(train_data)

    return {
        "Most Frequent Item (Baseline)": evaluate(baseline, test_data),
        "N-Gram": evaluate(ngram, test_data),
        "Small Language Model": evaluate(neural, test_data)
    }


def milestone_5(results=None):
    """Compare models and save results."""
    results = results or milestone_4()

    df = pd.DataFrame(results).T
    print("\n========== MODEL COMPARISON ==========")
    print(df.to_string())
    print("======================================")

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/model_comparison.csv")

    return df
