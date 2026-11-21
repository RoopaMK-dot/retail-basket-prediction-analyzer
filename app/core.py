
import random
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------
# MILESTONE 1: GENERATE RETAIL BASKET DATASET
# --------------------------------------------------

def milestone_1(num_baskets=1000, seed=42):
    """Generate synthetic retail shopping baskets."""

    random.seed(seed)

    products = [
        "Bread", "Milk", "Butter", "Eggs", "Cheese",
        "Rice", "Sugar", "Tea", "Coffee", "Biscuits",
        "Apples", "Bananas", "Chips", "Juice", "Soap"
    ]

    baskets = []

    for _ in range(num_baskets):
        basket = random.sample(products, random.randint(2, 6))

        # Add some frequently occurring product combinations.
        if random.random() < 0.4:
            basket = list(set(basket + ["Bread", "Milk"]))

        if random.random() < 0.3:
            basket = list(set(basket + ["Tea", "Biscuits"]))

        baskets.append(sorted(basket))

    df = pd.DataFrame({
        "basket_id": range(1, len(baskets) + 1),
        "items": [",".join(basket) for basket in baskets]
    })

    df.to_csv("data/baskets.csv", index=False)

    print(f"Generated {len(baskets)} baskets.")

    return baskets


# --------------------------------------------------
# N-GRAM MODEL
# --------------------------------------------------

class NGramModel:
    """Predict the next item using previous basket items."""

    def __init__(self):
        self.ngrams = defaultdict(Counter)

    def train(self, baskets):
        for basket in baskets:
            for i in range(len(basket)):
                context = tuple(basket[:i])
                next_item = basket[i]
                self.ngrams[context][next_item] += 1

        return self

    def predict(self, context, top_k=5):
        context = tuple(context)

        # Find the longest available context.
        while context not in self.ngrams and context:
            context = context[1:]

        if context not in self.ngrams:
            return []

        return [
            item
            for item, count in self.ngrams[context].most_common(top_k)
            if item not in context
        ]


def milestone_2(baskets=None):
    """Train the N-Gram model."""

    baskets = baskets or milestone_1()

    model = NGramModel()
    model.train(baskets)

    return model


# --------------------------------------------------
# SMALL NEURAL LANGUAGE MODEL
# --------------------------------------------------

class SmallLanguageModel:
    """A lightweight neural next-item prediction model."""

    def __init__(self, embedding_dim=16, learning_rate=0.05, epochs=30):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.items = []
        self.item_to_id = {}
        self.embeddings = None
        self.W = None
        self.b = None

    def train(self, baskets):
        self.items = sorted(set(
            item for basket in baskets for item in basket
        ))

        self.item_to_id = {
            item: index for index, item in enumerate(self.items)
        }

        vocab_size = len(self.items)

        if vocab_size == 0:
            return self

        rng = np.random.default_rng(42)

        self.embeddings = rng.normal(
            0, 0.1, (vocab_size, self.embedding_dim)
        )

        self.W = rng.normal(
            0, 0.1, (self.embedding_dim, vocab_size)
        )

        self.b = np.zeros(vocab_size)

        training_examples = []

        for basket in baskets:
            for i in range(1, len(basket)):
                context = basket[:i]
                target = basket[i]

                training_examples.append((context, target))

        for _ in range(self.epochs):
            for context, target in training_examples:

                context_ids = [
                    self.item_to_id[item]
                    for item in context
                    if item in self.item_to_id
                ]

                if not context_ids:
                    continue

                x = np.mean(self.embeddings[context_ids], axis=0)

                target_id = self.item_to_id[target]

                logits = np.dot(x, self.W) + self.b

                logits -= np.max(logits)

                probabilities = np.exp(logits)
                probabilities /= np.sum(probabilities)

                probabilities[target_id] -= 1

                grad_W = np.outer(x, probabilities)

                self.W -= self.learning_rate * grad_W
                self.b -= self.learning_rate * probabilities

        return self

    def predict(self, context, top_k=5):
        if self.W is None or len(self.items) == 0:
            return []

        context_ids = [
            self.item_to_id[item]
            for item in context
            if item in self.item_to_id
        ]

        if not context_ids:
            return []

        x = np.mean(self.embeddings[context_ids], axis=0)

        scores = np.dot(x, self.W) + self.b

        ranked_ids = np.argsort(scores)[::-1]

        selected = set(context)

        predictions = []

        for index in ranked_ids:
            item = self.items[index]

            if item not in selected:
                predictions.append(item)

            if len(predictions) == top_k:
                break

        return predictions


def milestone_3(baskets=None):
    """Train the Small Language Model."""

    baskets = baskets or milestone_1()

    model = SmallLanguageModel()
    model.train(baskets)

    return model


# --------------------------------------------------
# BASELINE MODEL
# --------------------------------------------------

class MostFrequentItemModel:
    """Recommend the most frequently purchased items."""

    def __init__(self):
        self.frequencies = Counter()

    def train(self, baskets):
        for basket in baskets:
            self.frequencies.update(basket)

        return self

    def predict(self, context, top_k=5):
        already_selected = set(context)

        return [
            item
            for item, count in self.frequencies.most_common()
            if item not in already_selected
        ][:top_k]


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------


def evaluate(model, test_baskets):
    """Evaluate models using Accuracy@1 and Mean Reciprocal Rank."""

    correct = 0
    reciprocal_rank_sum = 0
    total_predictions = 0

    for basket in test_baskets:

        for i in range(1, len(basket)):

            # Use previous items as context.
            context = basket[:i]

            # The next item is the actual target.
            actual_item = basket[i]

            # Predict the next item.
            predictions = model.predict(context, top_k=5)

            total_predictions += 1

            if predictions:

                # Accuracy@1
                if predictions[0] == actual_item:
                    correct += 1

                # Mean Reciprocal Rank
                if actual_item in predictions:
                    rank = predictions.index(actual_item) + 1
                    reciprocal_rank_sum += 1 / rank

    accuracy = (
        correct / total_predictions
        if total_predictions > 0 else 0
    )

    mrr = (
        reciprocal_rank_sum / total_predictions
        if total_predictions > 0 else 0
    )

    return {
        "accuracy": accuracy,
        "mrr": mrr,
        "total_predictions": total_predictions
    }

# --------------------------------------------------
# MILESTONE 4: EVALUATE ALL THREE MODELS
# --------------------------------------------------

def milestone_4(baskets=None):
    """Evaluate baseline, N-Gram and neural models."""

    baskets = baskets if baskets is not None else milestone_1()

    if len(baskets) < 2:
        raise ValueError("At least two baskets are required.")

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
# --------------------------------------------------
# MILESTONE 5: MODEL COMPARISON
# --------------------------------------------------

def milestone_5(results):
    """Compare models and save results to CSV."""

    import pandas as pd

    df = pd.DataFrame.from_dict(results, orient="index")

    print("\n========== MODEL COMPARISON ==========")
    print(df)
    print("======================================")

    df.to_csv("results/model_comparison.csv")

    return df