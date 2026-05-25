import random
from src.ml_signal import DecisionTree, RandomForestSignal

def test_random_forest_signal():
    # Simulate price history for training (500 ticks)
    # Adding a slight trend to ensure we get some > 0.0005 returns
    train_prices = [100.0]
    for _ in range(499):
        # random return between -0.001 and 0.0015
        ret = random.uniform(-0.001, 0.0015)
        train_prices.append(train_prices[-1] * (1 + ret))

    model = RandomForestSignal(n_trees=5)
    model.fit(train_prices)

    assert len(model.trees) == 5, "Should have 5 trees after fitting"

    # Simulate price history for testing (200 ticks)
    test_prices = [100.0]
    for _ in range(199):
        ret = random.uniform(-0.0015, 0.001)
        test_prices.append(test_prices[-1] * (1 + ret))

    # Predict on the test set incrementally
    # Need at least 11 prices to predict (idx >= 10)
    for i in range(11, len(test_prices) + 1):
        window = test_prices[:i]
        pred = model.predict(window)
        assert pred in ["BUY", "SELL", "HOLD"], f"Invalid prediction: {pred}"

def test_decision_tree():
    # Basic logic check of Decision Tree with synthetic data
    X = [
        [0.1, 0.2],
        [0.4, 0.5],
        [-0.1, -0.2],
        [-0.4, -0.5],
        [0.0, 0.0]
    ]
    y = [1, 1, -1, -1, 0]

    dt = DecisionTree(max_depth=3, min_samples_split=2)
    dt.fit(X, y)

    for i, x in enumerate(X):
        pred = dt.predict(x)
        assert pred in [1, -1, 0]
