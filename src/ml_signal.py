import math
import random

class DecisionTree:
    def __init__(self, max_depth=5, min_samples=10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def fit(self, X: list[list[float]], y: list[int]):
        dataset = [X[i] + [y[i]] for i in range(len(X))]
        self.tree = self._get_best_split(dataset)
        self._split(self.tree, 1)

    def _get_best_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None

        if not dataset:
            return {'index': 0, 'value': 0, 'groups': ([], [])}

        n_features = len(dataset[0]) - 1
        features = list(range(n_features))
        random.shuffle(features)
        num_features = max(1, int(math.sqrt(n_features)))
        features = features[:num_features]

        for index in features:
            unique_values = set(row[index] for row in dataset)
            for value in unique_values:
                left, right = [], []
                for row in dataset:
                    if row[index] < value:
                        left.append(row)
                    else:
                        right.append(row)

                n_instances = float(len(left) + len(right))
                gini = 0.0
                for group in (left, right):
                    size = float(len(group))
                    if size == 0:
                        continue
                    score = 0.0
                    for class_val in class_values:
                        p = [r[-1] for r in group].count(class_val) / size
                        score += p * p
                    gini += (1.0 - score) * (size / n_instances)

                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, value, gini, (left, right)

        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _to_terminal(self, group):
        if not group:
            return 0
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def _split(self, node, depth):
        if node is None or node.get('groups') is None:
            return
        left, right = node['groups']
        del(node['groups'])

        if not left or not right:
            node['left'] = node['right'] = self._to_terminal(left + right)
            return

        if depth >= self.max_depth:
            node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
            return

        if len(left) <= self.min_samples:
            node['left'] = self._to_terminal(left)
        else:
            node['left'] = self._get_best_split(left)
            self._split(node['left'], depth + 1)

        if len(right) <= self.min_samples:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self._get_best_split(right)
            self._split(node['right'], depth + 1)

    def predict(self, x: list[float]) -> int:
        if self.tree is None:
            return 0
        return self._predict_node(self.tree, x)

    def _predict_node(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict_node(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict_node(node['right'], row)
            else:
                return node['right']

class RandomForestSignal:
    def __init__(self, n_trees=10, max_depth=5, feature_cols=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.feature_cols = feature_cols
        self.trees = []

    def _calculate_features(self, prices: list[float], i: int):
        ret_1 = (prices[i] - prices[i-1]) / prices[i-1] if i >= 1 else 0.0
        ret_5 = (prices[i] - prices[i-5]) / prices[i-5] if i >= 5 else 0.0
        ret_10 = (prices[i] - prices[i-10]) / prices[i-10] if i >= 10 else 0.0

        vol_5 = 0.0
        if i >= 5:
            rets = [(prices[j] - prices[j-1])/prices[j-1] for j in range(i-4, i+1)]
            mean_ret = sum(rets) / 5.0
            vol_5 = math.sqrt(sum((r - mean_ret)**2 for r in rets) / 5.0)

        vol_10 = 0.0
        if i >= 10:
            rets = [(prices[j] - prices[j-1])/prices[j-1] for j in range(i-9, i+1)]
            mean_ret = sum(rets) / 10.0
            vol_10 = math.sqrt(sum((r - mean_ret)**2 for r in rets) / 10.0)

        return [ret_1, ret_5, ret_10, vol_5, vol_10]

    def fit(self, price_history: list[float]):
        X = []
        y = []

        for i in range(10, len(price_history) - 1):
            features = self._calculate_features(price_history, i)
            next_ret = (price_history[i+1] - price_history[i]) / price_history[i]

            if next_ret > 0.0005:
                label = 1
            elif next_ret < -0.0005:
                label = -1
            else:
                label = 0

            X.append(features)
            y.append(label)

        if not X:
            return

        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples=10)
            n_samples = len(X)
            X_sample = []
            y_sample = []
            for _ in range(n_samples):
                idx = random.randint(0, n_samples - 1)
                X_sample.append(X[idx])
                y_sample.append(y[idx])
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, price_history: list[float]) -> str:
        if len(price_history) < 11 or not self.trees:
            return "HOLD"

        features = self._calculate_features(price_history, len(price_history) - 1)

        votes = []
        for tree in self.trees:
            votes.append(tree.predict(features))

        majority_vote = max(set(votes), key=votes.count)

        if majority_vote == 1:
            return "BUY"
        elif majority_vote == -1:
            return "SELL"
        else:
            return "HOLD"

if __name__ == "__main__":
    def generate_gbm(n, S0=100.0, mu=0.0001, sigma=0.005):
        prices = [S0]
        for _ in range(n - 1):
            Z = random.gauss(0, 1)
            next_S = prices[-1] * math.exp((mu - 0.5 * sigma**2) + sigma * Z)
            prices.append(next_S)
        return prices

    print("Generating 500 training prices...")
    train_prices = generate_gbm(500)

    rf_signal = RandomForestSignal(n_trees=10, max_depth=5, feature_cols=5)
    print("Training RandomForestSignal...")
    rf_signal.fit(train_prices)

    print("Generating 200 testing prices...")
    test_prices = generate_gbm(200)

    correct = 0
    total = 0

    print("Evaluating model...")
    for i in range(10, len(test_prices) - 1):
        history = test_prices[:i+1]
        next_ret = (test_prices[i+1] - test_prices[i]) / test_prices[i]

        if next_ret > 0.0005:
            true_label = "BUY"
        elif next_ret < -0.0005:
            true_label = "SELL"
        else:
            true_label = "HOLD"

        pred_label = rf_signal.predict(history)

        if pred_label == true_label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy on 200 test prices: {accuracy:.2%}")