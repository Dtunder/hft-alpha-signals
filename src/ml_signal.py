import random
import statistics
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: list[list], y: list[str]):
        self.tree = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, x: list) -> str:
        node = self.tree
        while isinstance(node, dict):
            if x[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

    def _gini(self, y_left, y_right):
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        def score(y_subset):
            if not y_subset: return 0.0
            counts = Counter(y_subset)
            return 1.0 - sum((count / len(y_subset)) ** 2 for count in counts.values())

        return (n_left / n_total) * score(y_left) + (n_right / n_total) * score(y_right)

    def _grow_tree(self, X, y, depth):
        n_samples = len(y)
        n_features = len(X[0])
        unique_classes = set(y)

        if (depth >= self.max_depth or n_samples < self.min_samples_split or len(unique_classes) == 1):
            return Counter(y).most_common(1)[0][0]

        best_gini = float('inf')
        best_criteria = None
        best_sets = None

        for feat_idx in range(n_features):
            thresholds = set([x[feat_idx] for x in X])
            for threshold in thresholds:
                left_indices = [i for i in range(n_samples) if X[i][feat_idx] <= threshold]
                right_indices = [i for i in range(n_samples) if X[i][feat_idx] > threshold]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                y_left = [y[i] for i in left_indices]
                y_right = [y[i] for i in right_indices]

                gini = self._gini(y_left, y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_criteria = {'feature_index': feat_idx, 'threshold': threshold}
                    best_sets = {
                        'left_X': [X[i] for i in left_indices],
                        'left_y': y_left,
                        'right_X': [X[i] for i in right_indices],
                        'right_y': y_right
                    }

        if best_gini == float('inf'):
            return Counter(y).most_common(1)[0][0]

        left_branch = self._grow_tree(best_sets['left_X'], best_sets['left_y'], depth + 1)
        right_branch = self._grow_tree(best_sets['right_X'], best_sets['right_y'], depth + 1)

        return {
            'feature_index': best_criteria['feature_index'],
            'threshold': best_criteria['threshold'],
            'left': left_branch,
            'right': right_branch
        }

class RandomForestSignal:
    def __init__(self, n_trees=10):
        self.n_trees = n_trees
        self.trees = []

    def _get_features(self, prices, index):
        if index < 10:
            return None

        p = prices[index]
        ret_1 = (p - prices[index - 1]) / prices[index - 1]
        ret_5 = (p - prices[index - 5]) / prices[index - 5]
        ret_10 = (p - prices[index - 10]) / prices[index - 10]

        returns_1_history = [
            (prices[j] - prices[j - 1]) / prices[j - 1]
            for j in range(index - 9, index + 1)
        ]

        vol_5 = statistics.stdev(returns_1_history[-5:]) if len(returns_1_history[-5:]) >= 2 else 0.0
        vol_10 = statistics.stdev(returns_1_history) if len(returns_1_history) >= 2 else 0.0

        return [ret_1, ret_5, ret_10, vol_5, vol_10]

    def fit(self, prices: list[float]):
        X = []
        y = []
        for i in range(10, len(prices) - 1):
            features = self._get_features(prices, i)
            if features is None:
                continue

            next_ret = (prices[i + 1] - prices[i]) / prices[i]
            if next_ret > 0.001:
                label = "BUY"
            elif next_ret < -0.001:
                label = "SELL"
            else:
                label = "HOLD"

            X.append(features)
            y.append(label)

        if not X:
            return self

        dataset = list(zip(X, y))
        n_samples = len(dataset)

        for _ in range(self.n_trees):
            bootstrap_sample = random.choices(dataset, k=n_samples)
            X_sample = [item[0] for item in bootstrap_sample]
            y_sample = [item[1] for item in bootstrap_sample]

            tree = DecisionTree(max_depth=5, min_samples_split=2)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, prices: list[float]) -> str:
        if not self.trees:
            return "HOLD"

        idx = len(prices) - 1
        features = self._get_features(prices, idx)

        if features is None:
            return "HOLD"

        predictions = [tree.predict(features) for tree in self.trees]
        return Counter(predictions).most_common(1)[0][0]
