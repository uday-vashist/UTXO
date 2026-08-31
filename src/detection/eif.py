"""Pure-Python implementation of the Extended Isolation Forest (EIH) anomaly detection model.

Uses random slope hyperplanes to avoid axis-parallel artifacts of standard Isolation Forests.
"""

import numpy as np
import pandas as pd


class ExtendedIsolationTree:
    """A single tree in the Extended Isolation Forest."""

    def __init__(
        self,
        X: np.ndarray,
        current_height: int,
        max_height: int,
        extension_level: int = 0,
    ):
        self.height = current_height
        self.size = len(X)
        self.left = None
        self.right = None
        self.w = None  # Normal vector defining split hyperplane orientation
        self.q = None  # Split intercept value
        self.is_leaf = False

        # Terminate if max height is reached or node contains only 1 sample
        if current_height >= max_height or self.size <= 1:
            self.is_leaf = True
            return

        n_features = X.shape[1]

        # 1. Generate random slope vector (coefficients from standard normal)
        w = np.random.normal(0.0, 1.0, n_features)

        # 2. Extension level handling:
        # ExtensionLevel = 0 means fully extended (all slopes can be non-zero).
        # ExtensionLevel = k means we restrict the slopes to have only (k) non-zero coefficients.
        # ExtensionLevel must be in range [0, n_features - 1]
        if 0 < extension_level < n_features:
            # Randomly select features to zero out
            zero_indices = np.random.choice(
                n_features, n_features - extension_level, replace=False
            )
            w[zero_indices] = 0.0

        # Normalize w to unit length
        w_norm = np.linalg.norm(w)
        if w_norm > 0.0:
            w = w / w_norm
        else:
            w[0] = 1.0  # Fallback in case of zero vector

        # 3. Project data points onto normal vector w
        p = X.dot(w)
        min_p = p.min()
        max_p = p.max()

        if min_p == max_p:
            self.is_leaf = True
            return

        # 4. Draw split point q uniformly from range of projected values
        q = np.random.uniform(min_p, max_p)

        # 5. Split data into left and right sub-trees
        left_indices = p <= q
        right_indices = p > q

        # Fallback to leaf if partition fails to split samples
        if not np.any(left_indices) or not np.any(right_indices):
            self.is_leaf = True
            return

        self.w = w
        self.q = q

        self.left = ExtendedIsolationTree(
            X[left_indices], current_height + 1, max_height, extension_level
        )
        self.right = ExtendedIsolationTree(
            X[right_indices], current_height + 1, max_height, extension_level
        )


def c_factor(n: int) -> float:
    """Average path length of an unsuccessful search in a Binary Search Tree (BST)

    with n nodes. Used to normalize the path lengths of isolation trees.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # Euler-Mascheroni constant (gamma) is ~0.5772156649
    return 2.0 * (np.log(n - 1.0) + 0.5772156649) - (2.0 * (n - 1.0) / n)


class ExtendedIsolationForest:
    """Extended Isolation Forest anomaly detection model implemented in pure Python."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        extension_level: int = 0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level
        self.random_state = random_state
        self.trees = []
        self.n_features = 0

    def fit(self, X: pd.DataFrame | np.ndarray) -> "ExtendedIsolationForest":
        """Fits the forest on the training data X."""
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X

        np.random.seed(self.random_state)
        n_samples, self.n_features = X_np.shape
        sample_size = min(self.max_samples, n_samples)
        max_height = int(np.ceil(np.log2(max(sample_size, 2))))

        self.trees = []
        for _ in range(self.n_estimators):
            # Select random subsample of indices
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X_np[indices]
            # Build and append tree
            tree = ExtendedIsolationTree(
                X_sample, 0, max_height, self.extension_level
            )
            self.trees.append(tree)

        return self

    def _path_length(self, x: np.ndarray, tree: ExtendedIsolationTree) -> float:
        """Computes depth path length of x in the tree."""
        if tree.is_leaf or tree.left is None or tree.right is None:
            return tree.height + c_factor(tree.size)

        # Project x onto the tree's split hyperplane normal vector w
        proj = x.dot(tree.w)
        if proj <= tree.q:
            return self._path_length(x, tree.left)
        else:
            return self._path_length(x, tree.right)

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Computes the raw decision scores (average path length scaled).

        Returns values in [-0.5, 0.5] range, where lower values are anomalous
        (conforms to scikit-learn's standard IsolationForest decision_function).
        """
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X

        n_samples = X_np.shape[0]
        path_lengths = np.zeros(n_samples)

        for i in range(n_samples):
            lengths = [self._path_length(X_np[i], tree) for tree in self.trees]
            path_lengths[i] = np.mean(lengths)

        # Scale scores using BST normalization factor
        c_n = c_factor(min(self.max_samples, n_samples))
        if c_n > 0.0:
            scores = 2.0 ** (-path_lengths / c_n)
        else:
            scores = np.zeros(n_samples)

        # Match scikit-learn interface: raw scores are (0.5 - anomaly_score)
        # s -> 1 (most anomalous) => score -> -0.5
        # s -> 0 (most normal) => score -> +0.5
        return 0.5 - scores
