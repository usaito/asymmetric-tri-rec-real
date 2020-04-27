
"""Tools to Evaluate Recommendation models with Ranking Metrics."""

import numpy as np


def calc_ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Calculate a nDCG score for a given user."""
    y_max_sorted = y_true[y_true.argsort()[::-1]]
    y_true_sorted = y_true[y_score.argsort()[::-1]]

    num_items = y_true.shape[0]
    k = num_items if num_items < k else k

    dcg_score = y_true_sorted[0] - 1
    for i in np.arange(1, k):
        dcg_score += y_true_sorted[i] / np.log2(i + 1)

    max_score = 2 ** (y_max_sorted[0]) - 1
    for i in np.arange(1, k):
        max_score += y_max_sorted[i] / np.log2(i + 1)

    return dcg_score / max_score


class PredictRankings:
    """Predict rankings by trained recommendations."""

    def __init__(self, user_embed: np.ndarray, item_embed: np.ndarray, item_bias: np.ndarray) -> None:
        """Initialize Class."""
        self.user_embed = user_embed
        self.item_embed = item_embed
        self.item_bias = item_bias

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        scores = (user_emb @ item_emb.T).flatten() + self.item_bias[items]
        return scores


def aoa_evaluator(user_embed: np.ndarray, item_embed: np.ndarray, item_bias: np.ndarray,
                  test: np.ndarray, at_k: int = 3) -> float:
    """Calculate ranking metrics with average-over-all evaluator."""
    users = test[:, 0]
    items = test[:, 1]
    ratings = test[:, 2]

    # define model
    model = PredictRankings(user_embed=user_embed, item_embed=item_embed, item_bias=item_bias)

    # calculate ranking metrics
    results = []
    np.random.seed(12345)
    for user in set(users):
        indices = users == user
        items_for_user = items[indices]
        ratings_for_user = ratings[indices]

        scores = model.predict(users=user, items=items_for_user)
        results.append(calc_ndcg_at_k(ratings_for_user, scores, at_k))

    return np.mean(results)
