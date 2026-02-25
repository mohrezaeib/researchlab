from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


@dataclass(frozen=True)
class NgramFeaturizer:
    ngram_min: int = 3
    ngram_max: int = 6
    max_features: int = 200_000

    def make_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(
            analyzer="char",
            ngram_range=(self.ngram_min, self.ngram_max),
            lowercase=False,
            max_features=self.max_features,
        )


def one_hot(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = mapping.get(ch)
        if j is not None:
            arr[j, i] = 1.0
    return arr

