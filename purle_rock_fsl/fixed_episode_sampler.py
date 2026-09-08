"""Immutable episode manifests and disjoint validation pools."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence

import numpy as np
import torch


def split_classwise_indices(
    labels: Sequence[int],
    *,
    seed: int,
    minimum_per_pool: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split every class into disjoint A/B image pools."""

    labels_array = np.asarray(labels)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    left: list[int] = []
    right: list[int] = []
    for class_id in np.unique(labels_array):
        indices = np.flatnonzero(labels_array == class_id)
        if len(indices) < 2 * minimum_per_pool:
            raise ValueError(
                f"class {class_id} has {len(indices)} samples; "
                f"need {2 * minimum_per_pool} for disjoint validation"
            )
        order = torch.randperm(len(indices), generator=generator).numpy()
        split = len(indices) // 2
        split = max(minimum_per_pool, min(split, len(indices) - minimum_per_pool))
        left.extend(int(indices[i]) for i in order[:split])
        right.extend(int(indices[i]) for i in order[split:])
    left_tuple = tuple(sorted(left))
    right_tuple = tuple(sorted(right))
    if set(left_tuple) & set(right_tuple):
        raise RuntimeError("validation pools unexpectedly overlap")
    return left_tuple, right_tuple


def split_disjoint_class_indices(
    labels: Sequence[int],
    *,
    seed: int,
    minimum_classes_per_pool: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Partition validation classes, and therefore images, between A and B."""

    labels_array = np.asarray(labels)
    classes = np.unique(labels_array)
    if len(classes) < 2 * minimum_classes_per_pool:
        raise ValueError(
            f"need at least {2 * minimum_classes_per_pool} validation classes, "
            f"got {len(classes)}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(len(classes), generator=generator).numpy()
    split = len(classes) // 2
    split = max(
        minimum_classes_per_pool,
        min(split, len(classes) - minimum_classes_per_pool),
    )
    classes_a = tuple(sorted(int(classes[index]) for index in order[:split]))
    classes_b = tuple(sorted(int(classes[index]) for index in order[split:]))
    set_a, set_b = set(classes_a), set(classes_b)
    indices_a = tuple(
        index for index, label in enumerate(labels_array) if int(label) in set_a
    )
    indices_b = tuple(
        index for index, label in enumerate(labels_array) if int(label) in set_b
    )
    return indices_a, indices_b, classes_a, classes_b


class FixedEpisodeBatchSampler:
    """Replay an immutable, fingerprinted episode list on every iteration."""

    def __init__(
        self,
        labels: Sequence[int],
        *,
        classes_per_it: int,
        num_samples: int,
        iterations: int,
        seed: int,
        allowed_indices: Iterable[int] | None = None,
    ) -> None:
        self.classes_per_it = int(classes_per_it)
        self.num_samples = int(num_samples)
        self.iterations = int(iterations)
        self.seed = int(seed)
        labels_array = np.asarray(labels)
        allowed = (
            tuple(range(len(labels_array)))
            if allowed_indices is None
            else tuple(int(value) for value in allowed_indices)
        )
        pools: dict[int, list[int]] = {}
        for index in allowed:
            pools.setdefault(int(labels_array[index]), []).append(index)
        eligible = sorted(
            class_id
            for class_id, indices in pools.items()
            if len(indices) >= self.num_samples
        )
        if len(eligible) < self.classes_per_it:
            raise ValueError("not enough eligible classes for fixed episodes")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        episodes = []
        for _ in range(self.iterations):
            selected_rows = torch.randperm(
                len(eligible), generator=generator
            )[: self.classes_per_it]
            episode = []
            for row in selected_rows.tolist():
                class_pool = pools[eligible[row]]
                selected = torch.randperm(
                    len(class_pool), generator=generator
                )[: self.num_samples]
                episode.extend(class_pool[column] for column in selected.tolist())
            order = torch.randperm(len(episode), generator=generator).tolist()
            episodes.append(tuple(episode[position] for position in order))
        self.episodes = tuple(episodes)
        digest = hashlib.sha256()
        for episode in self.episodes:
            digest.update(np.asarray(episode, dtype=np.int64).tobytes())
        self._fingerprint = digest.hexdigest()

    def __iter__(self) -> Iterator[torch.Tensor]:
        for episode in self.episodes:
            yield torch.tensor(episode, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def sample_indices(self) -> frozenset[int]:
        return frozenset(index for episode in self.episodes for index in episode)
