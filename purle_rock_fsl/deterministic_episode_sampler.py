# coding=utf-8
from __future__ import annotations

import hashlib

import numpy as np
import torch


class DeterministicEpisodeSampler(object):
    """Episode sampler with RNG state isolated from model initialization.

    Every ablation variant receives the same seed per split, so it samples the
    same classes, support images, query images, and within-episode ordering.
    """

    def __init__(
        self,
        labels,
        classes_per_it,
        num_samples,
        iterations,
        seed,
        record_episodes=False,
    ):
        self.labels = labels
        self.classes_per_it = int(classes_per_it)
        self.sample_per_class = int(num_samples)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.record_episodes = bool(record_episodes)

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.as_tensor(self.classes, dtype=torch.long)
        if len(self.classes) < self.classes_per_it:
            raise ValueError(
                "Not enough classes for an episode: have {}, need {}".format(
                    len(self.classes), self.classes_per_it
                )
            )
        if np.any(self.counts < self.sample_per_class):
            raise ValueError(
                "Some classes have fewer samples than support+query: "
                "min={}, need={}".format(
                    int(self.counts.min()), self.sample_per_class
                )
            )

        max_count = int(max(self.counts))
        self.indexes = torch.full(
            (len(self.classes), max_count), -1, dtype=torch.long
        )
        self.numel_per_class = torch.zeros(len(self.classes), dtype=torch.long)
        class_to_row = {
            int(class_id): row for row, class_id in enumerate(self.classes.tolist())
        }
        for sample_index, label in enumerate(self.labels):
            row = class_to_row[int(label)]
            column = int(self.numel_per_class[row])
            self.indexes[row, column] = sample_index
            self.numel_per_class[row] += 1

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self._digest = hashlib.sha256()
        self.episode_records = []
        self.generated_episodes = 0

    def __iter__(self):
        samples_per_class = self.sample_per_class
        classes_per_episode = self.classes_per_it

        for _ in range(self.iterations):
            batch = torch.empty(
                samples_per_class * classes_per_episode, dtype=torch.long
            )
            class_rows = torch.randperm(
                len(self.classes), generator=self.generator
            )[:classes_per_episode]
            for position, row in enumerate(class_rows.tolist()):
                start = position * samples_per_class
                end = start + samples_per_class
                sample_columns = torch.randperm(
                    int(self.numel_per_class[row]), generator=self.generator
                )[:samples_per_class]
                batch[start:end] = self.indexes[row, sample_columns]
            order = torch.randperm(len(batch), generator=self.generator)
            batch = batch[order]

            record = [int(index) for index in batch.tolist()]
            self._digest.update(np.asarray(record, dtype=np.int64).tobytes())
            self.generated_episodes += 1
            if self.record_episodes:
                self.episode_records.append(record)
            yield batch

    def __len__(self):
        return self.iterations

    @property
    def fingerprint(self):
        return self._digest.hexdigest()

