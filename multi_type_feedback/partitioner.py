from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import wandb

from multi_type_feedback.stratification import split_partition_avoiding_ties


def _compute_partition_statistics(partition_ids: List[int], examples_modified: int = 0) -> Dict:
    """Compute statistics about partition sizes."""
    final_partition_ids = partition_ids
    stratum_counts = Counter(final_partition_ids)
    partition_sizes = list(stratum_counts.values())
    non_singleton_sizes = [size for size in partition_sizes if size > 1]

    singleton_examples = sum(1 for size in partition_sizes if size == 1)
    non_singleton_examples = len(partition_ids) - singleton_examples

    size_counts = Counter(partition_sizes)
    table_data = [[size, count] for size, count in sorted(size_counts.items())]
    size_table = wandb.Table(data=table_data, columns=["partition_size", "count"])

    return {
        "partition_count": len(stratum_counts),
        "partition_size_min": min(stratum_counts.values()) if stratum_counts else 0,
        "partition_size_max": max(stratum_counts.values()) if stratum_counts else 0,
        "partition_size_mean": np.mean(partition_sizes),
        "partition_sizes": size_table,
        "nonsingleton_partition_size_mean": np.mean(non_singleton_sizes)
        if non_singleton_sizes
        else 0.0,
        "examples_modified": examples_modified,
        "singleton_partitions": sum(1 for size in partition_sizes if size == 1),
        "non_singleton_partitions": sum(1 for size in partition_sizes if size > 1),
        "singleton_examples": singleton_examples,
        "non_singleton_examples": non_singleton_examples,
        "non_singleton_fraction": non_singleton_examples / len(partition_ids),
    }


class BasePartitioner(ABC):
    """Base class for partitioning data after filtering and before sampling."""

    @abstractmethod
    def partition_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Partition examples by modifying their partition_id values.

        Args:
            examples: List of example dictionaries with partition_id field
            rng: Random number generator

        Returns:
            Tuple of (modified_examples, partition_info_dict) where:
            - modified_examples: Examples with potentially modified partition_id values
            - partition_info_dict: Dictionary with partitioning statistics
        """
        pass


class TieBreakingPartitioner(BasePartitioner):
    """Base class for partitioners that split partitions to avoid tied ranks."""

    @abstractmethod
    def _get_strategy(self) -> str:
        """Return the tie-breaking strategy string."""
        pass

    def partition_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Split partitions to avoid tied ranks within each partition."""
        partition_groups = defaultdict(list)
        for idx, ex in enumerate(examples):
            partition_groups[ex["partition_id"]].append(idx)

        examples_modified = 0
        next_partition_id = max(ex["partition_id"] for ex in examples) + 1

        partition_ids = [0] * len(examples)
        for partition_id, indices in partition_groups.items():
            if len(indices) <= 1:
                continue

            ranks = [examples[idx]["rank"] for idx in indices]

            subpartitions = split_partition_avoiding_ties(
                ranks=ranks,
                strategy=self._get_strategy(),
                rel_tol=1e-9,
                rng=rng,
            )
            
            if len(subpartitions) > 1:
                for subpartition_indices in subpartitions:
                    for local_idx in subpartition_indices[1:]:
                        global_idx = indices[local_idx]
                        examples[global_idx]["partition_id"] = next_partition_id
                        partition_ids[global_idx] = next_partition_id
                        examples_modified += 1
                    next_partition_id += 1

        return partition_ids, _compute_partition_statistics(partition_ids, examples_modified)


class MaxSizeTieBreakingPartitioner(TieBreakingPartitioner):
    """Partitioner that splits partitions to avoid ties while maximizing partition sizes."""

    def _get_strategy(self) -> str:
        return "max"


class RoundRobinPartitioner(TieBreakingPartitioner):
    """Partitioner that splits partitions to avoid ties using round-robin assignment."""

    def __init__(self, target_size):
        self.target_size = target_size

    def _get_strategy(self) -> str:
        if self.target_size == "auto":
            return "targeted_size_spread:auto"
        return f"targeted_size_spread:{self.target_size}"


class RandomTieAvoidingPartitioner(TieBreakingPartitioner):
    """Partitioner that splits partitions to avoid ties using random assignment."""

    def __init__(self, target_size):
        self.target_size = target_size

    def _get_strategy(self) -> str:
        return f"targeted_size_random:{self.target_size}"


class RandomPartitioner(BasePartitioner):
    """Partitioner that creates randomly shuffled partitions of fixed size."""

    def __init__(self, partition_size):
        self.partition_size = partition_size

    def partition_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Create randomly shuffled partitions of the specified size."""
        if rng is None:
            raise ValueError("RandomPartitioner requires rng parameter")

        filtered_indices = []
        valid_indices = []
        for i, ex in enumerate(examples):
            if ex.get("rank") is None:
                filtered_indices.append(i)
            else:
                valid_indices.append(i)

        rng.shuffle(valid_indices)

        current_partition_id = 0

        partition_ids = [0] * len(examples)
        for i in range(0, len(valid_indices), self.partition_size):
            batch_indices = valid_indices[i : i + self.partition_size]
            for idx in batch_indices:
                examples[idx]["partition_id"] = current_partition_id
                partition_ids[idx] = current_partition_id
            current_partition_id += 1

        for idx in filtered_indices:
            examples[idx]["partition_id"] = current_partition_id
            partition_ids[idx] = current_partition_id
            current_partition_id += 1

        return partition_ids, _compute_partition_statistics(
            partition_ids, examples_modified=len(examples)
        )


class NoOpPartitioner(BasePartitioner):
    """Pass-through partitioner that doesn't modify partition assignments."""

    def partition_examples(self, examples: List[Dict], rng) -> Tuple[List[Dict], Dict]:
        """Return examples unchanged."""
        return examples, _compute_partition_statistics(examples, examples_modified=0)
