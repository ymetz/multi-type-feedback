"""Stratification module for RT-rank loss implementation."""

import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


def _group_by_annotator(examples: List[Dict]) -> Dict[str, List[int]]:
    """Group examples by annotator/evaluator identity."""
    evaluator_to_indices = defaultdict(list)
    for i, example in enumerate(examples):
        # Use a default evaluator if not specified
        evaluator = example.get("evaluator", "default")
        evaluator_to_indices[evaluator].append(i)
    return evaluator_to_indices


def _calculate_example_length(example: Dict) -> float:
    """Calculate length metric for stratification (trajectory length for RL)."""
    # For RL environments, we can use the number of steps in the trajectory
    # This assumes the example has trajectory data in a specific format
    if "trajectory_length" in example:
        return example["trajectory_length"]
    
    # Fallback: estimate from observation/action data if available
    if "observations" in example and example["observations"] is not None:
        return len(example["observations"])
    
    # Default length for cases where we can't determine trajectory length
    return 50.0


def split_partition_avoiding_ties(ranks: List[float]) -> List[List[int]]:
    """Split a partition to avoid ties in rankings."""
    if not ranks:
        return []
    
    # Group indices by rank value
    rank_to_indices = defaultdict(list)
    for i, rank in enumerate(ranks):
        rank_to_indices[rank].append(i)
    
    # Each unique rank gets its own partition
    partitions = []
    for indices in rank_to_indices.values():
        partitions.append(indices)
    
    return partitions


def _compute_base_partitions(
    examples: List[Dict], partition_clusters: int, min_cluster_size: int
) -> List[int]:
    """
    Base partition computation logic.
    Groups by evaluator and clusters by trajectory length.
    """
    evaluator_to_indices = _group_by_annotator(examples)

    lengths = np.array([_calculate_example_length(ex) for ex in examples])

    partition_ids = [None] * len(examples)
    current_pid = 0
    small_bucket: List[int] = []
    
    for evaluator, inds in evaluator_to_indices.items():
        if len(inds) < min_cluster_size:
            small_bucket.extend(inds)
            continue

        eval_lens = lengths[inds].reshape(-1, 1)
        if np.std(eval_lens) < 1e-6:
            for i in inds:
                partition_ids[i] = current_pid
            current_pid += 1
            continue

        max_clusters = len(inds) // min_cluster_size
        n_clusters = min(partition_clusters, max_clusters, len(np.unique(eval_lens)))
        n_clusters = max(1, n_clusters)

        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
            labels = kmeans.fit_predict(eval_lens)

            counts = np.bincount(labels)
            centers = kmeans.cluster_centers_.flatten()
            order = np.argsort(centers)

            for c in order:
                if counts[c] < min_cluster_size:
                    candidates = {
                        o: abs(centers[c] - centers[o])
                        for o in order
                        if o != c and counts[o] + counts[c] >= min_cluster_size
                    }
                    target = (
                        min(candidates, key=candidates.get)
                        if candidates
                        else int(np.argmax(counts))
                    )
                    labels[labels == c] = target
                    counts[target] += counts[c]
                    counts[c] = 0

            uniq = np.unique(labels)
            remap = {old: new for new, old in enumerate(uniq)}
            labels = np.array([remap[label] for label in labels])
        else:
            labels = np.zeros(len(inds), dtype=int)

        for idx, lab in zip(inds, labels):
            partition_ids[idx] = current_pid + lab
        current_pid += int(labels.max()) + 1

    if small_bucket:
        for idx in small_bucket:
            partition_ids[idx] = current_pid
        current_pid += 1

    for i, pid in enumerate(partition_ids):
        if pid is None:
            partition_ids[i] = current_pid
            current_pid += 1

    return partition_ids


def _split_large_partitions(
    partition_ids: List[int], max_partition_size: int
) -> List[int]:
    """Split partitions that exceed max_partition_size into smaller chunks."""
    pid_to_inds: Dict[int, List[int]] = defaultdict(list)
    for i, pid in enumerate(partition_ids):
        pid_to_inds[pid].append(i)

    next_pid = max(pid_to_inds.keys()) + 1
    for pid, inds in list(pid_to_inds.items()):
        if len(inds) <= max_partition_size:
            continue
        for start in range(0, len(inds), max_partition_size):
            chunk = inds[start : start + max_partition_size]
            new_pid = pid if start == 0 else next_pid
            for idx in chunk:
                partition_ids[idx] = new_pid
            if start != 0:
                next_pid += 1

    return partition_ids


def _log_partition_statistics(
    partition_ids: List[int],
    min_cluster_size: int,
    function_name: str,
    max_partition_size: Optional[int] = None,
):
    """Log partition statistics."""
    counts = Counter(partition_ids)
    sizes = list(counts.values())
    total = len(sizes)

    print(f"\n[{function_name}] Partition statistics:")
    print(f"  Total partitions: {total}")
    print(f"  Minimum partition size: {min(sizes)}")
    print(f"  Maximum partition size: {max(sizes)}")
    print(f"  Average partition size: {sum(sizes) / total:.2f}")

    smalls = sum(1 for sz in sizes if sz < min_cluster_size)
    print(f"  Partitions below min_cluster_size ({min_cluster_size}): {smalls}")

    if max_partition_size is not None:
        oversized = sum(1 for sz in sizes if sz > max_partition_size)
        print(
            f"  Partitions above max_partition_size ({max_partition_size}): {oversized}"
        )
    print()


class BaseStratifier(ABC):
    """Abstract base class for stratification methods."""

    @abstractmethod
    def compute_partitions(self, examples: List[Dict], rng) -> List[int]:
        """Compute partition IDs for examples.

        Args:
            examples: List of example dictionaries
            rng: Random number generator (not used by all subclasses)

        Returns:
            List of partition IDs
        """
        pass


class GlobalPartitionStratifier(BaseStratifier):
    """Stratifier that assigns all examples to a single global partition."""

    def __init__(self, split_on_ties: bool = False):
        self.split_on_ties = split_on_ties

    def compute_partitions(self, examples: List[Dict], rng) -> List[int]:
        if not self.split_on_ties:
            return [0] * len(examples)

        ranks = [ex["rank"] for ex in examples]
        sub_partitions = split_partition_avoiding_ties(ranks)

        partition_ids = [None] * len(examples)
        for pid, sub_partition in enumerate(sub_partitions):
            for idx in sub_partition:
                partition_ids[idx] = pid

        print("\n[GlobalPartitionStratifier] After tie-splitting:")
        print(f"  Total partitions: {len(sub_partitions)}")
        _log_partition_statistics(partition_ids, 1, "GlobalPartitionStratifier")

        return partition_ids


class KnnStratifier(BaseStratifier):
    """Stratifier that groups by evaluator and clusters by trajectory length using KMeans."""

    def __init__(
        self, 
        partition_clusters: int = 8, 
        min_cluster_size: int = 4, 
        max_partition_size: Optional[int] = None
    ):
        self.partition_clusters = partition_clusters
        self.min_cluster_size = min_cluster_size
        self.max_partition_size = max_partition_size

    def compute_partitions(self, examples: List[Dict], rng) -> List[int]:
        """
        Compute partitions based on evaluator identity and trajectory length,
        merging all evaluator-groups smaller than `min_cluster_size`
        into one "small" partition. Optionally splits partitions larger
        than `max_partition_size` into smaller chunks.
        """
        if not examples:
            raise ValueError("Examples list cannot be empty")

        if (
            self.max_partition_size is not None
            and self.max_partition_size < self.min_cluster_size
        ):
            raise ValueError("max_partition_size must be ≥ min_cluster_size")

        partition_ids = _compute_base_partitions(
            examples, self.partition_clusters, self.min_cluster_size
        )

        if self.max_partition_size is not None:
            partition_ids = _split_large_partitions(
                partition_ids, self.max_partition_size
            )

        _log_partition_statistics(
            partition_ids,
            self.min_cluster_size,
            "KnnStratifier",
            self.max_partition_size,
        )

        return partition_ids


class StdWindowStratifier(BaseStratifier):
    """Stratifier that groups examples within standard deviation windows of trajectory length."""

    def __init__(
        self, 
        std_window: float = 1.0, 
        min_cluster_size: int = 1, 
        max_size: Optional[int] = None, 
        split_on_ties: bool = False
    ):
        self.std_window = std_window
        self.min_cluster_size = min_cluster_size
        self.max_size = max_size
        self.split_on_ties = split_on_ties

    def compute_partitions(self, examples: List[Dict], rng) -> List[int]:
        """
        Compute partitions based on trajectory length standard deviation windows.

        Each partition contains examples within `std_window` standard deviations
        of trajectory length, grouped by evaluator. Single-element partitions are allowed.
        """
        if not examples:
            raise ValueError("Examples list cannot be empty")

        evaluator_to_indices = _group_by_annotator(examples)

        lengths = np.array([_calculate_example_length(ex) for ex in examples])
        std_len = np.std(lengths)
        window_size = self.std_window * std_len

        partition_ids = [None] * len(examples)
        current_pid = 0

        for evaluator, indices in evaluator_to_indices.items():
            if not indices:
                continue

            eval_lengths = lengths[indices]
            sorted_indices = sorted(indices, key=lambda i: lengths[i])

            i = 0
            while i < len(sorted_indices):
                partition_start_idx = sorted_indices[i]
                partition_start_len = lengths[partition_start_idx]
                partition_size = 0

                j = i
                while j < len(sorted_indices):
                    current_idx = sorted_indices[j]
                    current_len = lengths[current_idx]

                    if self.max_size is not None and partition_size >= self.max_size:
                        break

                    if current_len - partition_start_len <= window_size:
                        partition_ids[current_idx] = current_pid
                        partition_size += 1
                        j += 1
                    else:
                        break

                current_pid += 1
                i = j

        for i, pid in enumerate(partition_ids):
            if pid is None:
                partition_ids[i] = current_pid
                current_pid += 1

        if self.split_on_ties:
            # Group examples by partition
            partition_to_indices = defaultdict(list)
            for i, pid in enumerate(partition_ids):
                partition_to_indices[pid].append(i)

            new_partition_ids = [None] * len(examples)
            new_pid = 0

            for old_pid, indices in partition_to_indices.items():
                partition_ranks = [examples[i]["rank"] for i in indices]
                sub_partitions = split_partition_avoiding_ties(partition_ranks)

                for sub_partition in sub_partitions:
                    for local_idx in sub_partition:
                        global_idx = indices[local_idx]
                        new_partition_ids[global_idx] = new_pid
                    new_pid += 1

            partition_ids = new_partition_ids

            print("\n[StdWindowStratifier] After tie-splitting:")
            print(f"  Total partitions increased to: {new_pid}")

        _log_partition_statistics(
            partition_ids, self.min_cluster_size, "StdWindowStratifier"
        )

        counts = Counter(partition_ids)
        single_element_partitions = sum(1 for count in counts.values() if count == 1)
        print(f"  Single-element partitions: {single_element_partitions}")

        print("\n  Per-evaluator statistics:")
        for evaluator, indices in evaluator_to_indices.items():
            eval_lengths = lengths[indices]
            print(
                f"    {evaluator}: n={len(indices)}, mean_len={np.mean(eval_lengths):.1f}, std_len={np.std(eval_lengths):.1f}"
            )

        return partition_ids