from torch.utils.data import Sampler
from collections import defaultdict
import math
import random

class PartitionBatchSampler(Sampler[list[int]]):
    """
    Yields one full partition per batch (variable batch size).
    Use when partitions are already sized as desired.
    """
    def __init__(self, partition_ids, seed: int = 0, shuffle_partitions: bool = True, shuffle_within: bool = True):
        self.partition_ids = list(partition_ids)
        self.seed = int(seed)
        self.shuffle_partitions = shuffle_partitions
        self.shuffle_within = shuffle_within
        self._epoch = 0

        groups = defaultdict(list)
        for idx, pid in enumerate(self.partition_ids):
            groups[pid].append(idx)
        self._groups = dict(groups)
        self._pkeys = list(self._groups.keys())

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        pkeys = self._pkeys[:]
        if self.shuffle_partitions:
            rng.shuffle(pkeys)

        for pid in pkeys:
            idxs = self._groups[pid][:]
            if self.shuffle_within:
                rng.shuffle(idxs)
            yield idxs  # whole partition becomes one batch

    def __len__(self):
        return len(self._groups)