from __future__ import division, absolute_import
import copy
import numpy as np
import random
from collections import defaultdict,Counter
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler

AVAI_SAMPLERS = ['RandomIdentitySampler', 'SequentialSampler', 'RandomSampler','AugmentedRatioBatchSampler']

class AugmentedRatioBatchSampler(Sampler):
    """
    base_batch_sampler(list[int|tuple]) 가 만든 '원본' 배치에
    같은 PID의 증강 이미지를 aug_ratio 만큼 추가.
    튜플(idx, set) → idx 로 자동 변환.
    """
    def __init__(self, data_source, base_batch_sampler, aug_ratio=0.1):
        self.data_source = data_source
        self.base_batch_sampler = base_batch_sampler
        self.aug_ratio = aug_ratio

        # PID → 증강 인덱스 풀
        self.aug_pool = defaultdict(list)
        for idx, s in enumerate(data_source):
            if s.get('is_augmented', False):
                self.aug_pool[s['pid']].append(idx)

    def __len__(self):
        return len(self.base_batch_sampler)

    def _idx(self, elem):
        """tuple(idx, set) -> idx  |  int -> int"""
        return elem[0] if isinstance(elem, tuple) else elem

    def __iter__(self):
        for orig_batch in self.base_batch_sampler:        # list[int|tuple]
            int_batch = [self._idx(x) for x in orig_batch]  # ← 튜플 제거

            pid_cnt = Counter(self.data_source[i]['pid'] for i in int_batch)
            need_aug = int(round(len(int_batch) * self.aug_ratio /
                                 (1 - self.aug_ratio)))
            aug_batch = []
            pid_cycle = random.sample(list(pid_cnt.keys()), len(pid_cnt))
            while len(aug_batch) < need_aug and pid_cycle:
                pid = pid_cycle.pop(0)
                pool = self.aug_pool.get(pid, [])
                if pool:
                    aug_batch.append(random.choice(pool))

            yield int_batch + aug_batch  
class _BatchPidWrapper(Sampler):
    """batch(list[int]) → [(idx, batch_pids)] 형태로 변환"""
    def __init__(self, base_sampler, dataset):
        self.base_sampler = base_sampler
        self.dataset = dataset
    def __len__(self): return len(self.base_sampler)
    def __iter__(self):
        for batch in self.base_sampler:
            # pids = [self.dataset.train[i]['pid'] for i in batch]
            # yield [(idx, pids) for idx in batch]
            int_batch = [i[0] if isinstance(i, tuple) else i for i in batch]
            pids = [self.dataset.train[i]['pid'] for i in int_batch]
            yield [(idx[0] if isinstance(idx, tuple) else idx, pids) for idx in batch]

def wrap_with_batch_pids(sampler, dataset, enable=False):
    return _BatchPidWrapper(sampler, dataset) if enable else sampler

class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, sample in enumerate(self.data_source):
            self.index_dic[sample['pid']].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                # for each sample, add information about other pids that are in the same training batch
                # this will be used in dataset.__getitem__() to perform batch aware transformations, such as copy
                # pasting other pedestrians from the same batch into the current image
                for idx in batch_idxs:
                    pids_without_current = set(selected_pids)
                    pids_without_current.remove(pid)
                    final_idxs.append((idx, pids_without_current))
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_sampler(
    data_source, train_sampler, batch_size=32, num_instances=4, **kwargs
):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (when using ``RandomIdentitySampler``). Default is 4.
    """
    assert train_sampler in AVAI_SAMPLERS, \
        'train_sampler must be one of {}, but got {}'.format(AVAI_SAMPLERS, train_sampler)

    if train_sampler == 'AugmentedRatioBatchSampler':
        # sampler = AugmentedRatioBatchSampler(data_source, batch_size, kwargs.get('aug_ratio',0.1))
        base_sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
        aug_ratio = kwargs.get('aug_ratio', 0.1)
        # If aug_ratio is 0, just return the base sampler
        if aug_ratio <= 0:
            return base_sampler
        # Otherwise create the augmented sampler
        sampler = AugmentedRatioBatchSampler(data_source, base_sampler, aug_ratio)

    elif train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)

    elif train_sampler == 'SequentialSampler':
        sampler = SequentialSampler(data_source)

    elif train_sampler == 'RandomSampler':
        sampler = RandomSampler(data_source)

    return sampler
