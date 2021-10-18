import numpy as np

import torch


class ReplayBuffer(object):
    def __init__(self, obs_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def fetch(self, idxs, discount, n):
        assert idxs.max() + n <= len(self)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs + n - 1]

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = np.zeros((idxs.shape[0], 1), dtype=np.float32)
        not_dones = np.ones((idxs.shape[0], 1), dtype=np.float32)
        for i in range(n):
            rewards += (discount**i) * not_dones * np.sign(
                self.rewards[idxs + i])
            not_dones = np.minimum(not_dones, self.not_dones[idxs + i])

        rewards = torch.as_tensor(rewards, device=self.device)
        not_dones = torch.as_tensor(not_dones, device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def sample_idxs(self, batch_size, n):
        last_idx = (self.capacity if self.full else self.idx) - (n - 1)
        idxs = np.random.randint(0, last_idx, size=batch_size)

        return idxs

    def sample_multistep(self, batch_size, discount, n):
        assert n <= self.idx or self.full
        idxs = self.sample_idxs(batch_size, n)

        return self.fetch(idxs, discount, n)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_shape, capacity, alpha, device):
        super().__init__(obs_shape, capacity, device)

        assert alpha >= 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0

    def add(self, obs, action, reward, next_obs, done):
        super().add(obs, action, reward, next_obs, done)
        self.sum_tree[self.idx] = self.max_priority**self.alpha
        self.min_tree[self.idx] = self.max_priority**self.alpha

    def sample_idxs(self, batch_size, n):
        idxs = []
        p_total = self.sum_tree.sum(0, len(self) - n - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            while True:
                mass = np.random.rand() * every_range_len + i * every_range_len
                idx = self.sum_tree.find_prefixsum_idx(mass)
                if idx + n <= len(self):
                    idxs.append(idx)
                    break
        return np.array(idxs)

    def sample_multistep(self, batch_size, beta, discount, n):
        assert n <= self.idx or self.full
        assert beta > 0

        idxs = self.sample_idxs(batch_size, n)

        weights = []
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self))**(-beta)

        for idx in idxs:
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (p_sample * len(self))**(-beta)
            weights.append(weight / max_weight)
        weights = torch.as_tensor(np.array(weights),
                                  device=self.device).unsqueeze(dim=1)

        sample = self.fetch(idxs, discount, n)

        return tuple(list(sample) + [weights, idxs])

    def update_priorities(self, idxs, prios):
        assert idxs.shape[0] == prios.shape[0]

        for idx, prio in zip(idxs, prios):
            assert prio > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = prio**self.alpha
            self.min_tree[idx] = prio**self.alpha

            self.max_priority = max(self.max_priority, prio)




import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (
                capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1,
                                           node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1,
                                        node_end))

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx],
                                               self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity,
                                             operation=operator.add,
                                             neutral_element=0.0)

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity,
                                             operation=min,
                                             neutral_element=float('inf'))

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)
