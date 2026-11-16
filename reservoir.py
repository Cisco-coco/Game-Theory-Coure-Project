"""Algorithm R reservoir sampling, for later Deep CFR sample memories.

Reference: Vitter (1985), Random Sampling with a Reservoir,
ACM TOMS 11(1), doi:10.1145/3147.3165. Project-written implementation.
Not needed by the tabular MCCFR solver itself.
"""
import copy
import random


class ReservoirBuffer:
    def __init__(self, capacity, seed=42):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.seen = 0
        self.items = []
        self.rng = random.Random(seed)

    def add(self, sample):
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(copy.deepcopy(sample))
        else:
            slot = self.rng.randrange(self.seen)
            if slot < self.capacity:
                self.items[slot] = copy.deepcopy(sample)

    def sample(self, size):
        if size < 0 or size > len(self.items):
            raise ValueError("sample size must be within current buffer length")
        return copy.deepcopy(self.rng.sample(self.items, size))

    def __len__(self):
        return len(self.items)

    def state_dict(self):
        return copy.deepcopy({"capacity": self.capacity, "seen": self.seen,
                              "items": self.items, "rng": self.rng.getstate()})

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or len(state["items"]) > self.capacity or state["seen"] < len(state["items"]):
            raise ValueError("Incompatible reservoir state")
        self.items = copy.deepcopy(state["items"])
        self.seen = state["seen"]
        self.rng.setstate(state["rng"])
