from dataclasses import dataclass, field

BATCH_SIZE = 20


@dataclass
class Memory:
    states = field(default_factory=list)

    def is_full(self):
        return len(self.states) >= BATCH_SIZE * 5

    def remember(self):
        pass

    def forget(self):
        self.states = []
