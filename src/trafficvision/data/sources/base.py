from abc import ABC, abstractmethod


class DatasetSource(ABC):
    @abstractmethod
    def __iter__(self):
        """Yield (PIL RGB image, canonical annotation mapping)."""

    def set_epoch(self, epoch):
        self.epoch = epoch
