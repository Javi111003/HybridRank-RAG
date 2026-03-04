from abc import ABC, abstractmethod


class FusionStrategy(ABC):

    @abstractmethod
    def fuse(self, results_list, top_k):
        pass