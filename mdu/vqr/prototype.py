from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader


class BaseMultidimensionalOrdering(ABC):
    @abstractmethod
    def fit(self, train_loader: DataLoader, train_params: dict):
        pass

    @abstractmethod
    def predict(self, scores_test: torch.Tensor):
        pass

    @abstractmethod
    def predict_ranks(self, scores_test: torch.Tensor):
        pass
