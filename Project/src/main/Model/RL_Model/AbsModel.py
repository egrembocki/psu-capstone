import abc as absract


class AbsModel(absract.ABC):
    @absract.abstractmethod
    def train(self, *args, **kwargs) -> object:
        pass

    @absract.abstractmethod
    def predict(self, *args, **kwargs) -> object:
        pass

    @absract.abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @absract.abstractmethod
    def load(self, filepath: str) -> None:
        pass
