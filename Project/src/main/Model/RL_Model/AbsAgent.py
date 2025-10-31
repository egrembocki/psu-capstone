import abc as absract


class AbsAgent(absract.ABC):
    @absract.abstractmethod
    def select_action(self, state: object) -> object:
        pass

    @absract.abstractmethod
    def update_policy(self, *args, **kwargs) -> None:
        pass

    @absract.abstractmethod
    def save_agent(self, filepath: str) -> None:
        pass

    @absract.abstractmethod
    def load_agent(self, filepath: str) -> None:
        pass