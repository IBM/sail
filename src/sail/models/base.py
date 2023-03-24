import abc


class SAILWrapper(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError("Save method is not implemented.")

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError("Load method is not implemented.")
