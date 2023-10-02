from typing import Literal
from sail.utils.logging import configure_logger

LOGGER = configure_logger(logger_name="VerboseManager", bare_format=True)


class VerboseManager:
    def __init__(
        self,
        name: str = "VerboseManager",
        verbosity: Literal[0, 1] | None = None,
        verbosity_interval: int | None = None,
    ) -> None:
        self.name = name

        assert (
            verbosity == 0 or verbosity == 1 or verbosity is None
        ), "Invalid Verbosity value. Verbosity can only take values from [None, 0, 1]."
        self.verbosity = 0 if verbosity is None else verbosity

        assert (
            isinstance(verbosity_interval, int) or verbosity_interval is None
        ), "Invalid verbosity_interval value. Verbosity interval can only be a None or an int value."
        self.verbosity_interval = verbosity_interval

        self._current_epoch_n = 0
        self._samples_seen_n = 0

    def get_state(self):
        return {
            "current_epoch_n": self._current_epoch_n,
            "samples_seen_n": self._samples_seen_n,
        }

    def set_state(self, state: dict):
        self._current_epoch_n = state["current_epoch_n"]
        self._samples_seen_n = state["samples_seen_n"]

    @property
    def current_epoch_n(self):
        return self._current_epoch_n

    @property
    def samples_seen_n(self):
        return self._samples_seen_n

    def get(self, override=None) -> int:
        if self.if_logging_allowed():
            if override is not None:
                if isinstance(override, int):
                    if override == 0 or override == 1:
                        return override
                else:
                    raise Exception(
                        "Cannot override verbosity value. Override verbosity should only have values from [0, 1]."
                    )
            return self.verbosity
        return 0

    def reset(self) -> int:
        self.verbosity = self.verbosity

    def enabled(self):
        self.verbosity = 1

    def disabled(self):
        self.verbosity = 0

    def if_logging_allowed(self):
        return (
            True
            if self.verbosity_interval is None
            else (
                self._current_epoch_n == 1
                or (self._current_epoch_n % self.verbosity_interval == 0)
            )
        )

    def log_epoch(self, X):
        self._current_epoch_n += 1

        if self.get() == 1:
            self.print_epoch_head()

        self._samples_seen_n += X.shape[0]

    def print_epoch_head(self):
        LOGGER.info("\n")
        LOGGER.info(
            f">> Epoch: {self._current_epoch_n} | Samples Seen: {self._samples_seen_n} -------------------------------------------------------------------------------------"
        )
