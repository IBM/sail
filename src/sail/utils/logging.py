import logging
import logzero
import sail


def configure_logger(
    logger_name: str,
    logging_level=sail.get_logging_level(),
    json=False,
):
    if logger_name is None or logger_name == "":
        raise Exception(
            "Logger name cannot be None or empty. Accessing root logger is restricted. Please use routine configure_root_logger() to access the root logger."
        )

    heading = "SAIL" + " " + "(" + logger_name + ")"

    info_log_format = (
        "%(color)s[%(asctime)s:%(msecs)d] - %(levelname)s - " + heading + " - "
        "%(end_color)s%(message)s"
    )
    rest_log_format = (
        "%(color)s[%(asctime)s:%(msecs)d] - %(levelname)s - " + heading + " - "
        "%(module)s:%(funcName)s:%(lineno)d%(end_color)s %(message)s"
    )

    logging_level = logging.getLevelName(logging_level)
    if logging.INFO == logging_level:
        log_format = info_log_format
    else:
        log_format = rest_log_format

    formatter = logzero.LogFormatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    return logzero.setup_logger(
        name=logger_name, level=logging_level, formatter=formatter, json=json
    )


def configure_root_logger(
    logging_level=sail.get_logging_level(),
    json=False,
):
    heading = "SAIL"

    info_log_format = (
        "%(color)s[%(asctime)s:%(msecs)d] - %(levelname)s - "
        + heading
        + "%(end_color)s%(message)s"
    )
    rest_log_format = (
        "%(color)s[%(asctime)s:%(msecs)d] - %(levelname)s - "
        + heading
        + "%(module)s:%(funcName)s:%(lineno)d%(end_color)s %(message)s"
    )

    logging_level = logging.getLevelName(logging_level)
    if logging.INFO == logging_level:
        log_format = info_log_format
    else:
        log_format = rest_log_format

    formatter = logzero.LogFormatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    return logzero.setup_logger(
        name=None, level=logging_level, formatter=formatter, json=json
    )


class SAILVerbosity:
    def __init__(self, verbosity=1, log_interval=None) -> None:
        self.verbosity = verbosity
        self.log_interval = log_interval
        self.reset()

    def get(self) -> int:
        if self.if_logging_allowed():
            return self._verbosity
        else:
            return 0

    def reset(self) -> int:
        self._verbosity = self.verbosity
        self._n_epochs = -1

    def enabled(self):
        self._verbosity = 1

    def disabled(self):
        self._verbosity = 0

    def resolve(self, override):
        if self.if_logging_allowed():
            if override is None:
                return self.get()
            elif isinstance(override, int):
                if override == 0 or override == 1:
                    return override
            else:
                raise Exception(
                    "Cannot override verbosity value. Override verbosity should only have values from [0, 1]."
                )
        else:
            return 0

    def if_logging_allowed(self):
        return (
            True
            if self.log_interval is None
            else self._n_epochs % self.log_interval == 0
        )

    def log_epoch(self):
        self._n_epochs += 1
