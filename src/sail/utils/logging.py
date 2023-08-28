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
