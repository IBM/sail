import logging


def configure_logger(logger_name="Sail", logging_level=logging.DEBUG):
    """Performs logging configuration

    Args:
        logger_name ([str], optional): [description]. Defaults logging name
        logging_level ([str], optional): [description]. Defaults logging level
    """

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s : %(message)s",
        handlers=[logging.StreamHandler()],
    )

    return logging.getLogger(logger_name)
