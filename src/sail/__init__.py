__version__ = "0.6.4"
__package_name__ = "SAIL"
_logging_level = "INFO"


def set_logging_level(logging_level):
    global _logging_level
    _logging_level = logging_level


def get_logging_level():
    return _logging_level
