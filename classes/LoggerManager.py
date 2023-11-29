import logging
import sys
from typing import Optional, Dict
from colorama import Fore, Back, Style


# Source code: https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad
# Doc: https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)


class LoggerManager:

    def __init__(self, log_level):
        assert log_level in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET', 50, 40, 30, 20, 10, 0], \
            'Incorrect Level. See https://docs.python.org/3/library/logging.html#levels'

        self.log_level = log_level

        logging.DEBUG

    def get_logger_configured(self):
        formatter = ColoredFormatter(
            # '{asctime} |{color} {levelname:8} {reset}| {message}',
            '{color}{asctime} | {message}{reset}',
            style='{', datefmt='%Y-%m-%d %H:%M:%S',
            colors={
                'DEBUG': Fore.BLUE,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
            }
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.handlers[:] = []
        logger.addHandler(handler)
        logger.setLevel(self.log_level)

        return logger
