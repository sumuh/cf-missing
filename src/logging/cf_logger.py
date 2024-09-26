import logging
from logging.handlers import RotatingFileHandler


class CfLogger:

    def __init__(self, results_dir_path: str):
        log_file_path = f"{results_dir_path}/log.log"
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)
        handler = RotatingFileHandler(log_file_path, maxBytes=100000, backupCount=1)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_debug(self, content: str):
        """Saves content to log file.

        :param str content: content to save
        """
        self.logger.debug(content)
