import logging
import os


__all__ = ['get_default_logger', 'logger']


def get_default_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
    return logger


logger = get_default_logger('AND')
