"""This module exposes logging methods.

Copyright Marko Vejnovic <contact@markovejnovic.com> 2021
"""

TERM_COLORS = {
    'INFO': '\033[94m',
    'WARNING': '\033[93m',
    'ERROR': '\033[91m',
    'END': '\033[0m',
    'UNDERLINE': '\033[4m'
}

DEBUG_ENABLED = False


def out(tag: str, msg: str):
    """Outputs a message to the log.

    Parameters:
        tag - The Tag with which to log.
        msg - The message to output.
    """
    print(f"[{tag}]: {msg}")


def info(msg: str):
    """Outputs an informational message to the log.

    Parameters:
        msg - The message to write.
    """
    out(f"{TERM_COLORS['INFO']}INFO{TERM_COLORS['END']}", msg)


def err(msg: str):
    """Outputs an error message to the log.

    Parameters:
        msg - The message to write.
    """
    out(f"{TERM_COLORS['ERROR']}ERROR{TERM_COLORS['END']}", msg)


def dbg(msg: str, tag=''):
    """Outputs a debug message to the log.

    Parameters:
        msg - The message to write.
        tag - An optional tag with which to identify the message.
    """
    if DEBUG_ENABLED:
        out(f"{TERM_COLORS['UNDERLINE']}DEBUG{TERM_COLORS['END']}:{tag}", msg)
