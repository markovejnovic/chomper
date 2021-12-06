from typing import Any
from tensorflow import keras
from os.path import join, exists, isdir
from hashlib import md5
from os import listdir, makedirs
import sys
import pickle
import numpy as np

from config import DATA_PATH
from log import dbg, err
from exit_codes import UNEXPECTED_IO_ERROR


def get_cache_dir():
    """Returns the directory to the cache."""
    # TODO: Don't really need full qualified name here.
    path = DATA_PATH

    if not exists(path):
        makedirs(path)
    elif not isdir(path):
        err('The caching directory is not a directory. Nothing is touched. '
            f'Please check {path} is a directory. The program will exit')
        sys.exit(UNEXPECTED_IO_ERROR)

    return path


def cacheable(ser_technique, deser_technique):
    """A decorator used to signify a function will cache its outputs in a
    directory. The return values must be serializable. The function must also
    be mutable. This is not thread-safe due to IOps."""
    def cachable_inner(function):
        def wrapper(*args):
            # TODO: Check if this is the best way to do this.
            hashfact = md5(pickle.dumps(args))
            argshash = str(function.__name__) + hashfact.hexdigest()

            cachedir = get_cache_dir()
            filename = argshash + '.cached'
            path = join(cachedir, filename)

            if filename in listdir(cachedir):
                value = deser_technique(path)
                return value

            value = function(*args)
            ser_technique(value, path)
            return value
        return wrapper
    return cachable_inner


def generic_ser(object: Any, path: str) -> None:
    """Saves a numpy array into a path."""
    dbg(f'Saving to: {path}', 'generic_ser')
    pickle.dump(object, open(path, 'wb'))


def generic_deser(path: str) -> Any:
    dbg(f'Loading from: {path}', 'generic_deser')
    return pickle.load(open(path, 'rb'))


cached = cacheable(generic_ser, generic_deser)
