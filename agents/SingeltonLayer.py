from typing import Any
import threading
from abc import ABCMeta

class SingeltonLayer(ABCMeta):
    _instances : dict[type,Any] = {}
    _lock : threading.Lock = threading.Lock()

    @classmethod
    def __call__(cls, *args, **kwargs):
        with threading.Lock :
            if cls not in cls._instances :
                instance = super.__call__(*args,**kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]