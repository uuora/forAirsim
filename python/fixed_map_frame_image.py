"""Bounded immutable RGB cache keyed by exact bytes, with no freshness claims."""
import hashlib
from functools import lru_cache
import cv2
import numpy as np

@lru_cache(maxsize=3)
def _decode(encoded):
    image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.ndim!=3 or image.shape[2]!=3:raise ValueError('BOUND_RGB_DECODE_FAILED')
    image.setflags(write=False)
    return image

def decode_bound_rgb(encoded,expected_sha256):
    if not isinstance(encoded,bytes) or hashlib.sha256(encoded).hexdigest()!=expected_sha256:raise ValueError('BOUND_RGB_HASH_MISMATCH')
    return _decode(encoded)
