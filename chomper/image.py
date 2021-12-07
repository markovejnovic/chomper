import numpy as np
from typing import Tuple
from PIL import Image

def bounding_box(img: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def autocrop(img: np.ndarray) -> np.ndarray:
    y0, y1, x0, x1 = bounding_box(img)
    return img[y0:y1, x0:x1]


def resize_to_fit(img: np.ndarray, dims: Tuple[int, int]) -> np.ndarray:
    pilim = Image.fromarray(img)
    scaling_factor = min(dims[0] / pilim.width, dims[1] / pilim.height)
    pilim = pilim.resize(
        (round(pilim.width * scaling_factor),
         round(pilim.height * scaling_factor)), Image.ANTIALIAS)
    out = np.asarray(pilim)

    ypad = dims[1] - out.shape[0]
    xpad = dims[0] - out.shape[1]

    result = np.pad(out, (
        (ypad // 2, ypad // 2 + (1 if ypad %2 != 0 else 0)),
        (xpad // 2, xpad // 2 + (1 if xpad % 2 != 0 else 0))
    ), 'constant', constant_values=0)

    assert result.shape == dims  # I don't trust myself enough.
    return result
