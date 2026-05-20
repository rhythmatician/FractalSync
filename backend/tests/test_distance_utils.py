import numpy as np
import torch
from src import distance_utils as du


def test_numpy_complex_input():
    arr = np.array([0+0j, 2+0j, -1+0j], dtype=np.complex64)
    out = du.mandelbrot_distance_estimate_from_any(arr)
    assert isinstance(out, list)
    assert len(out) == 3


def test_torch_complex_input():
    t = torch.tensor([0+0j, 2+0j, -1+0j], dtype=torch.complex64)
    out = du.mandelbrot_distance_estimate_from_any(t)
    assert isinstance(out, list)
    assert len(out) == 3


def test_sequence_input():
    seq = [0+0j, 2+0j, -1+0j]
    out = du.mandelbrot_distance_estimate_from_any(seq)
    assert isinstance(out, list)
    assert len(out) == 3
