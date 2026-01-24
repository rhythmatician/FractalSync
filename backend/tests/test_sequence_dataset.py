from __future__ import annotations

import numpy as np
import torch

from src.data_loader import SequenceAudioDataset


def test_sequence_counts_and_boundaries():
    # File 0: 12 frames, File 1: 3 frames (should be skipped)
    feats0 = np.arange(12 * 2, dtype=np.float32).reshape((12, 2))
    feats1 = np.arange(3 * 2, dtype=np.float32).reshape((3, 2))
    ds = SequenceAudioDataset(features_list=[feats0, feats1], seq_len=5, stride=2)

    # For feats0: starts at 0,2,4,6 -> 4 sequences
    assert len(ds) == 4

    seq0, fi0, start0 = ds[0]
    assert fi0 == 0
    assert start0 == 0
    assert seq0.shape == (5, 2)

    # Check last sequence start
    seq_last, fi_last, start_last = ds[len(ds) - 1]
    assert start_last == 6
    assert (seq_last == feats0[6:11]).all()


def test_to_tensor_dataset_and_dataloader_collation():
    feats = np.random.rand(10, 6).astype("float32")
    ds = SequenceAudioDataset(features_list=[feats], seq_len=5, stride=1)
    td = ds.to_tensor_dataset()
    assert len(td) == 6

    loader = torch.utils.data.DataLoader(td, batch_size=2)
    batch = next(iter(loader))[0]
    assert batch.shape == (2, 5, 6)
