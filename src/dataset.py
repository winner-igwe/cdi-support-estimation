import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CDIDataset(Dataset):
    """
    HDF5-backed dataset for coherent diffraction support estimation.

    The HDF5 file stores raw diffraction intensity and ground-truth support.
    Normalization is applied when samples are loaded.
    """

    def __init__(self, h5_path, indices, normalization="max"):
        self.h5_path = h5_path
        self.indices = np.asarray(indices)
        self.normalization = normalization
        self.h5_file = None

    def __len__(self):
        return len(self.indices)

    def _normalize(self, diffraction):
        diffraction = diffraction.astype(np.float32)

        if self.normalization == "max":
            max_value = diffraction.max()

            if max_value > 0:
                diffraction = diffraction / max_value

        elif self.normalization == "log":
            diffraction = np.log1p(diffraction)

            max_value = diffraction.max()

            if max_value > 0:
                diffraction = diffraction / max_value

        elif self.normalization == "percentile":
            percentile_value = np.percentile(diffraction, 99)

            if percentile_value > 0:
                diffraction = np.clip(
                    diffraction,
                    0,
                    percentile_value
                )
                diffraction = diffraction / percentile_value

        else:
            raise ValueError(
                f"Unknown normalization: {self.normalization}"
            )

        return diffraction

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        real_idx = self.indices[idx]

        diffraction = self.h5_file["diffraction"][real_idx]
        support = self.h5_file["support"][real_idx]

        diffraction = self._normalize(diffraction)

        diffraction = diffraction[np.newaxis, :, :]
        support = support[np.newaxis, :, :]

        diffraction = torch.from_numpy(
            diffraction.astype(np.float32)
        )

        support = torch.from_numpy(
            support.astype(np.float32)
        )

        return diffraction, support
