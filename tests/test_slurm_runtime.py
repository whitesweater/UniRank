from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from unirank.pytorch.torch_utils import setup_visible_devices


class SlurmVisibleDevicesTest(unittest.TestCase):
    def test_preserves_slurm_cuda_visible_devices(self):
        environment = {
            "SLURM_JOB_ID": "123",
            "CUDA_VISIBLE_DEVICES": "3",
        }
        with patch.dict(os.environ, environment, clear=True):
            setup_visible_devices([0])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_rejects_unallocated_logical_device(self):
        environment = {
            "SLURM_JOB_ID": "123",
            "CUDA_VISIBLE_DEVICES": "3",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ValueError):
                setup_visible_devices([1])

    def test_sets_devices_outside_slurm(self):
        with patch.dict(os.environ, {}, clear=True):
            setup_visible_devices([2, 4])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,4")


if __name__ == "__main__":
    unittest.main()
