# Copyright (c) OpenMMLab. All rights reserved.
"""Dependency-free regression for the premature BF16 store in Ascend CE."""

import math
import struct
import unittest


def _fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


class TargetRoundingTest(unittest.TestCase):
    def test_high_confidence_target_retains_small_derivative(self):
        # A synthetic two-class softmax with target probability near one.
        p = _fp32(1.0 / (1.0 + math.exp(-10.75)))
        upstream = _fp32(2.3862588705e-7)
        old = _bf16(_fp32(_bf16(_fp32(p * upstream)) - upstream))
        fixed = _bf16(_fp32(_fp32(p - 1.0) * upstream))
        reference = (p - 1.0) * upstream
        self.assertGreater(abs(old), 30 * abs(reference))
        self.assertLess(abs(fixed - reference), abs(old - reference) / 100)

    def test_low_precision_store_before_subtraction_can_zero_a_gradient(self):
        p = _fp32(0.999)
        self.assertEqual(_bf16(_bf16(p) - 1.0), 0.0)
        self.assertNotEqual(_bf16(p - 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
