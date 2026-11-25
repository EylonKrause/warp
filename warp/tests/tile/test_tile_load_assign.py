# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests tile element assignment with 1024-thread blocks.

Covers one- through four-dimensional writes into local tiles. These tests are
forward-only. Add tests here when they assign individual tile elements rather
than storing complete tiles.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.set_module_options({"enable_backward": False})

TILE_SIZE = 4


@wp.kernel
def tile_assign_1d_kernel(input: wp.array1d[float], output: wp.array1d[float]):
    i = wp.tid()

    t = wp.tile_zeros(shape=(TILE_SIZE,), dtype=float)

    # assign to tile
    t[i] = input[i] * 2.0

    output[i] = t[i]


@wp.kernel
def tile_assign_2d_kernel(input: wp.array2d[float], output: wp.array2d[float]):
    i, j = wp.tid()

    t = wp.tile_zeros(shape=(TILE_SIZE, TILE_SIZE), dtype=float)

    # assign to tile
    t[i, j] = input[i, j] * 2.0

    output[i, j] = t[i, j]


@wp.kernel
def tile_assign_3d_kernel(input: wp.array3d[float], output: wp.array3d[float]):
    i, j, k = wp.tid()

    t = wp.tile_zeros(shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE), dtype=float)

    # assign to tile
    t[i, j, k] = input[i, j, k] * 2.0

    output[i, j, k] = t[i, j, k]


@wp.kernel
def tile_assign_4d_kernel(input: wp.array4d[float], output: wp.array4d[float]):
    i, j, k, l = wp.tid()

    t = wp.tile_zeros(shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE), dtype=float)

    # assign to tile
    t[i, j, k, l] = input[i, j, k, l] * 2.0

    output[i, j, k, l] = t[i, j, k, l]


def test_tile_assign(kernel, ndim):
    shape = (TILE_SIZE,) * ndim

    def test_run(test, device):
        rng = np.random.default_rng(42)

        input = wp.array(rng.random(shape), dtype=float, requires_grad=True, device=device)
        output = wp.zeros_like(input)

        wp.launch(
            kernel,
            dim=shape,
            inputs=[input, output],
            block_dim=1024,
            device=device,
        )

        assert_np_equal(output.numpy(), input.numpy() * 2.0)

    return test_run


@wp.func
def _busy(n: float) -> float:
    # loops n times and returns 0
    s = float(0.0)
    for _ in range(int(n)):
        s = s * 0.5 + 1.0
    return wp.where(s > 1.0e30, 1.0, 0.0)


@wp.kernel
def tile_assign_after_read_kernel(y_in: wp.array2d[float], work: wp.array2d[float], output: wp.array3d[float]):
    i = wp.tid()
    y = wp.tile_load(y_in, shape=(32, 32), storage="shared")
    w = wp.tile_load(work, shape=(32, 32))
    acc = wp.tile_zeros(shape=(32, 32), dtype=float)
    for it in range(4):
        # adds 0 but makes the odd rows finish their reads of y late
        acc += wp.tile_map(_busy, w)
        acc += y
        y[1, 0] = float(it + 100)
    wp.tile_store(output[i], acc)


def test_tile_assign_after_read(test, device):
    """Element assignment must not overwrite a shared tile while other threads still read it (GH-2306)."""
    num_blocks = 256 if device.is_cuda else 2

    rng = np.random.default_rng(0)
    y_np = rng.integers(0, 8, size=(32, 32)).astype(np.float32)
    work_np = np.zeros((32, 32), dtype=np.float32)
    work_np[1::2] = 2000.0

    output = wp.zeros((num_blocks, 32, 32), dtype=float, device=device)
    wp.launch_tiled(
        tile_assign_after_read_kernel,
        dim=num_blocks,
        inputs=[wp.array(y_np, device=device), wp.array(work_np, device=device)],
        outputs=[output],
        block_dim=64,
        device=device,
    )

    expected = 4.0 * y_np
    expected[1, 0] = y_np[1, 0] + 100.0 + 101.0 + 102.0
    assert_np_equal(output.numpy(), np.broadcast_to(expected, output.shape))


devices = get_test_devices()


class TestTileLoadAssign(unittest.TestCase):
    pass


add_function_test(
    TestTileLoadAssign,
    "test_tile_assign_2d_cpu_blocks",
    test_tile_assign(tile_assign_2d_kernel, 2),
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileLoadAssign,
    "test_tile_assign_1d",
    test_tile_assign(tile_assign_1d_kernel, 1),
    devices=devices,
)
add_function_test(
    TestTileLoadAssign,
    "test_tile_assign_2d",
    test_tile_assign(tile_assign_2d_kernel, 2),
    devices=devices,
)
add_function_test(
    TestTileLoadAssign,
    "test_tile_assign_3d",
    test_tile_assign(tile_assign_3d_kernel, 3),
    devices=devices,
)
add_function_test(
    TestTileLoadAssign,
    "test_tile_assign_4d",
    test_tile_assign(tile_assign_4d_kernel, 4),
    devices=devices,
)
add_function_test(
    TestTileLoadAssign,
    "test_tile_assign_after_read",
    test_tile_assign_after_read,
    devices=devices,
)
add_function_test(
    TestTileLoadAssign,
    "test_tile_assign_after_read_cpu_blocks",
    test_tile_assign_after_read,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
