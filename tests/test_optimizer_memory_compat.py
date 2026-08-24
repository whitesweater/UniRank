from __future__ import annotations

import unittest

import torch

from unirank.pytorch.torch_utils import chunked_adagrad_step, get_optimizer


class OptimizerMemoryCompatibilityTest(unittest.TestCase):
    def test_adagrad_foreach_can_be_disabled_explicitly(self):
        parameter = torch.nn.Parameter(torch.ones(4))
        optimizer = get_optimizer(
            "Adagrad",
            [parameter],
            0.05,
            optimizer_kwargs={"foreach": False},
        )
        self.assertFalse(optimizer.defaults["foreach"])

    def test_default_optimizer_path_remains_automatic(self):
        parameter = torch.nn.Parameter(torch.ones(4))
        optimizer = get_optimizer("Adagrad", [parameter], 0.05)
        self.assertIsNone(optimizer.defaults["foreach"])

    def test_chunked_adagrad_matches_standard_updates(self):
        standard_parameter = torch.nn.Parameter(torch.linspace(-1, 1, 17))
        chunked_parameter = torch.nn.Parameter(standard_parameter.detach().clone())
        standard = torch.optim.Adagrad(
            [standard_parameter],
            lr=0.05,
            foreach=False,
        )
        chunked = torch.optim.Adagrad(
            [chunked_parameter],
            lr=0.05,
            foreach=False,
        )

        for step in range(3):
            gradient = torch.linspace(0.1 + step, 1.7 + step, 17)
            standard_parameter.grad = gradient.clone()
            chunked_parameter.grad = gradient.clone()
            standard.step()
            chunked_adagrad_step(chunked, chunk_size=5)

        torch.testing.assert_close(
            chunked_parameter,
            standard_parameter,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            chunked.state[chunked_parameter]["sum"],
            standard.state[standard_parameter]["sum"],
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
