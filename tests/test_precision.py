"""CPU tests of powersig.precision: the float32 / float64 regimes, their four switches, the import-time default,
the environment opt-in, and that the backends follow the regime and really compute in float32."""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from powersig import precision
from powersig.jax.algorithm import PowerSigJax
from powersig.jax.jax_config import configure_jax

try:
    import torch
    from powersig.torch.algorithm import PowerSigTorch
except ImportError:  # the torch backend is optional
    torch = None


def _paths(dtype, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((6, 3)).cumsum(0) * 0.3
    y = rng.standard_normal((9, 3)).cumsum(0) * 0.3
    return x.astype(dtype), y.astype(dtype)


class PrecisionTestCase(unittest.TestCase):
    """Every test restores the environment, the JAX flags, the torch switches and the explicit regime."""

    def setUp(self):
        self.env = {k: os.environ.get(k) for k in (precision.ENV_VAR, precision.TF32_ENV_VAR)}
        self.x64 = jax.config.jax_enable_x64
        self.matmul = jax.config.jax_default_matmul_precision
        self.state = dict(precision._state)
        if torch is not None:
            self.torch = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                          torch.get_float32_matmul_precision())

    def tearDown(self):
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        jax.config.update("jax_enable_x64", self.x64)
        jax.config.update("jax_default_matmul_precision", self.matmul)
        precision._state.update(self.state)
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.torch[:2]
            torch.set_float32_matmul_precision(self.torch[2])


class TestConfigure(PrecisionTestCase):
    def test_float32_regime_sets_every_switch(self):
        record = precision.configure("float32")
        self.assertEqual(precision.active(), "float32")
        self.assertEqual(os.environ[precision.TF32_ENV_VAR], "0")
        self.assertFalse(jax.config.jax_enable_x64)
        self.assertEqual(str(jax.config.jax_default_matmul_precision), "highest")
        self.assertEqual(record["regime"], "float32")
        self.assertEqual(record[precision.TF32_ENV_VAR], "0")
        self.assertFalse(record["jax_enable_x64"])
        self.assertEqual(record["jax_default_matmul_precision"], "highest")
        if torch is not None:
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            self.assertFalse(record["cuda_matmul_allow_tf32"])
            self.assertFalse(record["cudnn_allow_tf32"])
            self.assertEqual(record["float32_matmul_precision"], "highest")

    def test_float64_restores_the_historical_regime(self):
        precision.configure("float32")
        record = precision.configure("float64")
        self.assertEqual(precision.active(), "float64")
        self.assertTrue(jax.config.jax_enable_x64)
        self.assertTrue(record["jax_enable_x64"])
        self.assertEqual(os.environ[precision.TF32_ENV_VAR], "0")   # TF32 stays off unless asked for

    def test_tf32_opt_in(self):
        precision.configure("float32", tf32=True, matmul_precision="high")
        self.assertEqual(os.environ[precision.TF32_ENV_VAR], "1")
        self.assertEqual(str(jax.config.jax_default_matmul_precision), "high")
        if torch is not None:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")

    def test_normalize_accepts_names_widths_and_dtypes(self):
        for value in ("float32", "FP32", "f32", 32, np.float32, np.dtype("float32"), jnp.float32):
            self.assertEqual(precision.normalize(value), "float32", value)
        for value in ("float64", "double", 64, np.dtype("float64"), jnp.float64):
            self.assertEqual(precision.normalize(value), "float64", value)
        if torch is not None:
            self.assertEqual(precision.normalize(torch.float32), "float32")
            self.assertEqual(precision.normalize(torch.float64), "float64")
        for value in ("float16", "bfloat16", 16, None, True):
            with self.assertRaises(ValueError):
                precision.normalize(value)


class TestImportTimeDefault(PrecisionTestCase):
    def test_explicit_regime_survives_the_backend_import_hook(self):
        precision.configure("float32")
        self.assertEqual(precision.apply_backend_import("jax"), "explicit")
        configure_jax()   # what importing powersig.jax.algorithm runs
        self.assertFalse(jax.config.jax_enable_x64)
        self.assertEqual(precision.active(), "float32")

    def test_environment_variable_opts_in_at_import(self):
        precision.reset()
        os.environ[precision.ENV_VAR] = "float32"
        self.assertEqual(precision.apply_backend_import("jax"), "environment")
        self.assertEqual(precision.active(), "float32")
        self.assertFalse(jax.config.jax_enable_x64)
        self.assertEqual(os.environ[precision.TF32_ENV_VAR], "0")

    def test_historical_default_without_a_regime(self):
        precision.reset()
        os.environ.pop(precision.ENV_VAR, None)
        jax.config.update("jax_enable_x64", False)
        self.assertEqual(precision.apply_backend_import("jax"), "default")
        self.assertTrue(jax.config.jax_enable_x64)
        self.assertEqual(str(jax.config.jax_default_matmul_precision), "highest")
        self.assertIsNone(precision.active())
        self.assertEqual(PowerSigJax(order=4).dtype, jnp.float64)
        self.assertEqual(precision.default_dtype("numpy"), np.float64)


class TestBackendsFollowTheRegime(PrecisionTestCase):
    def test_jax_backend_computes_in_float32(self):
        precision.configure("float64")
        x64, y64 = _paths(np.float64)
        reference = float(PowerSigJax(order=8).compute_signature_kernel(jnp.asarray(x64), jnp.asarray(y64)))
        precision.configure("float32")
        ps = PowerSigJax(order=8)
        self.assertEqual(ps.dtype, jnp.float32)
        self.assertEqual(ps.ic.dtype, jnp.float32)
        x32, y32 = _paths(np.float32)
        value = ps.compute_signature_kernel(jnp.asarray(x32), jnp.asarray(y32))
        self.assertEqual(value.dtype, jnp.float32)
        self.assertLess(abs(float(value) - reference) / abs(reference), 1e-5)
        gram = ps.compute_gram_matrix(jnp.asarray(np.stack([x32, x32])), jnp.asarray(np.stack([y32, y32, y32])))   # no progress kwarg on every JAX version
        self.assertEqual(gram.dtype, jnp.float32)
        self.assertEqual(gram.shape, (2, 3))

    def test_explicit_dtype_argument_still_wins(self):
        precision.configure("float32")
        self.assertEqual(PowerSigJax(order=4, dtype=jnp.float64).dtype, jnp.float64)

    @unittest.skipIf(torch is None, "torch backend not installed")
    def test_torch_backend_follows_the_regime(self):
        precision.configure("float32")
        self.assertEqual(precision.apply_backend_import("torch"), "explicit")
        ps = PowerSigTorch(order=8, device=torch.device("cpu"))
        self.assertEqual(ps.dtype, torch.float32)
        x32, y32 = _paths(np.float32)
        gram = ps.compute_gram_matrix(torch.tensor(np.stack([x32])), torch.tensor(np.stack([y32])), show_progress=False)
        self.assertEqual(gram.dtype, torch.float32)
        precision.configure("float64")
        reference = PowerSigTorch(order=8, device=torch.device("cpu")).compute_gram_matrix(
            torch.tensor(np.stack([_paths(np.float64)[0]])), torch.tensor(np.stack([_paths(np.float64)[1]])), show_progress=False)
        self.assertEqual(reference.dtype, torch.float64)
        self.assertLess(abs(float(gram[0, 0]) - float(reference[0, 0])) / abs(float(reference[0, 0])), 1e-5)


if __name__ == "__main__":
    unittest.main()
