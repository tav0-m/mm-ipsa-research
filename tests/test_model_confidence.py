import unittest

import numpy as np
import pandas as pd

from mm_ipsa.evaluation.model_confidence import (
    block_bootstrap_indices,
    model_confidence_set,
)


def _correlated_losses(
    n: int, offsets: dict[str, float], noise: float, seed: int
) -> pd.DataFrame:
    """Perdidas con un componente comun, como ocurre al evaluar sobre las mismas fechas."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n) * 0.01
    return pd.DataFrame(
        {
            name: 0.1 + offset + common + rng.standard_normal(n) * noise
            for name, offset in offsets.items()
        }
    )


class TestModelConfidenceSet(unittest.TestCase):
    def test_equivalent_models_all_survive(self):
        losses = _correlated_losses(
            250, {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}, 0.002, 5
        )
        result = model_confidence_set(losses, samples=1_000, seed=1)
        self.assertTrue(result["in_mcs"].all())

    def test_clearly_worse_model_is_excluded(self):
        losses = _correlated_losses(
            300, {"a": 0.0, "b": 0.0, "c": 0.0, "peor": 0.4}, 0.002, 7
        )
        result = model_confidence_set(losses, samples=1_000, seed=1)
        excluded = result.loc[~result["in_mcs"], "model"].tolist()
        self.assertEqual(excluded, ["peor"])

    def test_pvalues_are_monotone_in_elimination_order(self):
        losses = _correlated_losses(
            200, {"a": 0.0, "b": 0.05, "c": 0.15, "d": 0.30}, 0.003, 3
        )
        result = model_confidence_set(losses, samples=1_000, seed=2).sort_values("step")
        values = result["mcs_pvalue"].to_numpy()
        # El p-valor MCS se define como maximo acumulado; nunca decrece.
        self.assertTrue(np.all(np.diff(values) >= -1e-12))

    def test_best_model_always_belongs_to_the_set(self):
        losses = _correlated_losses(
            200, {"a": 0.0, "b": 0.2, "c": 0.4}, 0.002, 11
        )
        result = model_confidence_set(losses, samples=1_000, seed=4)
        best = result.loc[result["mean_loss"].idxmin(), "model"]
        self.assertTrue(bool(result.loc[result["model"] == best, "in_mcs"].iloc[0]))

    def test_alpha_controls_set_size(self):
        losses = _correlated_losses(
            200, {"a": 0.0, "b": 0.02, "c": 0.05, "d": 0.09}, 0.01, 13
        )
        strict = model_confidence_set(losses, alpha=0.5, samples=1_000, seed=6)
        lenient = model_confidence_set(losses, alpha=0.01, samples=1_000, seed=6)
        self.assertLessEqual(int(strict["in_mcs"].sum()), int(lenient["in_mcs"].sum()))

    def test_rejects_invalid_inputs(self):
        losses = _correlated_losses(50, {"a": 0.0, "b": 0.1}, 0.002, 1)
        with self.assertRaises(ValueError):
            model_confidence_set(losses.iloc[:, :1], samples=1_000)
        with self.assertRaises(ValueError):
            model_confidence_set(losses, alpha=1.5, samples=1_000)
        with self.assertRaises(ValueError):
            model_confidence_set(losses, samples=10)
        with self.assertRaises(TypeError):
            # Se pasa un ndarray a proposito: el contrato exige DataFrame para
            # que los nombres de modelo viajen con las perdidas.
            model_confidence_set(losses.to_numpy(), samples=1_000)  # type: ignore[arg-type]

    def test_missing_values_are_rejected(self):
        losses = _correlated_losses(50, {"a": 0.0, "b": 0.1}, 0.002, 1)
        losses.iloc[3, 0] = np.nan
        with self.assertRaises(ValueError):
            model_confidence_set(losses, samples=1_000)


class TestBlockBootstrapIndices(unittest.TestCase):
    def test_blocks_never_cross_group_boundaries(self):
        groups = np.repeat(["f1", "f2"], 20)
        rng = np.random.default_rng(0)
        indices = block_bootstrap_indices(40, 5, 50, rng, groups)
        # Las primeras 20 posiciones deben provenir del primer fold y el resto
        # del segundo; de lo contrario el remuestreo mezclaria periodos.
        self.assertTrue(np.all(indices[:, :20] < 20))
        self.assertTrue(np.all(indices[:, 20:] >= 20))

    def test_shape_and_range(self):
        rng = np.random.default_rng(1)
        indices = block_bootstrap_indices(30, 3, 12, rng)
        self.assertEqual(indices.shape, (12, 30))
        self.assertTrue(np.all((indices >= 0) & (indices < 30)))

    def test_block_size_cannot_exceed_segment_length(self):
        rng = np.random.default_rng(2)
        groups = np.repeat(["a", "b"], 5)
        with self.assertRaises(ValueError):
            block_bootstrap_indices(10, 6, 5, rng, groups)

    def test_rejects_nonpositive_parameters(self):
        rng = np.random.default_rng(3)
        with self.assertRaises(ValueError):
            block_bootstrap_indices(10, 0, 5, rng)
        with self.assertRaises(ValueError):
            block_bootstrap_indices(10, 2, 0, rng)


if __name__ == "__main__":
    unittest.main()
