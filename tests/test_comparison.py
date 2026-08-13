import unittest

import numpy as np
import pandas as pd

from mm_ipsa.evaluation.comparison import (
    block_size_sensitivity,
    compare_focal_model,
    compare_focal_model_by_group,
    diebold_mariano,
    grouped_moving_block_bootstrap_loss_difference,
    grouped_newey_west_long_run_variance,
    holm_adjust,
    moving_block_bootstrap_loss_difference,
    newey_west_long_run_variance,
    politis_white_block_length,
    pooled_autocovariance,
    resolve_block_size,
    sample_autocovariance,
)


def _ar1(n: int, rho: float, seed: int) -> np.ndarray:
    """Serie AR(1) con dependencia conocida para validar el selector."""
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(n)
    series = np.empty(n)
    series[0] = innovations[0]
    for step in range(1, n):
        series[step] = rho * series[step - 1] + innovations[step]
    return series


class TestDieboldMariano(unittest.TestCase):
    def test_no_difference_is_not_rejected(self):
        rng = np.random.default_rng(0)
        losses = 0.1 + rng.standard_normal(200) * 0.01
        result = diebold_mariano(losses, losses.copy())
        self.assertEqual(result["degenerate"], 1.0)

    def test_systematic_advantage_is_detected(self):
        rng = np.random.default_rng(1)
        common = rng.standard_normal(300) * 0.01
        focal = 0.10 + common + rng.standard_normal(300) * 0.001
        benchmark = 0.11 + common + rng.standard_normal(300) * 0.001
        result = diebold_mariano(focal, benchmark)
        self.assertLess(result["mean_difference"], 0.0)
        self.assertLess(result["pvalue"], 0.01)

    def test_pure_noise_is_rarely_rejected(self):
        rejections = 0
        trials = 40
        for seed in range(trials):
            rng = np.random.default_rng(100 + seed)
            focal = 0.1 + rng.standard_normal(150) * 0.01
            benchmark = 0.1 + rng.standard_normal(150) * 0.01
            if diebold_mariano(focal, benchmark)["pvalue"] < 0.05:
                rejections += 1
        # Bajo la nula la tasa de rechazo debe rondar el nivel nominal.
        self.assertLess(rejections / trials, 0.20)

    def test_hac_variance_grows_with_positive_autocorrelation(self):
        independent, _ = newey_west_long_run_variance(_ar1(400, 0.0, 1))
        persistent, _ = newey_west_long_run_variance(_ar1(400, 0.7, 1))
        self.assertGreater(persistent, independent)

    def test_hac_lag_zero_reduces_to_sample_variance(self):
        series = _ar1(100, 0.5, 2)
        variance, lag = newey_west_long_run_variance(series, lag=0)
        self.assertEqual(lag, 0)
        self.assertAlmostEqual(variance, float(np.var(series)))

    def test_grouped_variance_matches_within_segment_definition(self):
        first = _ar1(60, 0.6, 3)
        second = _ar1(60, 0.6, 4) + 50.0
        segments = [first, second]
        grouped, truncation = grouped_newey_west_long_run_variance(segments)

        # Reconstruccion manual: solo productos rezagados dentro de cada tramo,
        # centrados por la media global y normalizados por el total.
        total = len(first) + len(second)
        grand_mean = float(np.concatenate(segments).mean())
        expected = 0.0
        for k in range(truncation + 1):
            weight = 1.0 if k == 0 else 2.0 * (1.0 - k / (truncation + 1.0))
            accumulated = 0.0
            for segment in segments:
                centered = segment - grand_mean
                accumulated += float(centered[: len(centered) - k] @ centered[k:])
            expected += weight * accumulated / total
        self.assertAlmostEqual(grouped, expected, places=9)

    def test_grouped_variance_differs_from_naive_concatenation(self):
        # Tramos con niveles opuestos: concatenar introduce productos de
        # frontera de signo contrario que el diseno por folds excluye.
        first = np.full(40, 1.0) + _ar1(40, 0.0, 5) * 0.01
        second = np.full(40, -1.0) + _ar1(40, 0.0, 6) * 0.01
        grouped, _ = grouped_newey_west_long_run_variance([first, second])
        concatenated, _ = newey_west_long_run_variance(np.concatenate([first, second]))
        self.assertNotAlmostEqual(grouped, concatenated, places=6)

    def test_rejects_short_or_mismatched_series(self):
        with self.assertRaises(ValueError):
            diebold_mariano(np.array([0.1, 0.2]), np.array([0.1, 0.2]))
        with self.assertRaises(ValueError):
            diebold_mariano(np.zeros(10), np.zeros(9))


class TestBlockLengthSelection(unittest.TestCase):
    def test_block_length_grows_with_serial_dependence(self):
        weak = np.median(
            [politis_white_block_length(_ar1(200, 0.0, s))["block_length"] for s in range(25)]
        )
        strong = np.median(
            [politis_white_block_length(_ar1(200, 0.8, s))["block_length"] for s in range(25)]
        )
        self.assertGreater(strong, weak)
        # Ruido independiente no necesita bloques largos.
        self.assertLessEqual(weak, 4.0)

    def test_constant_series_is_degenerate(self):
        result = politis_white_block_length(np.full(50, 0.3))
        self.assertEqual(result["block_length"], 1.0)
        self.assertEqual(result["degenerate"], 1.0)

    def test_max_block_is_respected(self):
        result = politis_white_block_length(_ar1(200, 0.9, 3), max_block=5)
        self.assertLessEqual(result["block_length"], 5.0)

    def test_requires_exactly_one_input_form(self):
        with self.assertRaises(ValueError):
            politis_white_block_length()
        with self.assertRaises(ValueError):
            politis_white_block_length(_ar1(50, 0.2, 1), segments=[_ar1(50, 0.2, 2)])

    def test_pooled_autocovariance_never_crosses_segments(self):
        first = np.array([1.0, 2.0, 3.0])
        second = np.array([10.0, 11.0, 12.0])
        pooled, total = pooled_autocovariance([first, second], 1)
        self.assertEqual(total, 6)
        # El producto entre el 3.0 y el 10.0 nunca debe aparecer: se calcula
        # manualmente el valor esperado acumulando solo dentro de cada tramo.
        mean = np.concatenate([first, second]).mean()
        expected = (
            float((first[:-1] - mean) @ (first[1:] - mean))
            + float((second[:-1] - mean) @ (second[1:] - mean))
        ) / 6.0
        self.assertAlmostEqual(pooled[1], expected)

    def test_pooled_estimate_is_stable_against_one_noisy_segment(self):
        # Tres tramos limpios e independientes y uno corto y ruidoso.
        segments = [_ar1(49, 0.1, s) for s in range(3)] + [_ar1(12, 0.95, 99)]
        pooled = politis_white_block_length(segments=segments, max_block=12)
        worst = max(
            politis_white_block_length(segment, max_block=12)["block_length"]
            for segment in segments
        )
        self.assertLessEqual(pooled["block_length"], worst)

    def test_sample_autocovariance_matches_definition(self):
        series = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = sample_autocovariance(series, 2)
        centered = series - series.mean()
        self.assertAlmostEqual(result[0], float(centered @ centered) / 5)
        self.assertAlmostEqual(result[2], float(centered[:3] @ centered[2:]) / 5)

    def test_resolver_fixed_mode_returns_configured_value(self):
        chosen, diagnostics = resolve_block_size(
            _ar1(60, 0.5, 2), mode="fixed", configured=4, max_block=30
        )
        self.assertEqual(chosen, 4)
        self.assertEqual(diagnostics["block_size_mode_auto"], 0.0)

    def test_resolver_flags_when_dependence_exceeds_ceiling(self):
        chosen, diagnostics = resolve_block_size(
            _ar1(300, 0.95, 4), mode="auto", configured=4, max_block=3
        )
        self.assertEqual(chosen, 3)
        self.assertEqual(diagnostics["block_size_capped"], 1.0)

    def test_resolver_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            resolve_block_size(_ar1(40, 0.1, 1), mode="politis", configured=4, max_block=10)


class TestScoreComparison(unittest.TestCase):
    def test_identical_losses_have_zero_interval_and_no_rejection(self):
        losses = np.linspace(0.1, 0.2, 30)
        result = moving_block_bootstrap_loss_difference(
            losses,
            losses,
            block_size=4,
            samples=500,
            confidence_level=0.95,
            seed=7,
        )
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.0)
        self.assertEqual(result["pvalue_raw"], 1.0)

    def test_negative_difference_means_focal_is_better(self):
        benchmark = np.linspace(1.0, 2.0, 40)
        focal = benchmark - 0.2
        result = moving_block_bootstrap_loss_difference(
            focal,
            benchmark,
            block_size=5,
            samples=500,
            confidence_level=0.95,
            seed=11,
        )
        self.assertLess(result["mean_difference"], 0.0)
        self.assertLess(result["ci_high"], 0.0)
        self.assertGreater(result["probability_focal_better"], 0.99)

    def test_interval_and_equal_tail_test_are_directionally_coherent(self):
        rng = np.random.default_rng(101)
        benchmark = rng.normal(1.0, 0.1, size=80)
        focal = benchmark + rng.normal(-0.08, 0.01, size=80)
        result = moving_block_bootstrap_loss_difference(
            focal,
            benchmark,
            block_size=4,
            samples=1_000,
            confidence_level=0.95,
            seed=19,
        )
        self.assertLess(result["ci_high"], 0.0)
        self.assertLess(result["pvalue_raw"], 0.05)

    def test_holm_adjustment_is_valid_and_monotone(self):
        raw = np.array([0.001, 0.02, 0.04, 0.8])
        adjusted = holm_adjust(raw)
        self.assertTrue(np.all(adjusted >= raw))
        order = np.argsort(raw)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-15))

    def test_comparison_builds_nine_paired_contrasts(self):
        observations = [f"t{i:02d}" for i in range(24)]
        rows = []
        offsets = {
            "MM": 0.00,
            "gaussian_terminal": 0.02,
            "student_t_terminal": -0.01,
            "historical_weighted": 0.01,
        }
        for model, offset in offsets.items():
            for index, observation in enumerate(observations):
                base = 0.1 + index / 1_000
                rows.append(
                    {
                        "model": model,
                        "observation": observation,
                        "mean_crps": base + offset,
                        "energy_score": 2 * base + offset,
                        "variogram_score": 3 * base + offset,
                    }
                )
        frame = compare_focal_model(
            pd.DataFrame(rows),
            focal_model="MM",
            benchmark_models=[
                "gaussian_terminal",
                "student_t_terminal",
                "historical_weighted",
            ],
            block_size=4,
            samples=500,
            confidence_level=0.95,
            seed=5,
        )
        self.assertEqual(len(frame), 9)
        self.assertEqual(int(frame["registered_primary"].sum()), 1)
        self.assertTrue((frame["difference_direction"] == "focal_minus_benchmark").all())
        self.assertTrue((frame["pvalue_holm"] >= frame["pvalue_raw"]).all())

    def test_auto_mode_records_the_selected_block_per_contrast(self):
        rng = np.random.default_rng(11)
        observations = [f"t{i:03d}" for i in range(120)]
        rows = []
        for model, offset in {
            "MM": 0.0,
            "gaussian_terminal": 0.02,
            "student_t_terminal": -0.01,
        }.items():
            noise = rng.standard_normal(len(observations)) / 500
            for index, observation in enumerate(observations):
                base = 0.1 + noise[index]
                rows.append(
                    {
                        "model": model,
                        "observation": observation,
                        "mean_crps": base + offset,
                        "energy_score": 2 * base + offset,
                        "variogram_score": 3 * base + offset,
                    }
                )
        frame = compare_focal_model(
            pd.DataFrame(rows),
            focal_model="MM",
            benchmark_models=["gaussian_terminal", "student_t_terminal"],
            block_size=4,
            block_size_mode="auto",
            samples=500,
            confidence_level=0.95,
            seed=5,
        )
        self.assertTrue((frame["block_size_mode_auto"] == 1.0).all())
        self.assertTrue((frame["block_size"] >= 1).all())
        self.assertTrue((frame["block_size"] <= len(observations)).all())

    def test_sensitivity_repeats_every_contrast_per_block(self):
        observations = [f"t{i:02d}" for i in range(40)]
        rows = []
        for model, offset in {"MM": 0.0, "gaussian_terminal": 0.02}.items():
            for index, observation in enumerate(observations):
                base = 0.1 + index / 1_000
                rows.append(
                    {
                        "model": model,
                        "observation": observation,
                        "mean_crps": base + offset,
                        "energy_score": 2 * base + offset,
                        "variogram_score": 3 * base + offset,
                    }
                )
        grid = [2, 4, 8]
        frame = block_size_sensitivity(
            pd.DataFrame(rows),
            focal_model="MM",
            benchmark_models=["gaussian_terminal"],
            block_sizes=grid,
            samples=200,
            seed=3,
        )
        self.assertEqual(len(frame), len(grid) * 3)
        self.assertEqual(sorted(frame["requested_block_size"].unique()), grid)
        # El modo fijo debe respetarse exactamente en cada repeticion.
        self.assertTrue((frame["block_size"] == frame["requested_block_size"]).all())

    def test_sensitivity_rejects_empty_grid(self):
        rows = [
            {
                "model": model,
                "observation": f"t{i}",
                "mean_crps": 0.1,
                "energy_score": 0.2,
                "variogram_score": 0.3,
            }
            for model in ("MM", "gaussian_terminal")
            for i in range(10)
        ]
        with self.assertRaises(ValueError):
            block_size_sensitivity(
                pd.DataFrame(rows),
                focal_model="MM",
                benchmark_models=["gaussian_terminal"],
                block_sizes=[],
            )

    def test_grouped_bootstrap_preserves_zero_difference(self):
        losses = np.linspace(0.1, 0.3, 24)
        groups = np.repeat(["fold_a", "fold_b"], 12)
        result = grouped_moving_block_bootstrap_loss_difference(
            losses,
            losses,
            groups,
            block_size=4,
            samples=500,
            confidence_level=0.95,
            seed=31,
        )
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.0)
        self.assertEqual(result["n_groups"], 2.0)

    def test_grouped_comparison_builds_nine_contrasts(self):
        rows = []
        offsets = {
            "MM": 0.00,
            "gaussian_terminal": 0.02,
            "student_t_terminal": -0.01,
            "historical_weighted": 0.01,
        }
        for fold in ("a", "b"):
            for model, offset in offsets.items():
                for index in range(12):
                    base = 0.1 + index / 1_000
                    rows.append(
                        {
                            "fold_id": fold,
                            "model": model,
                            "observation": f"{fold}-{index:02d}",
                            "mean_crps": base + offset,
                            "energy_score": 2 * base + offset,
                            "variogram_score": 3 * base + offset,
                        }
                    )
        frame = compare_focal_model_by_group(
            pd.DataFrame(rows),
            group_column="fold_id",
            focal_model="MM",
            benchmark_models=[
                "gaussian_terminal",
                "student_t_terminal",
                "historical_weighted",
            ],
            block_size=4,
            samples=500,
            confidence_level=0.95,
            seed=37,
        )
        self.assertEqual(len(frame), 9)
        self.assertTrue((frame["n_groups"] == 2.0).all())
        self.assertTrue((frame["n_observations"] == 24).all())
        self.assertEqual(int(frame["registered_primary"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
