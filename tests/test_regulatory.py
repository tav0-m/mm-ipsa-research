import unittest

from mm_ipsa.evaluation.regulatory import (
    BASEL_LEVEL,
    BASEL_OBSERVATIONS,
    basel_traffic_light,
    observations_for_power,
    zone_boundaries,
)

# Tabla publicada por el Comite para 250 jornadas al 99 por ciento.
COMMITTEE_MULTIPLIERS = {
    0: 3.00, 1: 3.00, 2: 3.00, 3: 3.00, 4: 3.00,
    5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85,
    10: 4.00, 15: 4.00,
}


class TestReproducesTheCommitteeTable(unittest.TestCase):
    def test_zone_boundaries_match_the_published_zones(self):
        self.assertEqual(zone_boundaries(BASEL_OBSERVATIONS, BASEL_LEVEL), (4, 9))

    def test_every_published_multiplier_is_reproduced(self):
        for exceptions, expected in COMMITTEE_MULTIPLIERS.items():
            report = basel_traffic_light(exceptions, BASEL_OBSERVATIONS, BASEL_LEVEL)
            multiplier = report["capital_multiplier"]
            self.assertIsNotNone(multiplier)
            self.assertAlmostEqual(float(multiplier or 0.0), expected, places=10)

    def test_zone_labels_match_the_published_ranges(self):
        for exceptions in range(0, 5):
            self.assertEqual(
                basel_traffic_light(exceptions, 250, 0.01)["zone"], "verde"
            )
        for exceptions in range(5, 10):
            self.assertEqual(
                basel_traffic_light(exceptions, 250, 0.01)["zone"], "amarilla"
            )
        for exceptions in (10, 11, 30):
            self.assertEqual(
                basel_traffic_light(exceptions, 250, 0.01)["zone"], "roja"
            )


class TestRescaling(unittest.TestCase):
    def test_boundaries_grow_with_the_sample(self):
        small = zone_boundaries(250, 0.01)
        large = zone_boundaries(1_000, 0.01)
        self.assertGreater(large[0], small[0])
        self.assertGreater(large[1], small[1])

    def test_multiplier_is_withheld_outside_the_committee_configuration(self):
        self.assertIsNone(basel_traffic_light(3, 500, 0.01)["capital_multiplier"])
        self.assertIsNone(basel_traffic_light(3, 250, 0.05)["capital_multiplier"])
        self.assertIsNotNone(
            basel_traffic_light(3, 250, 0.01)["capital_multiplier"]
        )

    def test_a_short_sample_is_declared_inadequate(self):
        short = basel_traffic_light(0, 121, 0.01)
        self.assertFalse(short["sample_is_adequate"])
        self.assertLess(short["expected_exceptions"], 2.0)
        self.assertIn("insuficiente", short["note"])

    def test_the_committee_sample_is_adequate(self):
        self.assertTrue(basel_traffic_light(0, 250, 0.01)["sample_is_adequate"])


class TestPValue(unittest.TestCase):
    def test_falls_as_exceptions_accumulate(self):
        values = [
            basel_traffic_light(count, 250, 0.01)["pvalue"] for count in range(0, 12)
        ]
        self.assertTrue(all(b <= a for a, b in zip(values, values[1:])))

    def test_zero_exceptions_cannot_be_evidence_against_the_model(self):
        self.assertAlmostEqual(basel_traffic_light(0, 250, 0.01)["pvalue"], 1.0)


class TestPowerRequirement(unittest.TestCase):
    def test_detecting_a_doubled_rate_needs_years_of_history(self):
        required = observations_for_power(0.01, detectable_ratio=2.0, power=0.80)
        self.assertGreater(required, BASEL_OBSERVATIONS)

    def test_a_larger_violation_is_easier_to_detect(self):
        self.assertLess(
            observations_for_power(0.01, detectable_ratio=4.0),
            observations_for_power(0.01, detectable_ratio=2.0),
        )

    def test_rejects_a_ratio_that_is_not_a_violation(self):
        with self.assertRaises(ValueError):
            observations_for_power(0.01, detectable_ratio=1.0)


class TestValidation(unittest.TestCase):
    def test_rejects_negative_exceptions(self):
        with self.assertRaises(ValueError):
            basel_traffic_light(-1, 250, 0.01)

    def test_rejects_more_exceptions_than_observations(self):
        with self.assertRaises(ValueError):
            basel_traffic_light(251, 250, 0.01)

    def test_rejects_an_invalid_level(self):
        for level in (0.0, 1.0, -0.5):
            with self.assertRaises(ValueError):
                zone_boundaries(250, level)

    def test_rejects_an_empty_sample(self):
        with self.assertRaises(ValueError):
            zone_boundaries(0, 0.01)


if __name__ == "__main__":
    unittest.main()
