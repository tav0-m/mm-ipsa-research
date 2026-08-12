import unittest

from mm_ipsa.config import load_config, objective_weights, target_parameters


class TestConfig(unittest.TestCase):
    def test_main_config_contract(self):
        cfg = load_config("config.yaml")
        self.assertLess(cfg["data"]["end_train"], cfg["data"]["start_oos"])
        self.assertIn("cov_offdiag_only", objective_weights(cfg["mm"]))
        params = target_parameters(cfg["mm"])
        self.assertAlmostEqual(params["decay_lambda"] ** 55, 0.5, places=10)
        self.assertLessEqual(cfg["mm"]["x_ftol"], 1e-12)
        self.assertGreaterEqual(cfg["mm"]["x_inner_max_iter"], 300)


if __name__ == "__main__":
    unittest.main()
