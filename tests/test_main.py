import runpy
import unittest
from unittest.mock import patch


class MainTests(unittest.TestCase):
    def test_loading_experiment_settings_does_not_start_an_inversion(self):
        with patch("src.data_prepare.prepare_model") as prepare:
            settings = runpy.run_module("main")
        prepare.assert_not_called()
        self.assertTrue(settings["prune_gaussians"])
