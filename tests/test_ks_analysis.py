import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ks_analysis import KSCalculator


class TestKSCalculator:
    def test_perfect_separation(self):
        """Perfect model should have KS near 100%."""
        y_true = np.array([1]*50 + [0]*50)
        y_prob = np.array([0.9]*50 + [0.1]*50)
        ks = KSCalculator(y_true, y_prob, n_bins=10)
        assert ks.ks_statistic > 80

    def test_random_model(self):
        """Random model should have low KS."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.random(1000)
        ks = KSCalculator(y_true, y_prob, n_bins=10)
        assert ks.ks_statistic < 15

    def test_ks_table_shape(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.random(200)
        ks = KSCalculator(y_true, y_prob, n_bins=10)
        assert len(ks.ks_table) == 10
        assert 'decile' in ks.ks_table.columns
        assert 'ks' in ks.ks_table.columns

    def test_ks_decile_valid(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.random(200)
        ks = KSCalculator(y_true, y_prob, n_bins=10)
        assert 1 <= ks.ks_decile <= 10

    def test_summary_returns_string(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.random(200)
        ks = KSCalculator(y_true, y_prob, n_bins=10)
        summary = ks.summary()
        assert isinstance(summary, str)
        assert "Maximum KS" in summary
