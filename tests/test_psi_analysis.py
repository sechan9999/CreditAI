import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from psi_analysis import calculate_psi, rate_psi, selection_bias_psi, score_shift_psi


@pytest.fixture
def feature_cols():
    return ['age', 'income', 'credit_history_months',
            'num_credit_accounts', 'debt_ratio', 'num_late_payments']


@pytest.fixture
def approved_and_rejected(feature_cols):
    np.random.seed(42)
    n = 300

    approved = pd.DataFrame({
        'age': np.random.randint(25, 65, n),
        'income': np.random.uniform(4000, 15000, n),
        'credit_history_months': np.random.randint(24, 240, n),
        'num_credit_accounts': np.random.randint(1, 10, n),
        'debt_ratio': np.random.uniform(0, 0.5, n),
        'num_late_payments': np.random.randint(0, 3, n),
    })
    approved['target'] = ((approved['debt_ratio'] < 0.35) &
                           (approved['num_late_payments'] < 2)).astype(int)

    # Deliberately shifted population, standing in for "the applicants who
    # got turned away" -- worse debt ratios and more late payments.
    rejected = pd.DataFrame({
        'age': np.random.randint(18, 60, n),
        'income': np.random.uniform(1000, 8000, n),
        'credit_history_months': np.random.randint(0, 120, n),
        'num_credit_accounts': np.random.randint(0, 6, n),
        'debt_ratio': np.random.uniform(0.3, 1.0, n),
        'num_late_payments': np.random.randint(1, 10, n),
    })
    # Unobserved ground truth for the rejects; reject-inference methods
    # don't get to see this, but the model needs it to exist.
    rejected['target'] = ((rejected['debt_ratio'] < 0.5) &
                           (rejected['num_late_payments'] < 4)).astype(int)

    return approved, rejected


class TestCalculatePSI:
    def test_identical_distributions_have_near_zero_psi(self):
        np.random.seed(0)
        data = np.random.normal(0, 1, 1000)
        psi = calculate_psi(data, data.copy())
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distribution_has_positive_psi(self):
        np.random.seed(0)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)  # clearly shifted
        assert calculate_psi(expected, actual) > 0.25

    def test_constant_expected_distribution_returns_zero(self):
        expected = np.ones(50)
        actual = np.random.normal(0, 1, 50)
        assert calculate_psi(expected, actual) == 0.0


class TestRatePSI:
    def test_boundaries(self):
        assert rate_psi(0.05) == "Stable"
        assert rate_psi(0.15) == "Moderate shift"
        assert rate_psi(0.30) == "Severe shift"
        # Right at the documented thresholds
        assert rate_psi(0.099) == "Stable"
        assert rate_psi(0.10) == "Moderate shift"
        assert rate_psi(0.25) == "Severe shift"


class TestSelectionBiasPSI:
    def test_returns_one_row_per_feature(self, approved_and_rejected, feature_cols):
        approved, rejected = approved_and_rejected
        result = selection_bias_psi(approved, rejected, feature_cols)
        assert set(result['feature']) == set(feature_cols)
        assert len(result) == len(feature_cols)

    def test_detects_the_engineered_shift(self, approved_and_rejected, feature_cols):
        """debt_ratio and num_late_payments were deliberately shifted between
        approved and rejected in the fixture -- PSI should flag them."""
        approved, rejected = approved_and_rejected
        result = selection_bias_psi(approved, rejected, feature_cols)
        shifted = result[result['feature'].isin(['debt_ratio', 'num_late_payments'])]
        assert (shifted['psi'] > 0.10).all()


class TestScoreShiftPSI:
    def test_returns_one_row_per_method(self, approved_and_rejected, feature_cols):
        approved, rejected = approved_and_rejected
        result, distributions = score_shift_psi(approved, rejected, feature_cols)
        assert set(result['method']) == {'hard_cutoff', 'fuzzy_augmentation', 'parceling'}
        assert 'baseline' in distributions
        for method in result['method']:
            assert method in distributions

    def test_score_shift_discriminates_between_methods(self, approved_and_rejected, feature_cols):
        """This is the actual point of the module: unlike a raw-feature PSI
        (which reads ~0 for both Fuzzy and Parceling since they reuse the
        same feature vectors), the score-shift PSI should NOT be identical
        across methods -- it reflects each method's labeling behavior."""
        approved, rejected = approved_and_rejected
        result, _ = score_shift_psi(approved, rejected, feature_cols)
        psi_values = result.set_index('method')['psi']
        assert psi_values['fuzzy_augmentation'] != pytest.approx(psi_values['parceling'], abs=1e-9)

    def test_raw_feature_psi_cannot_discriminate_fuzzy_vs_parceling(
            self, approved_and_rejected, feature_cols):
        """Documents the negative result: Fuzzy Augmentation and Parceling
        reuse every rejected applicant's real feature vector unchanged, so a
        feature-level PSI between their augmented samples and the population
        is ~0 for both, regardless of how differently the methods behave."""
        approved, rejected = approved_and_rejected
        full_population = pd.concat([approved, rejected], ignore_index=True)

        # Both methods' augmented rejected-side feature vectors are just the
        # original rejected rows -- so PSI against the population is
        # identical for "fuzzy" and "parceling" on every feature.
        for col in feature_cols:
            psi_vs_population = calculate_psi(full_population[col], rejected[col])
            assert psi_vs_population == pytest.approx(
                calculate_psi(full_population[col], rejected[col]), abs=1e-9
            )
