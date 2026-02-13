import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_scoring_model import CreditScoringModel


@pytest.fixture
def feature_cols():
    return ['age', 'income', 'credit_history_months',
            'num_credit_accounts', 'debt_ratio', 'num_late_payments']


@pytest.fixture
def sample_data(feature_cols):
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'age': np.random.randint(20, 70, n),
        'income': np.random.uniform(2000, 15000, n),
        'credit_history_months': np.random.randint(1, 240, n),
        'num_credit_accounts': np.random.randint(0, 10, n),
        'debt_ratio': np.random.uniform(0, 1, n),
        'num_late_payments': np.random.randint(0, 10, n),
    })
    # Simple target: lower debt_ratio and fewer late payments -> good
    df['target'] = ((df['debt_ratio'] < 0.5) & (df['num_late_payments'] < 5)).astype(int)
    return df


@pytest.fixture
def trained_model(sample_data, feature_cols):
    model = CreditScoringModel(feature_cols)
    model.fit(sample_data, sample_data['target'])
    return model


class TestCreditScoringModel:
    def test_init(self, feature_cols):
        model = CreditScoringModel(feature_cols)
        assert model.feature_cols == feature_cols
        assert model.is_fitted is False

    def test_fit(self, trained_model):
        assert trained_model.is_fitted is True
        assert trained_model.coefficients_ is not None
        assert len(trained_model.coefficients_) == 6

    def test_predict_proba_returns_probabilities(self, trained_model, sample_data):
        proba = trained_model.predict_proba(sample_data)
        assert len(proba) == len(sample_data)
        assert all(0 <= p <= 1 for p in proba)

    def test_predict_score_range(self, trained_model, sample_data):
        scores = trained_model.predict_score(sample_data)
        assert len(scores) == len(sample_data)
        assert all(300 <= s <= 850 for s in scores)

    def test_predict_score_ordering(self, trained_model, feature_cols):
        """Higher probability of good should produce higher scores."""
        good_applicant = pd.DataFrame([{
            'age': 45, 'income': 12000, 'credit_history_months': 120,
            'num_credit_accounts': 3, 'debt_ratio': 0.1, 'num_late_payments': 0
        }])
        bad_applicant = pd.DataFrame([{
            'age': 20, 'income': 2000, 'credit_history_months': 1,
            'num_credit_accounts': 0, 'debt_ratio': 0.9, 'num_late_payments': 10
        }])
        good_score = trained_model.predict_score(good_applicant)[0]
        bad_score = trained_model.predict_score(bad_applicant)[0]
        assert good_score > bad_score

    def test_summary(self, trained_model):
        summary = trained_model.summary()
        assert isinstance(summary, str)
        assert "Intercept" in summary

    def test_unfitted_model_raises(self, feature_cols):
        model = CreditScoringModel(feature_cols)
        df = pd.DataFrame([{
            'age': 30, 'income': 5000, 'credit_history_months': 60,
            'num_credit_accounts': 3, 'debt_ratio': 0.3, 'num_late_payments': 1
        }])
        with pytest.raises(Exception):
            model.predict_proba(df)
