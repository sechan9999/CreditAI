import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from generate_data import generate_telecom_data


class TestDataGeneration:
    def test_default_sizes(self):
        approved, rejected = generate_telecom_data()
        assert len(approved) == 5000
        assert len(rejected) == 2000

    def test_custom_sizes(self):
        approved, rejected = generate_telecom_data(n_approved=100, n_rejected=50)
        assert len(approved) == 100
        assert len(rejected) == 50

    def test_approved_has_target(self):
        approved, _ = generate_telecom_data(n_approved=100, n_rejected=50)
        assert 'target' in approved.columns
        assert approved['target'].isin([0, 1]).all()

    def test_rejected_has_nan_target(self):
        _, rejected = generate_telecom_data(n_approved=100, n_rejected=50)
        assert 'target' in rejected.columns
        assert rejected['target'].isna().all()

    def test_status_column(self):
        approved, rejected = generate_telecom_data(n_approved=100, n_rejected=50)
        assert (approved['status'] == 'approved').all()
        assert (rejected['status'] == 'rejected').all()

    def test_feature_ranges(self):
        approved, rejected = generate_telecom_data(n_approved=500, n_rejected=200)
        for df in [approved, rejected]:
            assert df['age'].between(18, 70).all()
            assert (df['debt_ratio'] >= 0).all() and (df['debt_ratio'] <= 1).all()
            assert (df['num_late_payments'] >= 0).all()
            assert (df['income'] >= 0).all()

    def test_required_columns_exist(self):
        approved, rejected = generate_telecom_data(n_approved=100, n_rejected=50)
        expected_cols = {'age', 'income', 'credit_history_months',
                         'num_credit_accounts', 'debt_ratio', 'num_late_payments',
                         'target', 'status'}
        assert expected_cols.issubset(set(approved.columns))
        assert expected_cols.issubset(set(rejected.columns))
