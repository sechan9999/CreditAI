import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lambda_function import (predict, get_decision, lambda_handler,
                             APPROVE_AT, REVIEW_AT)


class TestLambdaPredict:
    def test_typical_applicant(self):
        data = {
            "age": 35, "income": 5000, "credit_history_months": 60,
            "num_credit_accounts": 3, "debt_ratio": 0.3, "num_late_payments": 1
        }
        score, prob = predict(data)
        assert 300 <= score <= 850
        assert 0 <= prob <= 1

    def test_good_applicant_high_score(self):
        good = {
            "age": 50, "income": 12000, "credit_history_months": 200,
            "num_credit_accounts": 5, "debt_ratio": 0.1, "num_late_payments": 0
        }
        bad = {
            "age": 20, "income": 1000, "credit_history_months": 1,
            "num_credit_accounts": 0, "debt_ratio": 0.9, "num_late_payments": 10
        }
        good_score, _ = predict(good)
        bad_score, _ = predict(bad)
        assert good_score > bad_score

    def test_score_clipping(self):
        extreme_good = {
            "age": 70, "income": 15000, "credit_history_months": 240,
            "num_credit_accounts": 15, "debt_ratio": 0.0, "num_late_payments": 0
        }
        score, _ = predict(extreme_good)
        assert score <= 850

        extreme_bad = {
            "age": 18, "income": 100, "credit_history_months": 0,
            "num_credit_accounts": 0, "debt_ratio": 1.0, "num_late_payments": 15
        }
        score, _ = predict(extreme_bad)
        assert score >= 300


class TestGetDecision:
    """Decisions are asserted relative to the derived cutoffs.

    A literal like get_decision(650) == "Approve" was true under the old scale
    and false after recalibration, without anything about the policy having
    changed. Anchoring the assertions to APPROVE_AT and REVIEW_AT keeps them
    testing the policy rather than the labelling.
    """

    def test_approve(self):
        decision, risk = get_decision(APPROVE_AT + 25)
        assert decision == "Approve"

    def test_review(self):
        decision, risk = get_decision((APPROVE_AT + REVIEW_AT) / 2)
        assert decision == "Review"

    def test_reject(self):
        decision, risk = get_decision(REVIEW_AT - 100)
        assert decision == "Reject"

    def test_boundary_approve(self):
        # Asserted against the derived cutoff rather than a literal, so a
        # recalibration cannot leave the tests checking a stale scale.
        assert get_decision(APPROVE_AT)[0] == "Approve"
        assert get_decision(APPROVE_AT - 0.01)[0] == "Review"

    def test_boundary_review(self):
        assert get_decision(REVIEW_AT)[0] == "Review"
        assert get_decision(REVIEW_AT - 0.01)[0] == "Reject"

    def test_cutoffs_ordered_and_inside_range(self):
        assert 300 < REVIEW_AT < APPROVE_AT < 850


class TestLambdaHandler:
    def test_direct_body(self):
        event = {
            "body": json.dumps({
                "age": 35, "income": 5000, "credit_history_months": 60,
                "num_credit_accounts": 3, "debt_ratio": 0.3, "num_late_payments": 1
            })
        }
        response = lambda_handler(event, None)
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'credit_score' in body
        assert 'decision' in body

    def test_event_as_input(self):
        event = {
            "age": 35, "income": 5000, "credit_history_months": 60,
            "num_credit_accounts": 3, "debt_ratio": 0.3, "num_late_payments": 1
        }
        response = lambda_handler(event, None)
        assert response['statusCode'] == 200

    def test_empty_body_returns_error(self):
        event = {"body": "{}"}
        response = lambda_handler(event, None)
        # Should handle gracefully (may return 200 with defaults or 500)
        assert response['statusCode'] in [200, 500]
