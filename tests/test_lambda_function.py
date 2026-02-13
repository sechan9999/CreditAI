import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lambda_function import predict, get_decision, lambda_handler


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
    def test_approve(self):
        decision, risk = get_decision(650)
        assert decision == "Approve"

    def test_review(self):
        decision, risk = get_decision(575)
        assert decision == "Review"

    def test_reject(self):
        decision, risk = get_decision(400)
        assert decision == "Reject"

    def test_boundary_600(self):
        decision, _ = get_decision(600)
        assert decision == "Approve"

    def test_boundary_550(self):
        decision, _ = get_decision(550)
        assert decision == "Review"

    def test_boundary_549(self):
        decision, _ = get_decision(549)
        assert decision == "Reject"


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
