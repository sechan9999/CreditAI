"""The one definition of the score scale and the decision policy.

Before this module existed the scale was written out in four places -- the web
page, the Lambda handler, `train_scoring_model.predict_score`'s default
arguments, and `create_scorecard_policy`'s -- and they had drifted. The page was
scoring the same applicant 122 points below the trained model, which is how a
demo ends up declining someone the model approves.

Two things are defined here and nowhere else.

**The scale.** A scorecard maps log-odds onto points with an affine transform:

    score = SCORE_OFFSET + SCORE_FACTOR * ln(odds)

and `ln(odds)` is the model's linear predictor, which is why a scorecard is
linear in its features. Two constants fix the transform: how many points double
the odds (`PDO`), and what odds the anchor score sits at (`BASE_SCORE` at
`BASE_ODDS`).

**The policy**, stated in odds rather than points. This is the part that makes a
recalibration safe. If the cutoffs were points, changing the scale would silently
move every decision; because they are odds, the point cutoffs are derived and move
with the scale, so a rescaling is a relabelling and nothing more.

`lambda_function.py` deliberately keeps its own copy of these numbers, because it
ships as a self-contained deployment artifact and cannot import from `src/`. The
web page keeps a copy for the same reason. `tests/test_scale_consistency.py`
asserts all three agree, so a copy cannot drift again without a test failing.
"""

import math

# --- Scale ---
# Recalibrated from PDO 20 with 600 at 5:1. On that scale the 7,000-applicant
# population spanned only 300-658, leaving the top third of a declared 300-850
# range unused, and the anchor needed a sentence to explain. PDO 40 with 600 at
# even money spreads the same population across 300-809 and states itself: 600
# points means an even chance of being good.
BASE_SCORE = 600
PDO = 40          # points to double the odds
BASE_ODDS = 1     # odds of being good at BASE_SCORE -- even money

SCORE_MIN = 300
SCORE_MAX = 850

SCORE_FACTOR = PDO / math.log(2)
SCORE_OFFSET = BASE_SCORE - SCORE_FACTOR * math.log(BASE_ODDS)

# --- Policy ---
# The bars the previous scale's 600 and 550 represented, preserved exactly.
# Derived from the previous scale rather than transcribed. Writing the review bar
# out as a literal is how a rounding slip becomes a policy change: a hand-computed
# 0.8839527 against the exact 0.8838834764831849 moves the boundary just enough to
# flip an applicant sitting on it. Deriving it makes that impossible.
#
# These are the bars the previous scale's 600 and 550 points represented. Replacing
# them with chosen round odds is a policy decision and should be made as one.
_PREV_PDO, _PREV_BASE_SCORE, _PREV_BASE_ODDS = 20, 600, 5
_prev_factor = _PREV_PDO / math.log(2)
_prev_offset = _PREV_BASE_SCORE - _prev_factor * math.log(_PREV_BASE_ODDS)

APPROVE_ODDS = _PREV_BASE_ODDS                                # the old 600 bar, p(good) = 83.3%
REVIEW_ODDS = math.exp((550 - _prev_offset) / _prev_factor)    # the old 550 bar, p(good) = 46.9%

APPROVE_AT = SCORE_OFFSET + SCORE_FACTOR * math.log(APPROVE_ODDS)
REVIEW_AT = SCORE_OFFSET + SCORE_FACTOR * math.log(REVIEW_ODDS)


def score_from_odds(odds):
    """Points for a given odds of being good, clipped to the declared range."""
    return max(SCORE_MIN, min(SCORE_MAX, SCORE_OFFSET + SCORE_FACTOR * math.log(odds + 1e-10)))


def score_from_prob(prob):
    """Points for a given probability of being good."""
    return score_from_odds(prob / (1 - prob + 1e-10))


def decision(score):
    """(decision, risk label) for a score, from the derived cutoffs."""
    if score >= APPROVE_AT:
        return "Approve", "Minimal/Low Risk"
    if score >= REVIEW_AT:
        return "Review", "Medium Risk"
    return "Reject", "High/Very High Risk"


def describe():
    """One-screen summary, for report headers."""
    return (
        f"Scale:  {PDO} points double the odds; {BASE_SCORE} points sits at "
        f"{BASE_ODDS}:1 odds (p(good) = {BASE_ODDS / (1 + BASE_ODDS):.1%}).\n"
        f"        score = {SCORE_OFFSET:.2f} + {SCORE_FACTOR:.4f} * ln(odds), "
        f"clipped to {SCORE_MIN}-{SCORE_MAX}.\n"
        f"Policy: approve at {APPROVE_ODDS}:1 odds and above "
        f"(p >= {APPROVE_ODDS / (1 + APPROVE_ODDS):.1%}), which is {APPROVE_AT:.0f} points.\n"
        f"        review from {REVIEW_ODDS:.4f}:1 "
        f"(p >= {REVIEW_ODDS / (1 + REVIEW_ODDS):.1%}), which is {REVIEW_AT:.0f} points.\n"
        f"        decline below that.\n"
    )


if __name__ == "__main__":
    print(describe())
