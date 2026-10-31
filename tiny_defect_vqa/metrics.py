"""Typed-answer, grounding and selective-prediction metrics."""
import itertools
import numpy as np
from sklearn.metrics import f1_score


def measure(probability, truth, intent, accepted):
    """Compute metrics with fixed 32-answer F1 and observed gate coverage."""
    probability, truth, intent, accepted = map(np.asarray, [probability, truth, intent, accepted])
    predicted = probability.argmax(1)
    confidence = probability.max(1)
    correct = predicted == truth
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        selected = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if selected.any():
            ece += selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    count = max(1, int(np.ceil(.8*len(truth))))
    order = np.argsort(-confidence, kind="stable")[:count]
    errors = ~correct
    return {"accuracy": float(correct.mean()), "macro_f1": float(f1_score(truth, predicted, labels=np.arange(32), average="macro", zero_division=0)),
            "per_intent_accuracy": {str(i): float(correct[intent == i].mean()) for i in np.unique(intent)},
            "ece15": float(ece), "coverage": float(accepted.mean()),
            "selective_risk": float(errors[accepted].mean()) if accepted.any() else None,
            "risk_at_80pct_test_ranking": float(errors[order].mean()),
            "error_capture_rate": float((errors & ~accepted).sum()/errors.sum()) if errors.any() else 0.}


def paired_permutation(left, right):
    """Exact two-sided paired sign test over all 2**N assignments, effect in pp."""
    delta = np.asarray(left, dtype=float)-np.asarray(right, dtype=float)
    observed = abs(delta.mean())
    values = [abs((delta*np.asarray(sign)).mean()) for sign in itertools.product([-1, 1], repeat=len(delta))]
    return {"mean_effect_percentage_points": float(100*delta.mean()), "pvalue": float(np.mean(np.asarray(values) >= observed-1e-12)), "pairs": len(delta)}
