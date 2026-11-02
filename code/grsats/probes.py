from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grsats.config import L2_GRID


def train_probe(X_train, y_train, X_dev, y_dev):
    """Fit a probe for each l2 strength and keep the one with the lowest dev NLL."""
    best = None
    for lam in L2_GRID:
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1 / lam, max_iter=5000))
        probe.fit(X_train, y_train)
        nll = log_loss(y_dev, probe.predict_proba(X_dev)[:, 1])
        if best is None or nll < best[1]:
            best = (lam, nll, probe)
    return best
