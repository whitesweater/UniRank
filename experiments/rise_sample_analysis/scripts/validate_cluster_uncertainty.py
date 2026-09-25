"""Validate linearized paired-AUC uncertainty against exact user resampling.

This is a synthetic numerical validation, never a recommendation experiment.
All predictions are fixed before resampling. Bootstrap weights retain all rows
of each selected user and are shared by the two models.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from statistical_metrics import paired_auc_difference


def main():
    output = Path(__file__).resolve().parents[1] / "results/cluster_uncertainty_validation.json"
    scenarios = []
    for seed, strength in ((101, 0.0), (102, 1.5), (103, 3.0)):
        rng = np.random.default_rng(seed)
        clusters = 600
        sizes = rng.integers(5, 25, size=clusters)
        users = np.repeat(np.arange(clusters), sizes)
        user_effect = rng.normal(size=clusters)[users] * strength
        x = rng.normal(size=len(users))
        y = rng.binomial(1, expit(-1.0 + x + user_effect))
        before = expit(0.7 * x + 0.3 * user_effect + rng.normal(size=len(users)) * 0.8)
        after = expit(0.9 * x + 0.6 * user_effect + rng.normal(size=len(users)) * 0.5)
        analytic, _, _ = paired_auc_difference(y, before, after, users)
        differences = []
        for _ in range(500):
            counts = rng.multinomial(clusters, np.full(clusters, 1 / clusters))
            weights = counts[users]
            differences.append(roc_auc_score(y, after, sample_weight=weights)
                               - roc_auc_score(y, before, sample_weight=weights))
        bootstrap_se = float(np.std(differences, ddof=1))
        ratio = analytic["cluster_se"] / bootstrap_se
        if not 0.80 < ratio < 1.25:
            raise AssertionError(f"Large linearization/bootstrap discrepancy: seed={seed}, ratio={ratio}")
        scenarios.append({"seed": seed, "synthetic_user_strength": strength,
                          "users": clusters, "rows": len(users), "positives": int(y.sum()),
                          "auc_effect": analytic["estimate"], "linearized_cluster_se": analytic["cluster_se"],
                          "exact_user_bootstrap_replicates": 500, "bootstrap_se": bootstrap_se,
                          "linearized_over_bootstrap_se": ratio,
                          "bootstrap_percentile_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
                          "linearized_normal_interval": [analytic["ci_lower"], analytic["ci_upper"]]})
    output.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_validation_only": True, "scenarios": scenarios,
        "scope": "Finite-sample numerical agreement in these generated scenarios, not proof of universal coverage or of empirical user independence.",
        "passed": True}, indent=2) + "\n")
    print(output)
    print(json.dumps(scenarios, indent=2))


if __name__ == "__main__":
    main()
