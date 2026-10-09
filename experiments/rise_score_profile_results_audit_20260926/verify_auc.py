"""Recompute existing AUC point estimates from raw predictions; no bootstrap/inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_profiles import ROOT, CASES, read, sha

LABELS = {'RankMixer': ['is_click', 'is_follow', 'is_like', 'is_comment', 'is_forward', 'long_view'],
          'HiFormer': ['is_click', 'is_conversion']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for model, seed, relative in CASES:
        source = ROOT / relative
        complete = read(source / 'complete.json')
        assert complete['completed'] and complete['all_gates_passed']
        prediction_sha = sha(source / 'predictions.csv')
        assert prediction_sha == complete['files_sha256']['predictions.csv']
        frame = pd.read_csv(source / 'predictions.csv', float_precision='round_trip', low_memory=False)
        if 'eligible' in frame:
            assert frame.eligible.dtype == bool
            frame = frame.loc[frame.eligible]
        metrics = pd.read_csv(source / 'analysis/metrics.csv')
        routes = {}
        for route in ('full', 'template_raw', 'template_rms'):
            values, errors = {}, {}
            for label in LABELS[model]:
                y, p = frame[label].to_numpy(), frame[f'{label}_{route}'].to_numpy(dtype=np.float64)
                assert np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                value = float(roc_auc_score(y, p))
                stored = float(metrics.loc[(metrics.route == route) & (metrics.label == label), 'diagnostic_auc'].iloc[0])
                assert abs(value - stored) < 1e-12, (model, seed, route, label)
                values[label], errors[label] = value, value - stored
            routes[route] = {'click_auc': values['is_click'], 'macro_auc': float(np.mean(list(values.values()))),
                             'per_label_auc': values, 'per_label_error_vs_stored': errors}
        rows.append({'model': model, 'checkpoint_seed': seed, 'rows': len(frame), 'prediction_sha256': prediction_sha,
                     'routes': routes, 'max_abs_recomputation_error': max(abs(v) for r in routes.values() for v in r['per_label_error_vs_stored'].values())})
    result = {'status': 'raw_auc_recomputed_verified', 'rows': rows, 'script_sha256': sha(__file__),
              'scope': 'Point estimates from unchanged fixed-window predictions; macro is equal mean of original six/two labels. No AUC bootstrap, intervals, inference or model training.'}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'max_abs_error': max(r['max_abs_recomputation_error'] for r in rows),
                      'checkpoint_rows': len(rows), 'auc_values_checked': sum(len(v['per_label_auc']) for r in rows for v in r['routes'].values())}))


if __name__ == '__main__':
    main()
