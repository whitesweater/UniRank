"""Prespecified paired diagnostics; uncertainty is conditional on frozen windows/templates."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ['is_click', 'is_conversion']
ROUTES = ['full', 'local_off', 'template_raw', 'template_rms',
          'donor_rms_20260925', 'donor_rms_20260926', 'donor_rms_20260927']
CONTRASTS = [('full', 'template_rms'), ('local_off', 'template_raw'),
             ('template_raw', 'template_rms'), ('template_rms', 'full'),
             ('full', 'local_off'), ('full', 'template_raw'), ('full', 'donor_mean')]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def loss(y, p):
    assert np.isin(y, [0, 1]).all() and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    p = np.clip(p.astype(np.float64), 1e-7, 1-1e-7)
    return -y*np.log(p) - (1-y)*np.log1p(-p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    source = args.results.resolve()
    completion = json.loads((source/'complete.json').read_text())
    assert completion['completed'] and completion['all_gates_passed']
    for name, expected in completion['files_sha256'].items():
        assert sha(source/name) == expected, name
    output = source/'analysis'
    output.mkdir(exist_ok=False)
    plan = json.loads((source/'plan.json').read_text())
    frame = pd.read_csv(source/'predictions.csv', float_precision='round_trip')
    meta = pd.read_csv(source/'meta.csv')
    assert len(frame) == 131072 == plan['rows']
    assert frame.row_index.tolist() == list(range(len(frame)))
    assert np.array_equal(frame.user_id, meta.user_id)
    for label in LABELS:
        assert np.array_equal(frame[label], meta[label])
    assert frame['eligible'].dtype == bool and meta['eligible'].dtype == bool
    eligible = frame['eligible'].to_numpy()
    assert np.array_equal(eligible, meta['eligible'])
    assert int(eligible.sum()) == plan['eligible_rows'] == 131070
    frame = frame.loc[eligible].copy()
    user_ids, inverse = np.unique(frame.user_id.to_numpy(), return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    assert len(user_ids) == plan['eligible_users']
    rng = np.random.default_rng(20260925)
    draws = rng.multinomial(len(user_ids), np.full(len(user_ids), 1/len(user_ids)), size=2000)
    denominators = draws @ counts
    losses, clusters, metric_rows = {}, {}, []
    for route in ROUTES:
        probabilities = frame[[f'{label}_{route}' for label in LABELS]].to_numpy(dtype=np.float64)
        values = loss(frame[LABELS].to_numpy(), probabilities)
        values = np.column_stack([values, values.mean(axis=1)])
        losses[route] = values
        clusters[route] = np.column_stack([np.bincount(inverse, weights=values[:,j]) for j in range(len(LABELS)+1)])
        for j, label in enumerate(LABELS+['label_macro']):
            auc = roc_auc_score(frame[label], probabilities[:,j]) if j < len(LABELS) and frame[label].nunique() == 2 else None
            metric_rows.append({'route':route, 'label':label, 'rows':len(frame), 'users':len(user_ids),
                'positive_rows':int(frame[label].sum()) if j<len(LABELS) else None,
                'mean_bce':float(values[:,j].mean()), 'diagnostic_auc':auc,
                'user_balanced_bce':float((clusters[route][:,j]/counts).mean())})
    losses['donor_mean'] = np.mean([losses[r] for r in ROUTES if r.startswith('donor_')], axis=0)
    clusters['donor_mean'] = np.mean([clusters[r] for r in ROUTES if r.startswith('donor_')], axis=0)
    contrast_rows = []
    for a,b in CONTRASTS:
        cluster_delta = clusters[a]-clusters[b]
        bootstrap = (draws @ cluster_delta) / denominators[:,None]
        balanced = draws @ (cluster_delta/counts[:,None]) / len(user_ids)
        for j,label in enumerate(LABELS+['label_macro']):
            lo,hi = np.quantile(bootstrap[:,j],[.025,.975])
            blo,bhi = np.quantile(balanced[:,j],[.025,.975])
            contrast_rows.append({'contrast':f'{a}_minus_{b}', 'label':label,
                'bce_difference':float(cluster_delta[:,j].sum()/counts.sum()),
                'ci_low':float(lo),'ci_high':float(hi),
                'user_balanced_difference':float((cluster_delta[:,j]/counts).mean()),
                'user_balanced_ci_low':float(blo),'user_balanced_ci_high':float(bhi),
                'primary':a=='full' and b=='template_rms' and label=='is_click'})
    pd.DataFrame(metric_rows).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(contrast_rows).to_csv(output/'contrasts.csv',index=False)
    pd.DataFrame({'user_id':user_ids,'rows':counts.astype(int)}).to_csv(output/'user_clusters.csv',index=False)
    cluster_rows=[]
    for r in clusters:
        for j,label in enumerate(LABELS+['label_macro']):
            cluster_rows.append(pd.DataFrame({'user_id':user_ids,'rows':counts.astype(int),
                'route':r,'label':label,'loss_sum':clusters[r][:,j]}))
    pd.concat(cluster_rows).to_csv(output/'cluster_loss_sums.csv',index=False)
    diag=pd.read_csv(source/'row_diagnostics.csv')
    summary=[]
    cols=['attention_tv_to_full','bias_cosine_to_full','actual_bias_centered_rms',
          'full_bias_centered_rms','attention_entropy','top_key_changed_from_full']
    for route, group in diag.groupby('route',sort=False):
        group = group.loc[group['eligible']].copy()
        assert len(group)==len(frame)
        for col in cols:
            x=group[col].to_numpy(dtype=float)
            summary.append({'route':route,'quantity':col,'mean':float(np.nanmean(x)),
                'median':float(np.nanmedian(x)),'p10':float(np.nanquantile(x,.1)),
                'p90':float(np.nanquantile(x,.9)),'finite_rows':int(np.isfinite(x).sum())})
    pd.DataFrame(summary).to_csv(output/'attention_summary.csv',index=False)
    full = np.load(source/'attention_full.npz')
    delta = full['delta'][eligible,0].astype(np.float64)
    mask = full['native_mask'][eligible,0].astype(bool)
    lengths = mask.sum(axis=1,keepdims=True)
    assert (lengths >= 2).all()
    row_mean = np.where(mask,delta,0).sum(axis=1,keepdims=True)/lengths
    centered = np.where(mask,delta-row_mean,0)
    rms = np.sqrt(np.sum(centered**2,axis=1)/lengths[:,0])
    normalized = np.divide(centered,rms[:,None],out=np.zeros_like(centered),where=rms[:,None]>0)
    valid_counts = mask.sum(axis=0)
    q10=np.full(100,np.nan);q90=np.full(100,np.nan); norm_mean=np.full(100,np.nan)
    for j in range(100):
        if valid_counts[j]:
            values=normalized[mask[:,j],j]
            q10[j],q90[j]=np.quantile(values,[.1,.9]);norm_mean[j]=values.mean()
    profile=pd.DataFrame({'position':np.arange(1,101),'valid_rows':valid_counts,'normalized_delta_mean':norm_mean,
        'normalized_delta_p10':q10,'normalized_delta_p90':q90,
        'native_weight_mean':full['weight_before'][eligible,0].mean(0),
        'full_weight_mean':full['weight_after'][eligible,0].mean(0)})
    for route in ['template_raw','template_rms']:
        arr=np.load(source/f'attention_{route}.npz')
        profile[f'{route}_weight_mean']=arr['weight'][eligible,0].mean(0)
    profile.to_csv(output/'position_profile.csv',index=False)
    record={'status':'complete','forward_rows':131072,'excluded_rows':2,'rows':len(frame),'unique_users':len(user_ids),
        'bootstrap_draws':2000,'bootstrap_seed':20260925,
        'uncertainty':'Fixed-window user-cluster bootstrap, conditional on trained checkpoint and estimated cross-fit templates; no training or population uncertainty claim',
        'primary_endpoint':next(r for r in contrast_rows if r['primary']),
        'user_clusters':{'min':int(counts.min()),'median':float(np.median(counts)),'max':int(counts.max()),
            'largest_user_fraction':float(counts.max()/counts.sum()),'effective_users_by_row_concentration':float(counts.sum()**2/(counts@counts))},
        'full_rms':{'mean':float(rms.mean()),'p10':float(np.quantile(rms,.1)),'p90':float(np.quantile(rms,.9))},
        'input_predictions_sha256':sha(source/'predictions.csv'),'plan_sha256':sha(source/'plan.json'),
        'script_sha256':sha(Path(__file__))}
    (output/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
