"""Mask-preserving cross-part correction controls, independent of outcomes."""
import numpy as np

DONOR_SEEDS = (20260925, 20260926, 20260927)


def moments(values, masks):
    values, masks = np.asarray(values, dtype=np.float64), np.asarray(masks, dtype=bool)
    count = masks.sum(axis=1, keepdims=True)
    mean = np.divide(np.where(masks, values, 0).sum(axis=1,keepdims=True), count,
                     out=np.zeros((len(values),1)), where=count>0)
    squares = np.where(masks, (values-mean)**2, 0).sum(axis=1,keepdims=True)
    rms = np.sqrt(np.divide(squares, count, out=np.zeros_like(mean), where=count>0))
    return mean, rms


def mask_groups(masks):
    _, inverse = np.unique(np.packbits(np.asarray(masks,dtype=bool),axis=1),axis=0,return_inverse=True)
    return [np.flatnonzero(inverse==i) for i in range(inverse.max()+1)]


def eligibility(parts, masks):
    parts, masks = np.asarray(parts), np.asarray(masks,dtype=bool)
    peer_counts = np.zeros(len(parts),dtype=np.int64)
    for group in mask_groups(masks):
        for part in np.unique(parts[group]):
            own = group[parts[group]==part]
            peer_counts[own] = len(group)-len(own)
    eligible = (peer_counts>0) & (masks.sum(1)>=2)
    return eligible, peer_counts


def match_moments(source, recipient, masks, eligible):
    source, recipient = np.asarray(source,dtype=np.float64), np.asarray(recipient,dtype=np.float64)
    masks, eligible = np.asarray(masks,dtype=bool), np.asarray(eligible,dtype=bool)
    sm,sr = moments(source,masks); rm,rr = moments(recipient,masks)
    if np.any(eligible[:,None] & (sr==0) & (rr>0)):
        raise ValueError('Flat donor cannot match a nonflat eligible recipient')
    scale = np.divide(rr,sr,out=np.zeros_like(rr),where=sr>0)
    result = recipient.copy()
    active = masks & eligible[:,None]
    result[active] = ((source-sm)*scale+rm)[active]
    return result


def crosspart_templates(delta, parts, masks):
    delta,parts,masks = np.asarray(delta),np.asarray(parts),np.asarray(masks,dtype=bool)
    if delta.ndim!=2 or masks.shape!=delta.shape or len(parts)!=len(delta) or not np.isfinite(delta).all():
        raise ValueError('Misaligned or nonfinite correction inputs')
    eligible, peer_counts = eligibility(parts,masks)
    templates = delta.astype(np.float64).copy()
    receipts=[]
    for group in mask_groups(masks):
        allowed=masks[group[0]]
        if allowed.sum()<2: continue
        total=delta[group].sum(axis=0,dtype=np.float64)
        for part in np.unique(parts[group]):
            own=group[parts[group]==part]
            peers=len(group)-len(own)
            if peers==0: continue
            template=(total-delta[own].sum(axis=0,dtype=np.float64))/peers
            templates[np.ix_(own,np.flatnonzero(allowed))]=template[allowed]
            receipts.append({'recipient_part':int(part),'native_length':int(allowed.sum()),
                'native_mask_sha256':__import__('hashlib').sha256(allowed.tobytes()).hexdigest(),
                'recipient_rows':len(own),'peer_rows':peers,'excluded_own_part':True})
    tm,_=moments(templates,masks); rm,_=moments(delta,masks)
    raw=delta.astype(np.float64).copy(); active=masks & eligible[:,None]
    raw[active]=(templates-tm+rm)[active]
    return raw,match_moments(templates,delta,masks,eligible),receipts


def crosspart_donor_indices(parts,masks,seed):
    parts,masks=np.asarray(parts),np.asarray(masks,dtype=bool)
    eligible,_=eligibility(parts,masks)
    rng=np.random.default_rng(seed)
    mapping=np.arange(len(parts),dtype=np.int64)
    for group in mask_groups(masks):
        for part in np.unique(parts[group]):
            own=group[(parts[group]==part) & eligible[group]]
            if len(own)==0: continue
            peers=group[parts[group]!=part]
            mapping[own]=rng.choice(peers,size=len(own),replace=True)
    if np.any(parts[mapping[eligible]]==parts[eligible]) or not np.array_equal(masks[mapping[eligible]],masks[eligible]):
        raise ValueError('Donor violated cross-part same-mask requirement')
    return mapping


def row_diagnostics(full_delta,actual_delta,full_weight,actual_weight,masks,eligible):
    masks=np.asarray(masks,dtype=bool); eligible=np.asarray(eligible,dtype=bool)
    fm,fr=moments(full_delta,masks);am,ar=moments(actual_delta,masks)
    count=masks.sum(1)
    numerator=np.divide(np.where(masks,(full_delta-fm)*(actual_delta-am),0).sum(1),count,
                        out=np.zeros(len(count)),where=count>0)
    denominator=fr[:,0]*ar[:,0]
    cosine=np.divide(numerator,denominator,out=np.full(len(count),np.nan),where=denominator>0)
    result={'actual_bias_mean':am[:,0],'actual_bias_centered_rms':ar[:,0],
        'full_bias_centered_rms':fr[:,0],'bias_cosine_to_full':cosine,
        'attention_tv_to_full':np.abs(full_weight-actual_weight).sum(1)/2,
        'top_key_changed_from_full':(full_weight.argmax(1)!=actual_weight.argmax(1)).astype(float),
        'attention_entropy':-(actual_weight*np.log(np.maximum(actual_weight,1e-30))).sum(1)}
    for values in result.values(): values[~eligible]=np.nan
    return result
