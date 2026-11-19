"""Claim-relative time-series evaluation primitives. NumPy + standard library.

All time intervals are half-open. Metrics do not silently concatenate trials.
No model training and no external-service calls.
"""
import hashlib
import itertools
import math

import numpy as np


def validate(scores, labels):
    s = np.asarray(scores, dtype=float)
    raw = np.asarray(labels)
    if s.ndim != 1 or raw.ndim != 1 or len(s) != len(raw) or not len(s):
        raise ValueError('scores and labels must be aligned nonempty one-dimensional arrays')
    if not np.all(np.isfinite(s)) or not np.all(np.isin(raw, [0, 1])):
        raise ValueError('finite scores and binary labels required')
    return s, raw.astype(bool)


def events(vector):
    y = np.asarray(vector, dtype=bool)
    edge = np.diff(np.r_[False, y, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def event_counts(y, pred):
    """Maximum-cardinality one-to-one overlap matching of disjoint intervals.

    For two ordered, internally disjoint interval lists, match the earliest
    overlapping pair. Choosing these earliest ends cannot reduce later matches.
    """
    truth, alarms = events(y), events(pred)
    i = j = matched = 0
    while i < len(truth) and j < len(alarms):
        a, b = truth[i]
        c, d = alarms[j]
        if b <= c:
            i += 1
        elif d <= a:
            j += 1
        else:
            matched += 1
            i += 1
            j += 1
    return matched, len(alarms)-matched, len(truth)-matched


def metrics(labels, pred):
    y, pred = np.asarray(labels, dtype=bool), np.asarray(pred, dtype=bool)
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError('unaligned predictions')
    P, N = int(y.sum()), int((~y).sum())
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    segs = events(y)
    touched = [(a,b) for a,b in segs if pred[a:b].any()]
    covered = sum(b-a for a,b in touched)
    etp, efp, efn = event_counts(y, pred)
    false_alarm_events = len(events(pred & ~y))
    f1 = 2*tp/(P+tp+fp) if P+tp+fp else 0.0
    return dict(n_points=len(y), n_positive=P, n_negative=N, tp=tp, fp=fp, fn=P-tp, tn=N-fp,
                p=P/len(y), f1=f1, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/P if P else None, fpr=fp/N if N else None,
                rho=fp/tp if tp else None, c=covered/P if P else None,
                rin=tp/covered if covered else None, segr=len(touched)/len(segs) if segs else None,
                n_true_events=len(segs), n_predicted_events=len(events(pred)),
                n_touched_events=len(touched), covered_positive_points=covered,
                event_tp=etp, event_fp=efp, event_fn=efn,
                event_f1=2*etp/(2*etp+efp+efn) if 2*etp+efp+efn else None,
                false_alarm_spells=false_alarm_events,
                false_alarm_spells_per_10000_normal=10000*false_alarm_events/N if N else None)


def ranking(scores, labels):
    s, y = validate(scores, labels)
    P, N = int(y.sum()), int((~y).sum())
    if not P or not N:
        return dict(auc=None, ap=None, oracle_f1=None, oracle_threshold=None)
    order = np.argsort(-s, kind='stable')
    ss, yy = s[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
    tp = np.cumsum(yy)[ends]
    fp = ends+1-tp
    auc = float(np.trapezoid(np.r_[0.,tp/P], np.r_[0.,fp/N]))
    ap = float((np.diff(np.r_[0.,tp/P])*tp/(ends+1)).sum())
    fs = 2*tp/(P+ends+1)
    best = int(np.argmax(fs))
    return dict(auc=auc, ap=ap, oracle_f1=float(fs[best]), oracle_threshold=float(ss[ends[best]]))


def calibrate(val_scores, val_labels, policy):
    s, y = validate(val_scores, val_labels)
    if policy == 'val_f1':
        r = ranking(s,y)
        if r['oracle_threshold'] is None:
            raise ValueError('val_f1 needs validation normal and positive examples')
        return r['oracle_threshold'], '>='
    alphas = {'normal_005':.005, 'normal_010':.01, 'normal_050':.05}
    if policy not in alphas:
        raise ValueError('unknown policy')
    normal = np.sort(s[~y])
    if not len(normal):
        raise ValueError('normal calibration observations required')
    index = min(len(normal)-1, max(0, math.ceil((1-alphas[policy])*len(normal))-1))
    return float(normal[index]), '>'


def predict(scores, threshold, comparison):
    if comparison == '>':
        return np.asarray(scores) > threshold
    if comparison == '>=':
        return np.asarray(scores) >= threshold
    raise ValueError('unknown comparison')


def preserves_observed_order(values, transform):
    """Check finite-precision admissibility, not just symbolic monotonicity."""
    unique=np.unique(np.asarray(values,dtype=float))
    transformed=transform(unique)
    return bool(np.all(np.isfinite(transformed)) and np.all(np.diff(transformed)>0))


def channel_f1(p, c, rin, fpr):
    if not 0 < p < 1:
        raise ValueError('three-channel formula requires both classes')
    if any(v is None or not 0 <= v <= 1 for v in [c,rin,fpr]):
        raise ValueError('channel arguments outside [0,1]')
    return 2*p*c*rin/(p*c*rin+(1-p)*fpr+p)


def allocation(a,b):
    """Signed symmetric allocation of F1(a)-F1(b), same prevalence only."""
    if abs(a['p']-b['p']) > 1e-15:
        raise ValueError('different prevalence: not an observed-gap three-channel allocation')
    av = [a['c'], a['rin'] or 0., a['fpr']]
    bv = [b['c'], b['rin'] or 0., b['fpr']]
    parts = np.zeros(3)
    for order in itertools.permutations(range(3)):
        state = list(bv)
        for k in order:
            before = channel_f1(a['p'],*state)
            state[k] = av[k]
            parts[k] += channel_f1(a['p'],*state)-before
    parts /= 6
    error = abs(float(parts.sum())-(a['f1']-b['f1']))
    return dict(coverage_allocation=float(parts[0]), recall_allocation=float(parts[1]),
                fpr_allocation=float(parts[2]), allocation_error=error)


def sign_p(diffs):
    d = np.asarray(diffs,dtype=float)
    n = int((d != 0).sum())
    wins, losses = int((d>0).sum()), int((d<0).sum())
    k = min(wins,losses)
    p = min(1., 2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.
    return dict(sign_p=p, nonzero_pairs=n, wins=wins, losses=losses, ties=len(d)-n)


def holm(pvalues):
    p = np.asarray(pvalues,dtype=float)
    order = np.argsort(p,kind='stable')
    adj = np.minimum(1.,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    result = np.empty(len(p));result[order]=adj
    return result.tolist()


def paired_summary(diffs, identifier, B=10000, K=4, alpha=.05):
    d = np.asarray(diffs,dtype=float)
    seed = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8],'little')
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0,len(d),size=(B,len(d)))].mean(axis=1)
    # For independent Xi in [-1,1], Hoeffding two-sided error <=2 exp(-n e^2/2).
    radius = math.sqrt(2*math.log(2*K/alpha)/len(d))
    return dict(n_pairs=len(d), difference_mean=float(d.mean()),
                difference_sd=float(d.std(ddof=1)) if len(d)>1 else None,
                boot_lo=float(np.quantile(boot,.025)), boot_hi=float(np.quantile(boot,.975)),
                simultaneous_hoeffding_lo=max(-1.,float(d.mean()-radius)),
                simultaneous_hoeffding_hi=min(1.,float(d.mean()+radius)),
                **sign_p(d))


def claim_assessment(record):
    """V2 minimal structured-evidence checks, not complete scientific adjudication.

    Confirmed contradictions take precedence over missing information, without
    hiding the latter. Unsupported inferences require review; absence of a
    contradiction in four implemented predicates is never a utility certificate.
    The frozen v1 remains in protocol_validation/frozen_engine/diagnostics.py.
    """
    required = {'target','metric','unit','calibration','snapshot_verified','causal_online',
                 'timeline_causal','event_boundaries_known','independence_verified'}
    absent = sorted(k for k in required if k not in record or record[k] is None)
    contradictions = []
    if record.get('metric')=='PA-F1' and record.get('claim_strict') is True:
        contradictions.append('point_adjustment_is_not_strict_pointwise')
    if record.get('target')=='deployment' and record.get('calibration')=='test_oracle':
        contradictions.append('test_label_oracle_cannot_set_a_deployment_threshold')
    if record.get('causal_online') is True and record.get('timeline_causal') is False:
        contradictions.append('future_context_is_not_causal_online_evidence')
    if record.get('snapshot_hash_mismatch') is True:
        contradictions.append('snapshot_hash_mismatch')
    limitations = []
    if record.get('snapshot_verified') is False:
        limitations.append('snapshot_identity_unverified')
    if record.get('target') in ['event','deployment'] and record.get('event_boundaries_known') is False:
        limitations.append('operational_event_boundaries_unavailable')
    if record.get('target') in ['generalization','deployment'] and record.get('independence_verified') is False:
        limitations.append('data_independence_unverified')
    unsupported = record.get('requested_inferences', [])
    if not isinstance(unsupported, list) or not all(isinstance(x,str) and x.strip() for x in unsupported):
        raise ValueError('requested_inferences must be a list of nonempty inference names')
    unsupported = set(unsupported)
    if record.get('equivalence_asserted') is True:
        unsupported.add('equivalence')
    if record.get('target') == 'generalization':
        unsupported.add('population_generalization')
    if record.get('target') == 'deployment':
        unsupported.add('operational_utility')
    if record.get('target') not in [None,'score','pipeline','event','deployment','generalization']:
        limitations.append('unknown_claim_target')
    for key in ['snapshot_verified','causal_online','timeline_causal','event_boundaries_known','independence_verified']:
        if key in record and record[key] is not None and type(record[key]) is not bool:
            limitations.append('invalid_boolean:'+key)
    missing_reasons = ['missing:'+x for x in absent]
    unsupported_reasons = ['unsupported_inference:'+x for x in sorted(unsupported)]
    reasons = contradictions+missing_reasons+limitations+unsupported_reasons
    status = ('contradicted' if contradictions else 'not_evaluable' if reasons else 'within_scope')
    return {'status':status, 'reasons':reasons or ['listed_gates_only_not_a_quality_certificate'],
            'contradictions':contradictions, 'missing_fields':absent,
            'limitations':limitations, 'unsupported_inferences':sorted(unsupported),
            'engine_version':'structured_checks_v2', 'certificate':False,
            'coverage':'four contradiction predicates; documentary, inferential and utility review not automated'}
