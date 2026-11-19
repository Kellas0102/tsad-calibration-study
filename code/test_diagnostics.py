import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

from diagnostics import (allocation, calibrate, channel_f1, claim_assessment,
                         event_counts, events, holm, metrics, predict, ranking, sign_p,
                         preserves_observed_order)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/affiliation-metrics-py'))
from affiliation.metrics import pr_from_events


class SpecificationTests(unittest.TestCase):
    def test_unequal_lengths_not_segment_fraction(self):
        y=np.r_[np.ones(9),0,1,0];pred=np.zeros(len(y),dtype=bool);pred[0]=True
        m=metrics(y,pred)
        self.assertEqual(m['segr'],.5);self.assertEqual(m['c'],.9)
        self.assertAlmostEqual(m['recall'],m['c']*m['rin'])

    def test_saturation_does_not_imply_high_tpr_or_false_alarms(self):
        y=np.r_[np.ones(9),0,1,0];pred=np.zeros(len(y),dtype=bool);pred[[0,10]]=True
        m=metrics(y,pred)
        self.assertEqual(m['segr'],1);self.assertEqual(m['recall'],.2)
        self.assertEqual(m['rho'],0)

    def test_undefined_conditional_recall(self):
        m=metrics([0,1,1],[0,0,0])
        self.assertIsNone(m['rin']);self.assertEqual(m['c'],0)
        self.assertEqual(channel_f1(m['p'],m['c'],0,m['fpr']),m['f1'])

    def test_no_events_not_perfect_recall(self):
        m=metrics([0,0,0],[0,0,0])
        self.assertIsNone(m['segr']);self.assertIsNone(m['c']);self.assertIsNone(m['event_f1'])

    def test_event_f1_not_pointwise_f1(self):
        m=metrics([0,1,1,1,0],[0,1,0,0,0])
        self.assertEqual(m['f1'],.5);self.assertEqual(m['event_f1'],1)

    def test_duplicate_alarm_is_penalized(self):
        m=metrics([0,1,1,1,1,1,0],[0,1,0,0,0,1,0])
        self.assertEqual(m['event_tp'],1);self.assertEqual(m['event_fp'],1)

    def test_event_matching_bruteforce(self):
        def brute(truth,alarm):
            edges=[[j for j,(c,d) in enumerate(alarm) if max(a,c)<min(b,d)] for a,b in truth]
            def visit(i,used):
                if i==len(edges):return 0
                return max([visit(i+1,used)]+[1+visit(i+1,used|{j}) for j in edges[i] if j not in used])
            return visit(0,set())
        vectors=list(itertools.product([0,1],repeat=6))
        for y in vectors:
            for p in vectors:
                self.assertEqual(event_counts(y,p)[0],brute(events(y),events(p)))

    def test_tied_auc_and_ap(self):
        r=ranking([1,1,1,1],[0,1,0,1])
        self.assertEqual(r['auc'],.5);self.assertEqual(r['ap'],.5)
        self.assertAlmostEqual(r['oracle_f1'],2/3)

    def test_calibration_invariance(self):
        s=np.array([0,0,.1,.4,.8,1.,2.]);y=np.array([0,0,0,0,1,0,1])
        t=np.array([-.2,0,.1,.5,1.,3.])
        for policy in ['val_f1','normal_005','normal_010','normal_050']:
            tau,cmp=calibrate(s,y,policy);base=predict(t,tau,cmp)
            for g in [lambda x:x*17+5,lambda x:np.arcsinh(x)]:
                new,cmp=calibrate(g(s),y,policy)
                np.testing.assert_array_equal(base,predict(g(t),new,cmp))

    def test_normal_budget_ties(self):
        s=np.r_[np.zeros(95),np.ones(5)];y=np.zeros(100,dtype=int)
        for pol,alpha in [('normal_005',.005),('normal_010',.01),('normal_050',.05)]:
            tau,cmp=calibrate(s,y,pol)
            self.assertLessEqual(predict(s,tau,cmp).mean(),alpha)

    def test_symbolic_monotone_can_collapse_in_floating_point(self):
        self.assertFalse(preserves_observed_order([0.,1e-20,1.],lambda x:17*x+5))
        self.assertTrue(preserves_observed_order([0.,1e-20,1.],lambda x:16*x))

    def test_allocation_closure_and_antisymmetry(self):
        y=np.array([0,1,1,0,1,0,0]);a=metrics(y,[0,1,0,1,1,0,0]);b=metrics(y,[1,0,0,0,1,0,0])
        ab=allocation(a,b);ba=allocation(b,a)
        self.assertLess(ab['allocation_error'],1e-14)
        for k in ['coverage_allocation','recall_allocation','fpr_allocation']:
            self.assertAlmostEqual(ab[k],-ba[k])

    def test_different_prevalence_allocation_rejected(self):
        with self.assertRaises(ValueError):allocation(metrics([0,1],[0,1]),metrics([0,0,1],[0,0,1]))

    def test_sign_test_scope(self):
        self.assertEqual(sign_p([1]*5)['sign_p'],.0625)
        self.assertEqual(sign_p([1]*10)['sign_p'],.001953125)
        self.assertEqual(sign_p([0]*5)['nonzero_pairs'],0)

    def test_holm(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_official_affiliation_reference(self):
        r=pr_from_events([(4,5),(8,9)],[(3,4),(7,10)],(0,10))
        self.assertAlmostEqual(r['precision'],.82,places=2)
        self.assertAlmostEqual(r['recall'],.84,places=2)

    def test_gate_missing_is_not_failure(self):
        self.assertEqual(claim_assessment({})['status'],'not_evaluable')

    def test_gate_fault_and_valid_counterexample(self):
        base=dict(target='pipeline',metric='strict-F1',unit='fixture',calibration='validation',
                  snapshot_verified=True,causal_online=False,timeline_causal=False,
                  event_boundaries_known=True,independence_verified=False)
        self.assertEqual(claim_assessment(base)['status'],'within_scope')
        self.assertEqual(claim_assessment(dict(base,target='deployment',calibration='test_oracle'))['status'],'contradicted')
        self.assertEqual(claim_assessment(dict(base,snapshot_verified=False))['status'],'not_evaluable')
        self.assertEqual(claim_assessment(dict(base,metric='PA-F1',claim_strict=True))['status'],'contradicted')
        self.assertEqual(claim_assessment(dict(base,causal_online=True))['status'],'contradicted')
        self.assertEqual(claim_assessment(dict(base,snapshot_hash_mismatch=True))['status'],'contradicted')
        from diagnose_claim import diagnose
        card={'claim':'Archived score arithmetic only','target':'pipeline',
              'gates':{g:{'status':'not_applicable','explanation':'Not part of this bounded fixture'}
                       for g in ['G1','G2','G3','G4','G5','G6','G7','G8']}}
        card['gates']['G1']={'status':'within_scope','evidence':[]}
        self.assertEqual(diagnose(card)['overall'],'not_evaluable')


if __name__=='__main__':
    unittest.main(verbosity=2)
