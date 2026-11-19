"""Development regression tests after observing the v1 challenge failures."""
import unittest

from diagnostics import claim_assessment
from diagnose_claim import diagnose, GATES


def complete():
    return dict(target='pipeline', metric='strict_F1', unit='fixture', calibration='validation',
                snapshot_verified=True, causal_online=False, timeline_causal=True,
                event_boundaries_known=True, independence_verified=True)


class ProtocolV2RegressionTests(unittest.TestCase):
    def test_hash_contradiction_kept_with_missing_calibration(self):
        r = complete(); r.pop('calibration'); r['snapshot_hash_mismatch'] = True
        result = claim_assessment(r)
        self.assertEqual(result['status'], 'contradicted')
        self.assertIn('calibration', result['missing_fields'])
        self.assertIn('snapshot_hash_mismatch', result['contradictions'])

    def test_pa_contradiction_kept_with_missing_unit(self):
        r = complete(); r.pop('unit'); r.update(metric='PA-F1', claim_strict=True)
        result = claim_assessment(r)
        self.assertEqual(result['status'], 'contradicted')
        self.assertIn('unit', result['missing_fields'])

    def test_explicit_null_is_unknown_not_false(self):
        r = complete(); r.update(causal_online=True, timeline_causal=None)
        result = claim_assessment(r)
        self.assertEqual(result['status'], 'not_evaluable')
        self.assertEqual(result['contradictions'], [])

    def test_string_false_not_used_as_boolean(self):
        r = complete(); r.update(causal_online='false')
        self.assertEqual(claim_assessment(r)['status'], 'not_evaluable')

    def test_equivalence_not_automatically_adjudicated(self):
        r = complete(); r.update(equivalence_asserted=True)
        result = claim_assessment(r)
        self.assertEqual(result['status'], 'not_evaluable')
        self.assertIn('equivalence', result['unsupported_inferences'])

    def test_generalization_needs_review_even_if_independence_declared(self):
        r = complete(); r['target'] = 'generalization'
        self.assertEqual(claim_assessment(r)['status'], 'not_evaluable')

    def test_operational_utility_not_certified_by_complete_safety_fields(self):
        r = complete(); r['target'] = 'deployment'
        result = claim_assessment(r)
        self.assertEqual(result['status'], 'not_evaluable')
        self.assertFalse(result['certificate'])

    def test_known_deployment_oracle_still_contradicted(self):
        r = complete(); r.update(target='deployment', calibration='test_oracle')
        result = claim_assessment(r)
        self.assertEqual(result['status'], 'contradicted')
        self.assertIn('operational_utility', result['unsupported_inferences'])

    def test_arbitrary_requested_inference_not_silently_ignored(self):
        r = complete(); r['requested_inferences'] = ['mean_superiority']
        self.assertEqual(claim_assessment(r)['status'], 'not_evaluable')

    def test_narrow_complete_control_remains_scoped(self):
        self.assertEqual(claim_assessment(complete())['status'], 'within_scope')

    def test_card_aggregation_keeps_contradiction_and_missing_separate(self):
        card = {'claim':'synthetic aggregation fixture', 'target':'pipeline',
                'gates':{g:{'status':'within_scope','evidence':['fixture:locator']} for g in GATES}}
        card['gates']['G2']['status'] = 'contradicted'
        card['gates'].pop('G5')
        result = diagnose(card)
        self.assertEqual(result['overall'], 'contradicted')
        self.assertEqual(next(r for r in result['gates'] if r['gate']=='G5')['status'], 'not_evaluable')
        self.assertFalse(result['certificate'])


if __name__ == '__main__': unittest.main(verbosity=2)
