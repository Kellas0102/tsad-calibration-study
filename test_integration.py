"""Tests run only synthetic score/count fixtures, never a detector."""
import unittest
import numpy as np
from integrate_evidence import segmented
from diagnostics import metrics


class IntegrationTests(unittest.TestCase):
    def test_boundaries_split_one_alarm_into_two_trials(self):
        y=np.array([0,1,1,0,0,1,1,0])
        pred=np.ones(8,dtype=bool)
        flat=metrics(y,pred)
        split=segmented(y,pred,[(0,4),(4,8)])
        self.assertEqual(flat['event_tp'],1)
        self.assertEqual(split['event_tp'],2)
        self.assertEqual(split['event_fn'],0)

    def test_no_event_trial_not_perfect_macro(self):
        y=np.array([0,0,0,1]);pred=y.astype(bool)
        split=segmented(y,pred,[(0,2),(2,4)])
        self.assertEqual(split['event_macro_defined_n'],1)
        self.assertEqual(split['event_f1'],1)
        self.assertEqual(split['event_f1_macro'],1)

    def test_pooled_is_not_average_of_trial_f1(self):
        y=np.array([0,1,0,0,1,0,0,0]);pred=np.array([0,1,0,1,0,1,0,1])
        split=segmented(y,pred,[(0,3),(3,8)])
        self.assertNotAlmostEqual(split['event_f1'],split['event_f1_macro'])


if __name__=='__main__':unittest.main(verbosity=2)
