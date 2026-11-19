"""Workflow tests use synthetic metadata only; no model code is executed."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile

from diagnose_claim import diagnose,GATES

ROOT=Path(__file__).resolve().parents[1]


class EvidenceWorkflowTests(unittest.TestCase):
    def test_gap_consolidation_and_event_false_alarm_partition(self):
        from secondary_analysis import consolidate,alarm_detail
        import numpy as np
        p=np.array([0,1,0,1,0,0,1,0],dtype=bool)
        np.testing.assert_array_equal(consolidate(p,0),p)
        np.testing.assert_array_equal(consolidate(p,1),[0,1,1,1,0,0,1,0])
        m=alarm_detail([0,1,1,1,0,0,0,0],p)
        self.assertEqual(m['event_fp'],m['unmatched_overlapping_alarms']+m['alarms_without_true_overlap'])

    def test_required_gates_cannot_all_be_not_applicable(self):
        for target in ['score','pipeline','event','deployment','generalization']:
            card={'claim':'fixture','target':target,'gates':{k:{'status':'not_applicable','explanation':'attempted bypass'} for k in GATES}}
            self.assertEqual(diagnose(card)['overall'],'not_evaluable')

    def test_unknown_gate_rejected(self):
        with self.assertRaises(ValueError):diagnose({'claim':'fixture','target':'score','gates':{'G99':{}}})

    def test_locator_string_not_treated_as_locator_list(self):
        with self.assertRaises(ValueError):diagnose({'claim':'fixture','target':'score','gates':{'G1':{'evidence':'some-file'}}})

    def test_narrow_claim_and_promotion_are_distinguished(self):
        a=json.loads((ROOT/'examples/archived_pipeline_claim.json').read_text())
        b=json.loads((ROOT/'examples/deployment_promotion.json').read_text())
        self.assertEqual(diagnose(a)['overall'],'within_scope')
        self.assertEqual(diagnose(b)['overall'],'not_evaluable')

    def test_exporter_only_synthetic_project_and_no_sensitive_contents(self):
        with tempfile.TemporaryDirectory(prefix='mtsad-export-fixture-') as temp:
            root=Path(temp)/'project';out=Path(temp)/'export'
            (root/'code').mkdir(parents=True)
            (root/'code/data.py').write_text('raise RuntimeError("DO NOT EXECUTE")\n')
            (root/'data/SKAB/anomaly-free').mkdir(parents=True)
            (root/'data/SKAB/anomaly-free/fixture.csv').write_text('timestamp,x\n0,123456789\n')
            rd=root/'results/runs/skab_catch_fixture';rd.mkdir(parents=True)
            (rd/'metrics.json').write_text(json.dumps({'dataset':'skab','model':'catch','f1':.2,'API_KEY':'NEVER-COPY-SECRET'}))
            command=[sys.executable,str(ROOT/'device_handoff/export_training_evidence.py'),
                     '--project-root',str(root),'--output',str(out),'--dataset','SKAB']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((out/'training_device_evidence.json').read_text())
            self.assertFalse(report['model_code_executed']);self.assertFalse(report['historical_usage_verified'])
            with zipfile.ZipFile(out.with_suffix('.zip')) as archive:
                combined=b'\n'.join(archive.read(n) for n in archive.namelist())
                self.assertNotIn(b'NEVER-COPY-SECRET',combined)
                self.assertNotIn(b'123456789',combined)
                self.assertNotIn('data/SKAB/anomaly-free/fixture.csv',archive.namelist())
            again=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)


if __name__=='__main__':unittest.main(verbosity=2)
