import json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from continue_evaluation import read_prefix,record_quality,validate_training_identity,BASES,CODE
from supervisor import classify


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_complete_prefix_bytes_are_preserved(self):
        raw=b'{"source_index":0,"predict":"answer"}\n'
        path=self.root/'tlm.jsonl';path.write_bytes(raw+b'{"source_index":1')
        rows,complete,tail=read_prefix(path)
        self.assertEqual(rows,[{'source_index':0,'predict':'answer'}])
        self.assertEqual(complete,raw);self.assertEqual(tail,b'{"source_index":1')
        self.assertEqual(path.read_bytes(),raw+b'{"source_index":1')

    def test_bad_indexes_rejected(self):
        p=self.root/'tlm.jsonl';p.write_text('{"source_index":2}\n')
        with self.assertRaises(ValueError):read_prefix(p)

    def test_alert_stays_alert_and_does_not_raise(self):
        report={'status':'ALERT','window_start':0,'n':100,'alerts':['empty']}
        got=record_quality(lambda rows,source:report,[],[],self.root)
        self.assertEqual(got['status'],'ALERT')
        self.assertEqual(json.loads((self.root/'quality_latest.json').read_text())['status'],'ALERT')
        self.assertEqual(json.loads((self.root/'quality_alerts.jsonl').read_text())['alerts'],['empty'])

    def test_pilot_or_wrong_base_rejected(self):
        m={'config':dict(code_id=CODE,model='qwen3b',dataset='dolly_5k',count=5000,seed=42,
            lora_init='gaussian',initialized_from='base',chunk_size=128,protocol='A')}
        t=dict(n=5000,global_step=5000,frozen_base_unchanged=True,adapter_parameters_finite=True,optimizer_state_finite=True)
        c={'frozen_sha256':BASES['qwen3b']}
        validate_training_identity(m,t,c,'qwen3b','dolly_5k')
        m['config']['count']=100
        with self.assertRaises(ValueError):validate_training_identity(m,t,c,'qwen3b','dolly_5k')
        m['config']['count']=5000;c['frozen_sha256']='old'
        with self.assertRaises(ValueError):validate_training_identity(m,t,c,'qwen3b','dolly_5k')

    def test_coordinator_keeps_active_workers_and_only_resumes_quality_failures(self):
        self.assertEqual(classify({'status':'RUNNING'},None,True,False),'ACTIVE')
        self.assertEqual(classify({'status':'FAIL'},{'type':'RuntimeError','message':'Output quality alert; results preserved'},False,False),'RESUME')
        self.assertEqual(classify({'status':'FAIL'},{'type':'ValueError','message':'Base differs'},False,False),'BLOCKED')
        self.assertEqual(classify({'status':'PASS'},None,False,True),'COMPLETE')
        self.assertEqual(classify({'status':'RUNNING'},None,False,False),'BLOCKED')

    def test_six_paths_do_not_substitute_for_six_registered_tasks(self):
        from supervisor import validate_jobs
        jobs=[dict(model='qwen3b',dataset='dolly_5k',gpu=i,run='/tmp/task'+str(i)) for i in range(6)]
        with self.assertRaises(ValueError):validate_jobs(jobs)


if __name__=='__main__':unittest.main()
