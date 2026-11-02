import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate import validate_preflight, validate_pilot, run_barriers,validate_all_receipts,launch_batch,sha,TASKS
CODE='test-frozen-code'


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = dict(code_id=CODE, model='qwen3b', dataset='dolly_5k',
                             seed=42, chunk_size=128, lora_init='gaussian', input_sha256='inputs')
        self.receipt = {**self.expected, 'status':'PASS', 'frozen_base_unchanged':True,
                        'checkpoint_continuation':'EXACT_MATCH', 'source_sample_match':True,
                        'precision_path_check':{'status':'PASS', 'failed_parameters':[],
                            'all_decoder_calls_under_autocast':True,
                            'fp16_reduced_precision_reduction':True},
                        'fp16_trainer_disk_resume':{'status':'EXACT_MATCH',
                            'successful_optimizer_steps':13, 'skipped_optimizer_steps':3, 'steps':16,
                            'declared_precision_lengths':[130,258],
                            'precision_path_checks':[{'status':'PASS','tokens':n,'probe_b_value':b}
                                for n in [130,258] for b in [0.,.001]]},
                        'longest_actual_trainer':{'status':'PASS','steps':16,'successful_optimizer_steps':16,
                            'parameters_finite':True,'optimizer_finite':True,'history_finite':True,
                            'adapter_changed':True,'adapter_restored':True},
                        'contract':{'frozen_sha256':'base', 'init_lora_weights':'gaussian',
                            'projection_policy':'whole-head-gemm; cpu-logits; streamed-fp32-ce',
                            'training_fp16_reduced_precision_reduction':True,
                            'generation_fp16_reduced_precision_reduction':True}}

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value), encoding='utf-8')

    def pilot(self):
        from gate import sha
        self.write('receipt.json', self.receipt)
        self.write('run_manifest.json', {'status':'PASS', 'stage':'complete', 'config':{
            **self.expected, 'count':100, 'initialized_from':'base', 'protocol':'A',
            'preflight_sha256':sha(self.root/'receipt.json')}})
        self.write('training_complete.json', {'n':100, 'global_step':100, 'frozen_base_unchanged':True,
            'successful_optimizer_steps':97,'skipped_optimizer_steps':3,
            'adapter_parameters_finite':True,'optimizer_state_finite':True})
        self.write('model_contract.json',self.receipt['contract'])
        self.write('resolved_config.json',dict(learning_rate=5e-5,num_train_epochs=1.,
            per_device_train_batch_size=1,gradient_accumulation_steps=1,lr_scheduler_type='cosine',
            warmup_ratio=.1,fp16=True,bf16=False,weight_decay=0.,max_grad_norm=1.,seed=42,
            adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,optim='adamw_torch',
            gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False}))
        self.write('adaptation.json', [dict(source_index=i, base_ce=4., current_ce=3., weight=.2,
                                          loss=.6, lr=5e-5) for i in range(100)])
        (self.root/'tlm.jsonl').write_text(''.join(json.dumps(dict(source_index=i,predict='answer'))+'\n'
                                               for i in range(100)), encoding='utf-8')
        self.write('metrics.json', {'n':100, 'predictions_sha256':sha(self.root/'tlm.jsonl')})
        self.write('quality_latest.json', {'status':'PASS', 'n':100, 'window_start':0, 'alerts':[]})
        return self.root/'receipt.json'

    def test_accepts_complete_preflight(self):
        validate_preflight(self.receipt, self.expected)

    def test_rejects_wrong_code_or_initialization(self):
        for key, value in [('code_id','old'), ('lora_init','kaiming')]:
            bad=copy.deepcopy(self.receipt); bad[key]=value
            with self.assertRaises(ValueError): validate_preflight(bad,self.expected)

    def test_rejects_superficial_pass_with_failed_gradient_or_no_updates(self):
        for change in ['gradient','updates','backend']:
            bad=copy.deepcopy(self.receipt)
            if change=='gradient': bad['precision_path_check']['failed_parameters']=['q.B']
            if change=='updates': bad['fp16_trainer_disk_resume']['successful_optimizer_steps']=0
            if change=='backend': bad['contract']['projection_policy']='old-tiled-head'
            with self.assertRaises(ValueError): validate_preflight(bad,self.expected)

    def test_accepts_complete_100_row_pilot(self):
        receipt=self.pilot(); validate_pilot(self.root,self.expected,receipt)

    def test_rejects_incomplete_longest_probe_or_precision_coverage(self):
        for key in ['parameters_finite','adapter_restored']:
            bad=copy.deepcopy(self.receipt);bad['longest_actual_trainer'][key]=False
            with self.assertRaises(ValueError):validate_preflight(bad,self.expected)
        bad=copy.deepcopy(self.receipt);bad['fp16_trainer_disk_resume']['precision_path_checks'].pop()
        with self.assertRaises(ValueError):validate_preflight(bad,self.expected)

    def test_rejects_wrong_learning_rate_or_no_optimizer_updates(self):
        receipt=self.pilot();configuration=json.loads((self.root/'resolved_config.json').read_text())
        configuration['learning_rate']=1e-3;self.write('resolved_config.json',configuration)
        with self.assertRaises(ValueError):validate_pilot(self.root,self.expected,receipt)
        self.pilot();summary=json.loads((self.root/'training_complete.json').read_text())
        summary['successful_optimizer_steps']=0;self.write('training_complete.json',summary)
        with self.assertRaises(ValueError):validate_pilot(self.root,self.expected,receipt)

    def test_rejects_16_row_pilot_and_quality_alert(self):
        receipt=self.pilot()
        self.write('quality_latest.json', {'status':'WAIT','n':16})
        with self.assertRaises(ValueError): validate_pilot(self.root,self.expected,receipt)
        self.pilot(); self.write('quality_alert.json', {'status':'ALERT'})
        with self.assertRaises(ValueError): validate_pilot(self.root,self.expected,receipt)

    def test_rejects_incomplete_or_duplicate_predictions(self):
        receipt=self.pilot()
        for text in ['{}\n', ''.join(json.dumps({'source_index':0})+'\n' for _ in range(100))]:
            (self.root/'tlm.jsonl').write_text(text,encoding='utf-8')
            with self.assertRaises(ValueError): validate_pilot(self.root,self.expected,receipt)

    def test_rejects_nonfinite_adaptation_and_receipt_swap(self):
        receipt=self.pilot()
        rows=json.loads((self.root/'adaptation.json').read_text());rows[3]['loss']=float('nan')
        self.write('adaptation.json',rows)
        with self.assertRaises(ValueError):validate_pilot(self.root,self.expected,receipt)
        self.pilot();self.write('receipt.json',{**self.receipt,'extra':'changed'})
        with self.assertRaises(ValueError):validate_pilot(self.root,self.expected,receipt)

    def test_any_failed_preflight_blocks_every_pilot_and_formal(self):
        calls=[]
        def stage(name):
            calls.append(name)
            return name!='preflight'
        self.assertFalse(run_barriers(stage));self.assertEqual(calls,['preflight'])

    def test_any_failed_pilot_blocks_every_formal(self):
        calls=[]
        def stage(name):
            calls.append(name)
            return name!='pilot'
        self.assertFalse(run_barriers(stage));self.assertEqual(calls,['preflight','pilot'])

    def test_all_stages_pass_in_order(self):
        calls=[]
        self.assertTrue(run_barriers(lambda name:calls.append(name) or True))
        self.assertEqual(calls,['preflight','pilot','formal'])

    def test_last_receipt_hash_mismatch_starts_zero_subprocesses(self):
        import subprocess
        jobs=[{'model':m,'dataset':d,'gpu':i} for i,(m,d) in enumerate(sorted(TASKS))]
        hashes={}
        def name(job):return job['model']+'__'+job['dataset']
        def expected(job):return {**self.expected,'model':job['model'],'dataset':job['dataset']}
        def path(job):return self.root/(name(job)+'.json')
        for job in jobs:
            self.write(name(job)+'.json',{**self.receipt,**expected(job)})
            hashes[name(job)]=sha(path(job))
        path(jobs[-1]).write_text('{}')
        with mock.patch('gate.verify_code'),mock.patch('subprocess.run') as process:
            with self.assertRaisesRegex(ValueError,'Receipt changed'):
                launch_batch(jobs,lambda:validate_all_receipts(jobs,self.root,CODE,path,expected,hashes),
                             lambda job:subprocess.run(['should-never-start']))
            process.assert_not_called()


if __name__=='__main__':unittest.main()
