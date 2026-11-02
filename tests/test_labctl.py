import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import io
import os
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/labctl.py'
if PATH.exists():
    spec=importlib.util.spec_from_file_location('labctl',PATH)
    lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)
else:lab=None

class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(lab,'Unified launcher has not been implemented')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.code=self.root/'code';self.code.mkdir()
        (self.code/'job.py').write_text("import json,sys\nfrom pathlib import Path\np=Path(sys.argv[1]);p.mkdir()\n(p/'metrics.json').write_text(json.dumps({'score':0.8,'quality_status':'ALERT'}))\n",encoding='utf-8')
        self.site={'root':str(self.root/'lab'),'python':sys.executable,'code_roots':[str(self.code)],
                   'input_roots':[str(self.root)],'legacy_lock_paths':[],'tracking_uri':None,
                   'gpu_ids':[0,1],'max_idle_memory_mib':128,'poll_seconds':0.02}
        self.request={'schema_version':1,'experiment_id':'smoke-001','owner':'member-a',
            'issue':'https://github.com/Falling-star1/llm-tta-lab/issues/1','method':'smoke',
            'model':'synthetic','dataset':'synthetic','protocol_id':'smoke-v1','seed':42,'commit':'a'*40,
            'code_root':str(self.code),'code_files':{'job.py':lab.sha(self.code/'job.py')},
            'entrypoint':'job.py','inputs':{},'argv':['{python}','{entrypoint}','{output}'],
            'gpu_count':0,'timeout_seconds':10,'queue_timeout_seconds':1,'legacy_lock_policy':'wrapper'}

    def test_cpu_success_preserves_quality_alert(self):
        r=lab.run_request(self.request,self.site)
        self.assertEqual((r['status'],r['quality_status'],r['returncode']),('SUCCEEDED','ALERT',0))
        self.assertTrue((Path(r['output_dir'])/'metrics.json').is_file())

    def test_same_job_rejected(self):
        lab.run_request(self.request,self.site)
        with self.assertRaises(lab.LabError):lab.run_request(self.request,self.site)

    def test_renamed_duplicate_rejected(self):
        lab.run_request(self.request,self.site)
        r=copy.deepcopy(self.request);r['experiment_id']='smoke-002'
        with self.assertRaises(lab.LabError):lab.run_request(r,self.site)

    def test_code_change_rejected_before_run_created(self):
        (self.code/'job.py').write_text('changed')
        with self.assertRaises(lab.LabError):lab.run_request(self.request,self.site)
        self.assertFalse((Path(self.site['root'])/'runs/smoke-001').exists())

    def test_input_fingerprint_checked(self):
        p=self.root/'data.json';p.write_text('old')
        self.request['inputs']={'dataset':{'path':str(p),'sha256':lab.sha(p)}};p.write_text('new')
        with self.assertRaises(lab.LabError):lab.validate(self.request,self.site)

    def test_unsafe_id_shell_and_path_rejected(self):
        for changes in [{'experiment_id':'../escape'},{'argv':['bash','-c','echo bad']},
                        {'entrypoint':'../job.py'},{'timeout_seconds':0}]:
            r=copy.deepcopy(self.request);r.update(changes)
            with self.assertRaises(lab.LabError):lab.validate(r,self.site)

    def test_unlisted_dependency_rejected(self):
        (self.code/'new.py').write_text('x=1')
        with self.assertRaises(lab.LabError):lab.validate(self.request,self.site)

    def test_exit_code_separate_from_success(self):
        (self.code/'job.py').write_text('raise SystemExit(7)')
        self.request['code_files']['job.py']=lab.sha(self.code/'job.py')
        r=lab.run_request(self.request,self.site)
        self.assertEqual((r['status'],r['returncode']),('FAILED',7))

    def test_cross_process_lease_exclusive(self):
        p=self.root/'gpu.lock'
        with lab.FileLease(p):
            code="import sys;sys.path.insert(0,sys.argv[1]);from labctl import FileLease\nwith FileLease(sys.argv[2]):pass"
            r=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'scripts'),str(p)],capture_output=True)
            self.assertNotEqual(r.returncode,0)
        with lab.FileLease(p):pass

    def test_busy_gpu_skipped(self):
        r=copy.deepcopy(self.request);r['gpu_count']=1
        devices=[{'index':0,'memory_mib':1000,'busy':True},{'index':1,'memory_mib':1,'busy':False}]
        with lab.reserve_gpus(r,self.site,probe=lambda:devices) as (ids,leases):
            self.assertEqual(ids,[1]);self.assertTrue(leases)

    def test_legacy_lock_blocks_selection(self):
        self.site['legacy_lock_paths']=[str(self.root/'old-{gpu}.lock')]
        r=copy.deepcopy(self.request);r.update(gpu_count=1,gpu_candidates=[0])
        with lab.FileLease(self.root/'old-0.lock'):
            with self.assertRaises(lab.LabError):
                with lab.reserve_gpus(r,self.site,probe=lambda:[{'index':0,'memory_mib':1,'busy':False}]):pass

    def test_child_policy_does_not_deadlock_legacy_code(self):
        self.site['legacy_lock_paths']=[str(self.root/'old-{gpu}.lock')]
        r=copy.deepcopy(self.request);r.update(gpu_count=1,gpu_candidates=[0],legacy_lock_policy='child')
        with lab.reserve_gpus(r,self.site,probe=lambda:[{'index':0,'memory_mib':1,'busy':False}]) as (ids,leases):
            with lab.FileLease(self.root/'old-0.lock'):pass
            self.assertEqual(ids,[0])

    def test_tracking_outage_preserves_completed_job(self):
        self.site['tracking_uri']='http://127.0.0.1:1'
        r=lab.run_request(self.request,self.site)
        self.assertEqual((r['status'],r['mlflow_sync']),('SUCCEEDED','PENDING'))

    def test_auto_sync_busy_preserves_successful_execution(self):
        with patch.object(lab,'sync_run',side_effect=lab.LabError('Already leased: sync')):
            result=lab.run_request(self.request,self.site)
        self.assertEqual(result['status'],'SUCCEEDED')

    def test_timeout_keeps_record(self):
        (self.code/'job.py').write_text('import time;time.sleep(30)')
        self.request['code_files']['job.py']=lab.sha(self.code/'job.py')
        self.request['timeout_seconds']=0.2
        self.assertEqual(lab.run_request(self.request,self.site)['status'],'TIMED_OUT')

    def test_gpu_request_requires_input_contract(self):
        self.request['gpu_count']=1
        with self.assertRaises(lab.LabError):lab.validate(self.request,self.site)

    def test_sync_refuses_running_job(self):
        directory=self.root/'running';directory.mkdir()
        lab.write(directory/'state.json',{'status':'RUNNING'})
        with self.assertRaises(lab.LabError):lab.sync_run(directory,self.site)

    def test_completed_sync_is_idempotent(self):
        directory=self.root/'done';directory.mkdir()
        state={'status':'SUCCEEDED','mlflow_sync':'SYNCED','mlflow_run_id':'existing'}
        lab.write(directory/'state.json',state)
        self.assertEqual(lab.sync_run(directory,self.site),state)

    @unittest.skipUnless(os.name=='posix' and Path('/proc').exists(),'Linux process-group integration')
    def test_timeout_waits_for_worker_after_leader_exits(self):
        script="""import subprocess,sys,time
worker='import signal,time;signal.signal(signal.SIGTERM,lambda *a:None);time.sleep(0.9)'
subprocess.Popen([sys.executable,'-c',worker],close_fds=True)
time.sleep(30)
"""
        (self.code/'job.py').write_text(script)
        self.request['code_files']['job.py']=lab.sha(self.code/'job.py')
        self.request['timeout_seconds']=0.4
        start=time.monotonic();result=lab.run_request(self.request,self.site)
        self.assertEqual(result['status'],'TIMED_OUT')
        self.assertGreater(time.monotonic()-start,0.8,'Released before worker finished')

    def test_concurrent_sync_cannot_create_duplicate_run(self):
        self.site['tracking_uri']='http://unused.test'
        directory=self.root/'done';directory.mkdir()
        state={'status':'SUCCEEDED','started_ms':1,'finished_ms':2,'identity':'x','output_dir':'synthetic'}
        lab.write(directory/'state.json',state);lab.write(directory/'request.json',self.request)
        searching=threading.Event();resume=threading.Event();creates=[];results=[]
        def fake_post(uri,endpoint,payload):
            if endpoint=='runs/search':
                searching.set();resume.wait(5);return {'runs':[]}
            if endpoint=='runs/create':creates.append(1);return {'run':{'info':{'run_id':'one'}}}
            return {}
        with (patch.object(lab.urllib.request,'urlopen',side_effect=lambda *a,**k:io.BytesIO(b'{"experiment":{"experiment_id":"2"}}')),
              patch.object(lab,'post',side_effect=fake_post)):
            thread=threading.Thread(target=lambda:results.append(lab.sync_run(directory,self.site)))
            thread.start();self.assertTrue(searching.wait(5))
            try:
                with self.assertRaises(lab.LabError):lab.sync_run(directory,self.site)
            finally:resume.set();thread.join(8)
        self.assertFalse(thread.is_alive());self.assertEqual(len(creates),1)
        self.assertEqual(results[0]['mlflow_sync'],'SYNCED')

if __name__=='__main__':unittest.main()
