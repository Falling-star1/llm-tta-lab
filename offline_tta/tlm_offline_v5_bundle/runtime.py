"""Pinned server loading, identities and shared generation/scoring utilities."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import time

from input_tokens import require_input_record, encode_input

HERE = Path(__file__).resolve().parent
REPOSITORY = Path('/home/chuwenjing/TTA_LLM/TLM')
COMMIT = '496c9ed97d0fb8bc4854ed2479259e7e9e70a790'
VERSIONS = {'torch':'2.5.1+cu118','transformers':'4.46.1','peft':'0.12.0',
            'accelerate':'1.0.1','datasets':'3.1.0','numpy':'1.26.4'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def output_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.run.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def wait_for_idle_gpu(index, attempts=10):
    """Allow the previous stage's utilization sample to expire before launch."""
    for attempt in range(attempts):
        result=subprocess.check_output(['nvidia-smi','--id='+index,
            '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        memory,utilization=[int(x.strip()) for x in result.strip().split(',')]
        if memory<128 and utilization<5:
            return
        if attempt+1<attempts:
            time.sleep(1)
    raise ValueError('Selected physical GPU is busy: '+index+' '+result.strip())


@contextmanager
def gpu_lock():
    """Coordinate all TLM release directories and reject an already busy GPU."""
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible.isdecimal():
        raise ValueError('Set CUDA_VISIBLE_DEVICES to one physical numeric GPU index')
    lock_root=Path('/data/chuwenjing/tta_offline/tlm_gpu_locks')/('gpu-'+str(int(visible)))
    with output_lock(lock_root):
        wait_for_idle_gpu(visible)
        yield


def code_identity():
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and
                   p.suffix in ('.py','.json','.sh','.md') and '__pycache__' not in p.parts)
    manifest = {p.relative_to(HERE).as_posix():sha256(p) for p in files}
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()


def specifications(model_id, dataset):
    profiles = json.loads((HERE/'profiles.json').read_text(encoding='utf-8'))
    spec = json.loads((HERE/'datasets.json').read_text(encoding='utf-8'))
    if spec['source_commit'] != COMMIT or model_id not in profiles:
        raise ValueError('Unregistered source or model')
    return profiles[model_id], next(d for d in spec['datasets'] if d['alias']==dataset)


def read_inputs(directory, expected_sha, expected_count=5000):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['source_sha256'] != expected_sha or manifest['record_count'] != expected_count:
        raise ValueError('Prepared dataset identity differs')
    path = directory/'inputs.jsonl'
    if sha256(path) != manifest['files']['inputs.jsonl']:
        raise ValueError('Prepared input hash differs')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(rows) != expected_count:
        raise ValueError('Prepared input count differs')
    for index, row in enumerate(rows):
        require_input_record(row)
        if row['source_index'] != index:
            raise ValueError('Prepared inputs contain duplicate or missing indexes')
    return rows, manifest


def check_environment():
    actual = {key:importlib.metadata.version(key) for key in VERSIONS}
    if actual != VERSIONS:
        raise ValueError('Server package versions differ: '+str(actual))
    if json.loads((REPOSITORY/'SOURCE_VERSION.json').read_text())['commit'] != COMMIT:
        raise ValueError('Wrong fixed TLM source commit')
    return actual


def configure(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python')
    # Avoid FP16 truncation in intermediate GEMM reductions during adaptation.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def frozen_digest(model):
    """Hash every frozen tensor, bounded CPU transfers, no cloned full model."""
    import torch
    digest = hashlib.sha256()
    count = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_' in name:
                continue
            if parameter.requires_grad or parameter.grad is not None:
                raise ValueError('Base parameter is trainable or has a gradient: '+name)
            digest.update(json.dumps([name,list(parameter.shape),str(parameter.dtype)]).encode())
            flat = parameter.detach().reshape(-1)
            for start in range(0,flat.numel(),1024*1024):
                digest.update(flat[start:start+1024*1024].cpu().contiguous().view(torch.uint8).numpy().tobytes())
            count += 1
    if not count:
        raise ValueError('No frozen base tensors')
    return digest.hexdigest()


class LoadedModel:
    def __init__(self, model_id, dataset, seed=42, initialization='gaussian'):
        import torch
        check_environment()
        configure(seed)
        self.profile, self.dataset = specifications(model_id,dataset)
        for item in self.profile['files']:
            path = Path(self.profile['model'])/item['filename']
            if not path.is_file() or sha256(path) != item['sha256']:
                raise ValueError('Model file differs: '+str(path))
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('Current validated implementation expects one visible CUDA GPU')
        sys.path.insert(0,str(REPOSITORY/'src'))
        from llamafactory.chat import ChatModel
        chat = ChatModel({'model_name_or_path':self.profile['model'],
            'template':self.profile['template'],'infer_backend':'huggingface',
            'infer_dtype':'float16','flash_attn':'sdpa','finetuning_type':'full',
            'trust_remote_code':False,'do_sample':False,'temperature':0.,
            'max_new_tokens':512,'cutoff_len':4096})
        base = chat.engine.model
        if base.config.model_type != self.profile['model_type']:
            raise ValueError('Wrong model architecture')
        if any(p.dtype!=torch.float16 for p in base.parameters() if p.is_floating_point()):
            raise ValueError('Base must be entirely FP16')
        from train_core import build_lora
        self.model = build_lora(base, seed, initialization=initialization)
        self.tokenizer, self.template = chat.engine.tokenizer,chat.engine.template
        self.system = chat.engine.generating_args['default_system']
        self.stop_ids = self.template.get_stop_token_ids(self.tokenizer)
        self.pad_id = self.tokenizer.pad_token_id
        self.device = self.model.get_input_embeddings().weight.device
        self.model.config.use_cache=False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if count != self.profile['expected_trainable_parameters']:
            raise ValueError('Unexpected LoRA parameter count: '+str(count))
        self.contract = {'model_id':model_id,'template':self.profile['template'],
            'trainable_parameters':count,'stop_ids':self.stop_ids,'pad_id':self.pad_id,
            'system':self.system,'seed':seed,'init_lora_weights':initialization,
            'training_precision_context':'accelerate-autocast',
            'projection_policy':'padded-tail-tiles; concatenate-token-losses; single-fp32-sum',
            'training_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            'generation_fp16_reduced_precision_reduction':True,
            'frozen_sha256':frozen_digest(self.model),
            'files':{x['filename']:x['sha256'] for x in self.profile['files']}}

    def tokens(self, rows):
        return [encode_input(row,self.tokenizer,self.template,self.system) for row in rows]

    def generate(self, ids, max_new_tokens=512):
        import torch
        from transformers import GenerationConfig
        self.model.gradient_checkpointing_disable()
        self.model.eval()
        tensor = torch.tensor([ids],device=self.device)
        cfg = GenerationConfig(max_new_tokens=max_new_tokens,do_sample=False,num_beams=1,
            repetition_penalty=1.,eos_token_id=self.stop_ids,pad_token_id=self.pad_id)
        previous_reduction = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        try:
            # Match the frozen Source generation environment, then restore the
            # adaptation policy even if generation raises an exception.
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            with torch.no_grad():
                sequence = self.model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),
                                                generation_config=cfg,use_cache=True)
        finally:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous_reduction
        generated = sequence[0,len(ids):].detach().cpu().tolist()
        return {'predict':self.tokenizer.decode(generated,skip_special_tokens=True),
                'generated_ids':generated,'generated_tokens':len(generated),
                'hit_generation_limit':len(generated)>=max_new_tokens}


class Scorer:
    def __init__(self, metric, extractor=None):
        self.metric=metric
        if metric=='exact_match':
            if extractor is None:
                sys.path.insert(0,str(REPOSITORY/'scripts/eval'))
                from eval_utils import extract_gsm8k_answer_number
                extractor=extract_gsm8k_answer_number
            self.extractor=extractor
        elif metric=='rougeLsum':
            import nltk
            from rouge_score import rouge_scorer
            if len(nltk.sent_tokenize('One sentence. Another sentence.'))!=2:
                raise ValueError('Missing fixed offline sentence tokenizer resources')
            self.scorer=rouge_scorer.RougeScorer(['rougeLsum'],use_stemmer=True,split_summaries=True)
        else:
            raise ValueError('Unknown metric')

    def score(self, prediction, reference):
        if self.metric=='exact_match':
            return float(self.extractor(prediction)==reference.strip('\n'))
        import nltk
        processed='\n'.join(nltk.sent_tokenize(prediction.replace('<n>',' ')))
        return self.scorer.score(reference,processed)['rougeLsum'].fmeasure
