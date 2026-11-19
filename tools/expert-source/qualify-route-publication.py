from pathlib import Path
import argparse,ctypes,hashlib,json,os,re,socket,subprocess,time,urllib.request
a=argparse.ArgumentParser(description="One explicitly requested Metal decode qualification case; never runs a matrix.")
a.add_argument('--protocol',type=Path,required=True)
a.add_argument('--output',type=Path,required=True)
a.add_argument('--binary',type=Path,required=True)
a.add_argument('--model',default='ornith122')
a.add_argument('--mode',choices=['off','event','shared','predictor','combined'],default='off')
a.add_argument('--layers',choices=['0','all'],default='0')
a.add_argument('--prompt',default='control')
a.add_argument('--diag',action='store_true',help='Requires the companion instrumented server build.')
a.add_argument('--draft-model',type=Path)
a.add_argument('--repeat',type=int,default=1)
args=a.parse_args()
if args.repeat < 1: a.error('--repeat must be positive')
if args.diag and args.repeat != 1: a.error('diagnostic captures require one request per output directory')
p=json.loads(args.protocol.read_text())
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
case_name=out.name
def sha(f):
 with Path(f).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def request(port,path,body=None,timeout=1800):
 q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(q,timeout=timeout)
libproc=ctypes.CDLL('/usr/lib/libproc.dylib')
def read_bytes(pid):
 data=(ctypes.c_uint64*64)()
 if libproc.proc_pid_rusage(pid,4,ctypes.byref(data)):return None
 return data[18] # 16-byte UUID, followed by sixteen fields before diskio_bytesread
# Refuse concurrent known inference workloads. This is a guard, not a GPU-wide profiler.
processes=subprocess.check_output(['ps','-axo','pid,command'],text=True)
for line in processes.splitlines():
 if any(x in line for x in ['server-logits-probe -m ','llama-server -m ','splash-server ']):raise RuntimeError('concurrent inference: '+line)
binary=args.binary.resolve()
env=dict(os.environ)
for key in list(env):
 if key.startswith(('LLAMA_EXPERIMENTAL_','ATOMIC_','LLAMA_EXPERT_','QWEN_BUDGET_')):env.pop(key)
env['LLAMA_MOE_STREAMING']='auto';env['ATOMIC_MOE_PUBLISH_LAYER']=args.layers
if args.mode in ['event','shared','combined']:env['ATOMIC_MOE_PUBLISH']='shared' if args.mode=='combined' else args.mode
if args.mode in ['predictor','combined']:env['ATOMIC_MOE_QWEN_PREDICTOR']='1'
if args.diag:env.update(ATOMIC_WINDOW_LOGITS=str(out/'logits.f32'),ATOMIC_ROUTE_PROFILE='1')
if args.draft_model:env['LLAMA_EXPERIMENTAL_BUDGET_DFLASH2']='1'
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
command=[str(binary),'-m',p['models'][args.model],'-c','1024','-b','256','-ub','8','-np','1','-t','8','-tb','8','-ngl','99','-ctk','bf16','-ctv','bf16','-fit','off','--cache-ram','0','--ctx-checkpoints','0','--no-warmup','--metrics','--slots','--slot-save-path',str(out),'--no-webui','--engine-budget-mib','18432','--host','127.0.0.1','--port',str(port)]
if args.draft_model:command += ['--spec-type','draft-dflash','-md',str(args.draft_model),'-ngld','99','--spec-draft-n-max','5']
result=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},command=command,binary_sha256=sha(binary),requests=[],status='started')
(out/'result.json').write_text(json.dumps(result,indent=2))
with (out/'server.log').open('w') as log:
 process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+300
  while time.monotonic()<deadline:
   if process.poll() is not None:raise RuntimeError('load exit '+str(process.returncode))
   try:
    with request(port,'/health',timeout=1) as r:
     if r.status==200:break
   except OSError:time.sleep(.25)
  else:raise TimeoutError('loading')
  # Texts were frozen before any performance measurement. Tokenization is fixed,
  # first 246 IDs; no candidate or group tuning from these workloads.
  if args.prompt=='control':ids=p['control_ids'][args.model]
  else:
   with request(port,'/tokenize',dict(content=p['texts'][args.prompt],add_special=True)) as r:ids=json.load(r)['tokens'][:246]
  assert len(ids)==246;result['prompt_ids']=ids
  with request(port,'/completion',dict(prompt=ids[:15],n_predict=8,temperature=0,seed=42,cache_prompt=False,ignore_eos=True)) as r:json.load(r)
  for repeat in range(args.repeat):
   first=None;first_bytes=None;final=None;output=[];before=read_bytes(process.pid);t0=time.perf_counter()
   with request(port,'/completion',dict(prompt=ids,n_predict=64,temperature=0,seed=42,cache_prompt=False,ignore_eos=True,stream=True,return_tokens=True,return_progress=True)) as r:
    for line in r:
     now=time.perf_counter()
     if not line.startswith(b'data:'):continue
     raw=line[5:].strip()
     if raw==b'[DONE]':break
     item=json.loads(raw);assert not item.get('error'),item
     if 'prompt_progress' in item:
      (out/'progress.json').write_text(json.dumps(dict(elapsed_s=now-t0,progress=item['prompt_progress'])))
      continue
     if item.get('stop'):final=item
     elif item.get('tokens'):
      output+=item['tokens']
      if first is None:first=now-t0;first_bytes=read_bytes(process.pid);print(case_name,'TTFT',round(first,3),flush=True)
   after=read_bytes(process.pid)
   assert final and first and len(output)==64,(len(output),final)
   assert final['timings']['prompt_n']==246 and final['timings']['cache_n']==0 and not final.get('truncated'),final['timings']
   with request(port,'/metrics') as r:metrics=r.read().decode()
   memory={k:int(float(v)) for k,v in re.findall(r'^llamacpp:engine_memory_(\w+)\s+(\S+)$',metrics,re.M)}
   assert memory['budget_applied_bytes']==18*1024**3 and memory['budget_high_water_bytes']<=18*1024**3,memory
   record=dict(ttft_s=first,timings=final['timings'],output_ids=output,memory=memory,physical_request_bytes=None if before is None else after-before,physical_decode_bytes=None if first_bytes is None else after-first_bytes)
   if args.diag:
    record['logits_sha256']=sha(out/'logits.f32');record['logits_bytes']=(out/'logits.f32').stat().st_size
    if (out/'logits.f32.profile.json').exists():record['profile']=json.loads((out/'logits.f32.profile.json').read_text())
    for action,body in [('save',{'filename':'result.slot'}),('erase',{}),('restore',{'filename':'result.slot'}),('save',{'filename':'restored.slot'})]:
     with request(port,'/slots/0?action='+action,body) as r:json.load(r)
    record['slot_sha256']=sha(out/'result.slot');assert record['slot_sha256']==sha(out/'restored.slot')
    record['slot_roundtrip']=True
    # Continue twice from precisely the same restored slot, including the last
    # sampled token (which has not yet been ingested into the saved state).
    (out/'logits.f32').rename(out/'request-logits.f32')
    continuations=[]
    for pass_id in range(2):
     with request(port,'/slots/0?action=restore',{'filename':'result.slot'}) as r:json.load(r)
     with request(port,'/completion',dict(prompt=ids+output,n_predict=16,temperature=0,seed=42,cache_prompt=True,ignore_eos=True,return_tokens=True)) as r:cont=json.load(r)
     with request(port,'/slots/0?action=save',{'filename':f'continuation-{pass_id}.slot'}) as r:json.load(r)
     continuations.append(dict(tokens=cont['tokens'],logits_sha256=sha(out/'logits.f32'),slot_sha256=sha(out/f'continuation-{pass_id}.slot')))
    assert continuations[0]==continuations[1],continuations
    record['continuation']=continuations[0]

   result['requests'].append(record)
   (out/'result.json').write_text(json.dumps(result,indent=2))
   print(case_name,'decode',round(final['timings']['predicted_per_second'],4),flush=True)
  result['status']='ok'
 except BaseException as e:
  result.update(status='failed',error=repr(e));raise
 finally:
  if process.poll() is None:
   process.terminate()
   try:process.wait(timeout=15)
   except subprocess.TimeoutExpired:process.kill();process.wait()
  result['rollback_events']=re.findall(r'^ROUTE_ROLLBACK (\d+) (\d+)$',(out/'server.log').read_text(),re.M)
  result['selected']=re.findall(r'^expert (?:acceleration|route publication):.*$',(out/'server.log').read_text(),re.M)
  (out/'result.json').write_text(json.dumps(result,indent=2))
