from pathlib import Path
import argparse,json,os,socket,subprocess,time,urllib.request,statistics,hashlib
# This deliberately includes prompt-to-decode graph setup after slot restore.
# Zero demand reads does not make it a steady-state GPU/kernel microbenchmark.
a=argparse.ArgumentParser(description="Restored-slot hot short-request control including graph setup, one model and budget only.")
a.add_argument('--protocol',type=Path,required=True)
a.add_argument('--output',type=Path,required=True)
a.add_argument('--binary',type=Path,required=True)
a.add_argument('--mode',choices=['off','predictor','shared','combined'],default='off')
a.add_argument('--model',default='qwen35')
a.add_argument('--diag',action='store_true')
a.add_argument('--samples',type=int,default=32)
args=a.parse_args()
if args.samples < 1: a.error('--samples must be positive')
p=json.loads(args.protocol.read_text())
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
for line in subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines():
 if any(x in line for x in ['server-logits-probe -m ','llama-server -m ','splash-server ']):raise RuntimeError('concurrent inference: '+line)
def request(port,path,body=None,timeout=1800):
 q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(q,timeout=timeout) as r:return json.load(r)
env=dict(os.environ)
for key in list(env):
 if key.startswith(('LLAMA_EXPERIMENTAL_','ATOMIC_','LLAMA_EXPERT_','QWEN_BUDGET_')):env.pop(key)
env['LLAMA_MOE_STREAMING']='auto'
if args.mode in ['shared','combined']:env.update(ATOMIC_MOE_PUBLISH='shared',ATOMIC_MOE_PUBLISH_LAYER='all')
if args.mode in ['predictor','combined']:env['ATOMIC_MOE_QWEN_PREDICTOR']='1'
if args.diag:env['ATOMIC_ROUTE_HOT']=str(out/'reads.jsonl')
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
binary=args.binary.resolve()
command=[str(binary),'-m',p['models'][args.model],'-c','1024','-b','256','-ub','8','-np','1','-t','8','-tb','8','-ngl','99','-ctk','bf16','-ctv','bf16','-fit','off','--cache-ram','0','--ctx-checkpoints','0','--no-warmup','--slots','--slot-save-path',str(out),'--no-webui','--engine-budget-mib','18432','--host','127.0.0.1','--port',str(port)]
with binary.open('rb') as f:binary_hash=hashlib.file_digest(f,'sha256').hexdigest()
result=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},command=command,binary_sha256=binary_hash,status='started',samples=[],warmups=[])
with (out/'server.log').open('w') as log:
 process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+300
  while time.monotonic()<deadline:
   if process.poll() is not None:raise RuntimeError('load exit '+str(process.returncode))
   try:
    if request(port,'/health',timeout=1)['status']=='ok':break
   except OSError:time.sleep(.25)
  else:raise TimeoutError('loading')
  ids=p['control_ids'][args.model]
  first=request(port,'/completion',dict(prompt=ids,n_predict=1,temperature=0,seed=42,cache_prompt=False,ignore_eos=True,return_tokens=True))
  assert len(first['tokens'])==1
  result['prefix_ids']=ids+first['tokens']
  request(port,'/slots/0?action=save',dict(filename='prefix.slot'))
  expected=None
  for i in range(8+args.samples):
   request(port,'/slots/0?action=restore',dict(filename='prefix.slot'))
   r=request(port,'/completion',dict(prompt=result['prefix_ids'],n_predict=2,temperature=0,seed=42,cache_prompt=True,ignore_eos=True,return_tokens=True))
   assert r['timings']['cache_n']==246 and r['timings']['prompt_n']==1,r['timings']
   assert len(r['tokens'])==2 and r['timings']['predicted_n']==2
   if expected is None:expected=r['tokens']
   assert r['tokens']==expected
   result['warmups' if i<8 else 'samples'].append(r['timings'])
   (out/'result.json').write_text(json.dumps(result,indent=2))
  result['tokens']=expected
  result['median_decode_ms']=statistics.median(x['predicted_ms'] for x in result['samples'])
  result['status']='ok'
  if args.diag:
   rows=[json.loads(x) for x in (out/'reads.jsonl').read_text().splitlines()]
   assert len(rows)==8+args.samples
   result['reads']=rows
   assert all(x['moe_calls']>0 for x in rows)
   assert all(x['logical_read_bytes']==0 for x in rows[8:]),rows[8:]
  print(out.name,result['median_decode_ms'],'ms',flush=True)
 except BaseException as e:result.update(status='failed',error=repr(e));raise
 finally:
  if process.poll() is None:
   process.terminate()
   try:process.wait(timeout=15)
   except subprocess.TimeoutExpired:process.kill();process.wait()
  (out/'result.json').write_text(json.dumps(result,indent=2))
