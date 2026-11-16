#!/usr/bin/env python3
# One selected HTTP profile; never invoked by aggregate verification.
from pathlib import Path
from contextlib import contextmanager
import argparse, json, os, subprocess, time, urllib.request, urllib.error, re

parser=argparse.ArgumentParser(description="Targeted fixed-budget LFM2.5-24B CUDA server qualification")
parser.add_argument("--server",type=Path,required=True)
parser.add_argument("--model",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True,help="Empty directory for logs, states and results")
args=parser.parse_args()
work=args.output.resolve()
if work.exists() and any(work.iterdir()): parser.error("output directory must be empty")
work.mkdir(parents=True,exist_ok=True)
model=args.model.resolve()
server_binary=args.server.resolve()
url='http://127.0.0.1:19149'
prompt='Finish this sentence: The capital of France is'
results={}
def req(route,body=None,timeout=120):
 data=None if body is None else json.dumps(body).encode()
 return urllib.request.urlopen(urllib.request.Request(url+route,data=data,headers={'Content-Type':'application/json'}),timeout=timeout)
def call(route,body=None):
 with req(route,body) as r:return json.load(r)
def completion(text=prompt,**extra):
 return call('/completion',dict(prompt=text,n_predict=4,temperature=0,seed=42,id_slot=0,cache_prompt=False,return_tokens=True,**extra))
def metrics(name):
 with req('/metrics') as r:s=r.read().decode()
 (work/(name+'.metrics')).write_text(s)
 d={k:int(v) for k,v in re.findall(r'llamacpp:engine_memory_(\w+) ([0-9]+)\n',s)}
 for domain,cap in [('ram',4096*1024**2),('vram',2048*1024**2)]:
  assert d[domain+'_applied_bytes']==cap,d
  assert d[domain+'_used_bytes']<=cap and d[domain+'_high_water_bytes']<=cap,d
 return d
@contextmanager
def server(name,streaming):
 slots=work/name;slots.mkdir(exist_ok=True)
 cmd=[str(server_binary),'-m',str(model),'--device','CUDA0','-ngl','14','-c','256','-b','256','-ub','8','-np','1','-t','4','-tb','4','-fit','off','--no-warmup','--cache-ram','0','--ctx-checkpoints','0','--metrics','--slots','--slot-save-path',str(slots),'--host','127.0.0.1','--port','19149']
 if streaming:cmd+=['--engine-ram-mib','4096','--engine-vram-mib','2048']
 env=os.environ.copy();env.pop('GGML_CUDA_ENABLE_UNIFIED_MEMORY',None)
 with (work/(name+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
  try:
   deadline=time.monotonic()+120
   while time.monotonic()<deadline:
    if p.poll() is not None:raise RuntimeError(f'{name} exited {p.returncode}: '+(work/(name+'.log')).read_text()[-2500:])
    try:
     if call('/health').get('status')=='ok':break
    except (OSError,urllib.error.URLError):time.sleep(.2)
   else:raise TimeoutError(name)
   yield slots
  finally:
   p.terminate()
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
def save(name):return call('/slots/0?action=save',{'filename':name})
try:
 with server('ordinary',False) as slots:
  results['ordinary']=completion();save('initial.slot')
  with req('/metrics') as r:assert 'llamacpp:engine_memory_' not in r.read().decode()
  print('ordinary IDs',results['ordinary']['tokens'],flush=True)
 with server('streaming',True) as slots:
  results['first']=completion();save('initial.slot')
  assert results['first']['tokens']==results['ordinary']['tokens'],(results['first'],results['ordinary'])
  assert (slots/'initial.slot').read_bytes()==(work/'ordinary/initial.slot').read_bytes(),'ordinary/streaming slot bytes differ'
  results['initial_metrics']=metrics('initial')
  results['repeat']=completion();assert results['repeat']['tokens']==results['first']['tokens']
  save('repeat.slot');assert (slots/'initial.slot').read_bytes()==(slots/'repeat.slot').read_bytes()
  text=results['first']['content'];stop='Paris' if 'Paris' in text else text.strip()[:3]
  assert stop
  results['stop']=call('/completion',dict(prompt=prompt,n_predict=32,temperature=0,seed=42,id_slot=0,cache_prompt=False,return_tokens=True,stop=[stop]))
  assert results['stop']['stop_type']=='word' and stop not in results['stop']['content'],results['stop']
  results['stop_metrics']=metrics('stop')
  results['chat']=call('/v1/chat/completions',{'messages':[{'role':'user','content':'What is the capital of France? Answer with the city name only.'}],'max_tokens':128,'temperature':0,'seed':42})
  assert results['chat']['choices'][0]['finish_reason']=='stop',results['chat']
  results['chat_metrics']=metrics('chat')
  with req('/completion',dict(prompt=prompt,n_predict=128,temperature=0,id_slot=0,cache_prompt=False,ignore_eos=True,stream=True,return_tokens=True)) as r:
   for line in r:
    if line.startswith(b'data:'):
     event=json.loads(line[5:])
     if event.get('content') and not event.get('stop'):
      results['cancel_first']=event;break
   else:raise AssertionError('no SSE token')
  deadline=time.monotonic()+30
  while time.monotonic()<deadline:
   if not call('/slots')[0]['is_processing']:break
   time.sleep(.1)
  else:raise AssertionError('slot stayed busy after disconnect')
  results['after_cancel']=completion();assert results['after_cancel']['tokens']==results['first']['tokens']
  results['cancel_metrics']=metrics('cancel')
  save('before_restore.slot')
  follow=prompt+results['first']['content']
  body=dict(prompt=follow,n_predict=4,temperature=0,seed=42,id_slot=0,cache_prompt=True,return_tokens=True)
  results['continuation']=call('/completion',body);save('continued.slot')
  results['erase']=call('/slots/0?action=erase',{})
  results['restore']=call('/slots/0?action=restore',{'filename':'before_restore.slot'})
  save('restored.slot');assert (slots/'restored.slot').read_bytes()==(slots/'before_restore.slot').read_bytes()
  results['restored_continuation']=call('/completion',body);save('restored_continued.slot')
  assert results['continuation']['tokens']==results['restored_continuation']['tokens']
  assert (slots/'continued.slot').read_bytes()==(slots/'restored_continued.slot').read_bytes()
  assert results['restored_continuation']['timings']['cache_n']>0
  results['restore_metrics']=metrics('restore')
  print('streaming cases passed', {k:v for k,v in results.items() if k.endswith('metrics')},flush=True)
finally:
 (work/'server-results.json').write_text(json.dumps(results,indent=2))
