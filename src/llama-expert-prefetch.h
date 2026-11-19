#pragma once
#include "llama-expert-read-pool.h"
#include "llama-expert-profile.h"
#include "llama-expert-store.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>

// Bounded early reads. The owner retains the admitted arena and every
// task until all readers finish; no speculative slice publishes store metadata.
class early_lookahead {
public:
 static constexpr size_t candidates=8, projections=3, slots=candidates*projections;
 static constexpr size_t max_rows=8, max_ranks=32;
private:
 struct slice {
  early_lookahead * owner=nullptr;
  llama_expert_source::read_request request{};
  uint64_t begin=0,end=0;
  bool started=false,ok=false,used=false,measure=false;
 };
 struct bank {
  std::array<slice,slots> slices{};
  std::array<early_io::pool::task,slots> tasks{};
  const llama_expert_tensor_info * key=nullptr;
  size_t count=0;uint64_t wait_ns=0;
 };
 std::shared_ptr<llama_expert_source> source;
 llama_expert_store & store;
 const size_t limit;
 size_t stride;
 std::unique_ptr<uint8_t[]> data;
 // Destroyed after the banks have drained, before their arena is released.
 std::unique_ptr<early_io::pool> owned_readers;
 early_io::pool & readers;
 std::array<bank,2> banks{};
 const llama_expert_tensor_info * current=nullptr;
 bool healthy=true;
 void retire(bank & b,llama_expert_profile_data * p){
  if(!b.key)return;
  uint64_t blocked=0,bytes=0,first=UINT64_MAX,last=0,failures=0;
  for(size_t i=0;i<b.count;++i){
   readers.cancel(b.tasks[i]);
  }
  for(size_t i=0;i<b.count;++i){
   blocked+=readers.wait(b.tasks[i],p!=nullptr);
   auto & s=b.slices[i];
   if(s.started){first=std::min(first,s.begin);last=std::max(last,s.end);if(s.ok)bytes+=s.request.tensor->expert_bytes();else ++failures;}
  }
  if(p){
   p->next_prefetch_bytes+=bytes;p->next_prefetch_wait_ns+=blocked;p->next_prefetch_failures+=failures;
   if(bytes){++p->next_prefetch_batches;const auto span=last-first;p->next_prefetch_read_ns+=span;p->next_prefetch_hidden_ns+=span>blocked+b.wait_ns?span-blocked-b.wait_ns:0;}
  }
  if(failures)healthy=false;
  b=bank{};
 }
public:
 static size_t configured_candidates(){
  const char * value=std::getenv("ATOMIC_PREFETCH_CANDIDATES");
  if(!value||std::strcmp(value,"8")==0)return 8;
  if(std::strcmp(value,"4")==0)return 4;
  throw std::invalid_argument("ATOMIC_PREFETCH_CANDIDATES must be 4 or 8");
 }
 static size_t memory_bytes(size_t stride,size_t limit=8,bool shared=false){
  return sizeof(early_lookahead)+(shared?0:early_io::pool::memory_bytes())+2*projections*limit*stride+16*1024;
 }
 early_lookahead(std::shared_ptr<llama_expert_source> source,llama_expert_store & store,size_t stride,
                 size_t limit=8,early_io::pool * shared=nullptr):
  source(std::move(source)),store(store),limit(limit),stride(stride),data(new uint8_t[2*projections*limit*stride]),
  owned_readers(shared?nullptr:std::make_unique<early_io::pool>()),readers(shared?*shared:*owned_readers){
   if(limit!=4&&limit!=8)throw std::invalid_argument("prefetch capacity must be four or eight");
  }
 ~early_lookahead(){finish(nullptr);}
 early_io::pool * pool(){return &readers;}
 void finish(llama_expert_profile_data * p){for(auto & b:banks)retire(b,p);current=nullptr;}
 void prepare(const char * name,const int64_t * actual,size_t count,llama_expert_profile_data * p){
  current=&source->tensor(name);
  for(auto & b:banks){
   if(b.key!=current){retire(b,p);continue;}
   for(size_t i=0;i<b.count;++i){
    auto & s=b.slices[i];
    if(std::binary_search(actual,actual+count,s.request.expert))readers.promote(b.tasks[i]);
    else readers.cancel(b.tasks[i]);
   }
  }
 }
 size_t consume(const llama_expert_source::read_request & r,llama_expert_profile_data * p){
  for(auto & b:banks)for(size_t i=0;i<b.count;++i){
   auto & s=b.slices[i];
   if(s.used||r.tensor!=s.request.tensor||r.expert!=s.request.expert)continue;
   readers.promote(b.tasks[i]);
   const auto waited=readers.wait(b.tasks[i],p!=nullptr);
   b.wait_ns+=waited;
   if(p){p->next_prefetch_wait_ns+=waited;p->next_prefetch_slice_wait_ns+=waited;p->next_prefetch_slice_waits+=waited>0;}
   if(!s.started||!s.ok)return 0;
   const size_t bytes=r.tensor->expert_bytes();std::memcpy(r.destination,s.request.destination,bytes);s.used=true;
   return bytes;
  }
  return 0;
 }
 template<class Target>
 void fetch(const Target & next,const ggml_tensor * input,const ggml_tensor * ids,const ggml_tensor * scores,
            llama_expert_profile_data * p){
  if(!healthy||!next.router||!ids||!scores||!ggml_backend_buffer_is_host(ids->buffer)||!ggml_backend_buffer_is_host(scores->buffer))return;
  const size_t rows=ids->ne[1],ranks=ids->ne[0];
  if(rows!=size_t(input->ne[2])||!rows||rows>max_rows||!ranks||ranks>max_ranks||size_t(input->ne[0])!=next.width)return;
  llama_expert_timer timer(p?&p->next_prefetch_predict_ns:nullptr);
  bank * b=nullptr;size_t bank_index=0;
  for(size_t i=0;i<banks.size();++i)if(banks[i].key!=current){retire(banks[i],p);b=&banks[i];bank_index=i;break;}
  if(!b)return;
  struct rank {int64_t id;float weight;};
  std::array<rank,max_rows*max_ranks> order{};size_t n=0;
  const auto * selected=static_cast<const int32_t *>(ids->data);
  const auto * logits=static_cast<const float *>(scores->data);
  for(size_t row=0;row<rows;++row){
   const float * q=logits+row*ranks;const float largest=*std::max_element(q,q+ranks);float sum=0;
   for(size_t j=0;j<ranks;++j){if(!std::isfinite(q[j])||selected[row*ranks+j]<0||size_t(selected[row*ranks+j])>=next.experts)return;sum+=std::exp(q[j]-largest);}
   for(size_t j=0;j<ranks;++j){
    const auto id=selected[row*ranks+j];auto * end=order.data()+n;
    auto * found=std::find_if(order.data(),end,[&](const rank & r){return r.id==id;});
    if(found==end){*found={id,0};++n;}found->weight+=std::exp(q[j]-largest)/sum;
   }
  }
  std::sort(order.begin(),order.begin()+n,[](const rank & a,const rank & b){return a.weight==b.weight?a.id<b.id:a.weight>b.weight;});
  size_t chosen=0;
  for(size_t i=0;i<n&&chosen<limit;++i){
   const auto before=b->count;
   for(const auto * w:next.weights){
    if(store.contains(w->name,order[i].id))continue;
    const size_t at=b->count++;auto & s=b->slices[at];
    s.measure=p!=nullptr;s.owner=this;s.request={w,order[i].id,data.get()+(bank_index*projections*limit+at)*stride};
    b->tasks[at].context=&s;b->tasks[at].fn=[](void * ptr)noexcept{
     auto & s=*static_cast<slice *>(ptr);s.started=true;s.begin=s.measure?early_io::now():0;
     try{s.owner->source->read_experts(&s.request,1,true,true);s.ok=true;}catch(...){s.ok=false;}
     s.end=s.measure?early_io::now():0;
    };
   }
   chosen+=before!=b->count;
  }
  if(p)++p->next_prefetch_graph_calls;
  if(!b->count)return;
  b->key=next.weights[0];readers.submit(b->tasks.data(),b->count,false);
 }
};
