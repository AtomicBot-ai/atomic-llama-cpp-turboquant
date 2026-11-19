#pragma once
#ifdef __APPLE__
#include "../src/llama-expert-lookahead.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include <filesystem>
#include <vector>
#include <iostream>
#include <unistd.h>
static void test_portable_prefetch(testing & t,size_t limit){
 auto check=[&](bool ok,const char * message){t.assert_true(message,ok);if(!ok)throw std::runtime_error(message);};
 {
  early_io::pool pool;
  std::array<early_io::pool::task,3> t{};
  std::array<std::atomic<unsigned>,3> calls{};
  for(size_t i=0;i<t.size();++i){t[i].context=&calls[i];t[i].fn=[](void * p)noexcept{++*static_cast<std::atomic<unsigned> *>(p);};}
  {
   early_io::pool::pause pause(&pool);
   pool.submit(t.data(),t.size(),false);
   pool.promote(t[1]);pool.wait(t[1]);
   check(calls[1]==1,"promoted demand executes while background paused");
   check(calls[0]==0&&calls[2]==0,"background did not steal demand priority");
   check(pool.cancel(t[0]),"queued background cancelled");
   check(!pool.cancel(t[1]),"completed task cannot be cancelled");
  }
  for(auto & task:t)pool.wait(task);
  check(calls[0]==0&&calls[1]==1&&calls[2]==1,"each noncancelled task executes exactly once");
 }
 ggml_backend_load_all();ggml_backend_ptr cpu(ggml_backend_init_by_name("CPU",nullptr));check(bool(cpu),"CPU backend");
 const auto path=std::filesystem::temp_directory_path()/("fixture-"+std::to_string(getpid())+".gguf");
 struct cleanup{std::filesystem::path p;~cleanup(){std::filesystem::remove(p);}}remove{path};
 ggml_context_ptr weights(ggml_init({2*1024*1024,nullptr,false}));gguf_context_ptr meta(gguf_init_empty());
 gguf_set_val_str(meta.get(),"general.architecture","qwen4exp");
 std::array<ggml_tensor *,6> tensors{};
 const char * names[]={"blk.0.ffn_up_exps.weight","blk.0.ffn_gate_exps.weight","blk.0.ffn_down_exps.weight","blk.1.ffn_up_exps.weight","blk.1.ffn_gate_exps.weight","blk.1.ffn_down_exps.weight"};
 for(size_t k=0;k<tensors.size();++k){auto * t=tensors[k]=ggml_new_tensor_3d(weights.get(),GGML_TYPE_F32,64,64,16);ggml_set_name(t,names[k]);auto * values=static_cast<float *>(t->data);for(size_t i=0;i<ggml_nbytes(t)/4;++i)values[i]=float((i+13*k)%97)/128;gguf_add_tensor(meta.get(),t);}
 check(gguf_write_to_file(meta.get(),path.c_str(),false),"GGUF fixture");
 auto source=std::make_shared<llama_expert_source>(std::vector<std::string>{path});
 auto budget=std::make_shared<llama_memory_budget>(64*1024*1024,64*1024*1024,true);
 llama_expert_store store(source,budget,cpu.get(),llama_memory_domain::host,64);
 ggml_context_ptr gc(ggml_init({65536,nullptr,true}));
 auto * x=ggml_new_tensor_3d(gc.get(),GGML_TYPE_F32,64,1,8);
 auto * ids=ggml_new_tensor_2d(gc.get(),GGML_TYPE_I32,3,8);
 auto * scores=ggml_new_tensor_2d(gc.get(),GGML_TYPE_F32,3,8);
 ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(gc.get(),cpu.get()));
 std::array<int32_t,24> selected{};std::array<float,24> values{};
 for(size_t row=0;row<8;++row)for(size_t rank=0;rank<3;++rank){selected[3*row+rank]=(row+rank)%8;values[3*row+rank]=float(3-rank);}
 ggml_backend_tensor_set(ids,selected.data(),0,sizeof(selected));ggml_backend_tensor_set(scores,values.data(),0,sizeof(values));
 llama_expert_lookahead::target target;target.router=x->data;target.width=64;target.experts=16;
 for(size_t i=0;i<3;++i)target.weights[i]=&source->tensor(names[3+i]);
 const auto before=budget->snapshot(llama_memory_domain::host).used();
 {
  auto charge=budget->reserve(llama_memory_domain::host,llama_memory_category::io,llama_expert_lookahead::working_bytes(64*64*4,true,limit));check(bool(charge),"lookahead arena admitted");
  llama_expert_lookahead ahead(source,store,std::move(charge),64*64*4,true,true,limit);
  llama_expert_profile_data profile{};ahead.begin_graph(true);
  const int64_t current[]={0};ahead.prepare_early(names[0],current,1,&profile);
  ahead.fetch_early(target,x,ids,scores,&profile);
  const int64_t actual[]={0,1,2,3,4,5,6,7};ahead.prepare_early(names[3],actual,8,&profile);
  size_t copied_slices=0;int64_t first_selected=-1;
  std::vector<float> output(64*64);
  for(auto id:actual)for(size_t p=0;p<3;++p){
   llama_expert_source::read_request request{target.weights[p],id,output.data()};
   const auto copied=ahead.consume(request,&profile);
   if(!copied)continue;
   ++copied_slices;if(first_selected<0)first_selected=id;
   check(copied==output.size()*4,"needed speculative slice available");
   check(std::memcmp(output.data(),static_cast<const float *>(tensors[3+p]->data)+id*output.size(),output.size()*4)==0,"prefetched bytes exact");
   check(ahead.consume(request,&profile)==0,"speculative slice consumed once");
  }
  ahead.end_graph(&profile);
  check(copied_slices==3*limit,"each selected candidate supplies exactly three slices");
  check(profile.next_prefetch_bytes==3*limit*output.size()*4,"only admitted candidate slices read");
  const size_t fixed=sizeof(early_lookahead)+early_io::pool::memory_bytes()+16*1024;
  check(early_lookahead::memory_bytes(output.size()*4,limit)==fixed+2*3*limit*output.size()*4,"both generations shrink with candidate limit");
  check(budget->snapshot(llama_memory_domain::host).used()<=64*1024*1024,"arena stays under budget");
  std::filesystem::resize_file(path,0);
  ahead.begin_graph(true);ahead.prepare_early(names[0],current,1,&profile);ahead.fetch_early(target,x,ids,scores,&profile);
  ahead.prepare_early(names[3],actual,8,&profile);
  llama_expert_source::read_request failed{target.weights[0],first_selected,output.data()};
  check(ahead.consume(failed,&profile)==0,"failed speculative read falls back without publication");
  ahead.end_graph(&profile);check(profile.next_prefetch_failures>0,"speculative failures recorded");
 }
 check(budget->snapshot(llama_memory_domain::host).used()==before,"arena and reader-stack reservations released after joins");
}
#endif
