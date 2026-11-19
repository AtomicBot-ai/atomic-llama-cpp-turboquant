#pragma once
#include <array>
#include <condition_variable>
#include <mutex>
#ifdef __APPLE__
#include <pthread.h>
#include <pthread/qos.h>
#endif
#include <stdexcept>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <cstdio>

namespace early_io {
inline uint64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
class pool {
public:
 static constexpr size_t workers=4, stack_bytes=256*1024, guard_bytes=16*1024;
 struct task {
   using function=void(*)(void *) noexcept;
   enum status {fresh,queued,running,done,cancelled};
   task * next=nullptr; function fn=nullptr; void * context=nullptr;
   status state=fresh; bool urgent=false;
 };
private:
#ifdef __APPLE__
 std::array<pthread_t,workers> threads{};size_t started=0;
#endif
 std::mutex mutex;std::condition_variable changed;
 task * high_head=nullptr,* high_tail=nullptr,* low_head=nullptr,* low_tail=nullptr;
 size_t pauses=0;bool stopping=false;
 static void append(task * & head,task * & tail,task & t){t.next=nullptr;if(tail)tail->next=&t;else head=&t;tail=&t;}
 static task * pop(task * & head,task * & tail){auto * t=head;head=t->next;if(!head)tail=nullptr;t->next=nullptr;return t;}
 static void unlink(task * & head,task * & tail,task & t){task * previous=nullptr;for(auto * p=head;p;p=p->next){if(p==&t){if(previous)previous->next=p->next;else head=p->next;if(tail==p)tail=previous;p->next=nullptr;return;}previous=p;}std::terminate();}
 void loop(){
#ifdef __APPLE__
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED,0);
#endif
  std::unique_lock<std::mutex> lock(mutex);
  for(;;){
   changed.wait(lock,[&]{return stopping||high_head||(!pauses&&low_head);});
   if(stopping&&!high_head&&!low_head)return;
   if(!high_head&&(pauses||!low_head))continue;
   auto * t=high_head?pop(high_head,high_tail):pop(low_head,low_tail);
   t->state=task::running;
   lock.unlock();t->fn(t->context);lock.lock();
   t->state=task::done;changed.notify_all();
  }
 }
public:
 pool(){
#ifdef __APPLE__
  pthread_attr_t attr;pthread_attr_init(&attr);
  const int size_error=pthread_attr_setstacksize(&attr,stack_bytes);
  const int guard_error=pthread_attr_setguardsize(&attr,guard_bytes);
  if(size_error||guard_error){pthread_attr_destroy(&attr);throw std::runtime_error("reader stack attributes");}
  for(;started<workers;++started){
   if(pthread_create(&threads[started],&attr,[](void * p)->void *{static_cast<pool *>(p)->loop();return nullptr;},this)){
    pthread_attr_destroy(&attr);{std::lock_guard<std::mutex> l(mutex);stopping=true;}changed.notify_all();
    for(size_t i=0;i<started;++i)pthread_join(threads[i],nullptr);
    throw std::runtime_error("cannot create priority readers");
   }
  }
  pthread_attr_destroy(&attr);
#else
  throw std::runtime_error("priority readers require Metal");
#endif
 }
 ~pool(){
  {std::lock_guard<std::mutex> l(mutex);stopping=true;pauses=0;}
  changed.notify_all();
#ifdef __APPLE__
  for(size_t i=0;i<started;++i)pthread_join(threads[i],nullptr);
#endif
 }
 pool(const pool &)=delete;pool & operator=(const pool &)=delete;
 static constexpr size_t memory_bytes(){return sizeof(pool)+workers*(stack_bytes+guard_bytes);}
 void submit(task * tasks,size_t count,bool urgent){
  {std::lock_guard<std::mutex> l(mutex);for(size_t i=0;i<count;++i){auto & t=tasks[i];if(t.state!=task::fresh||!t.fn)std::terminate();t.state=task::queued;t.urgent=urgent;append(urgent?high_head:low_head,urgent?high_tail:low_tail,t);}}
  changed.notify_all();
 }
 bool promote(task & t){
  {std::lock_guard<std::mutex> l(mutex);if(t.state!=task::queued||t.urgent)return false;
   unlink(low_head,low_tail,t);t.urgent=true;append(high_head,high_tail,t);}
  changed.notify_all();return true;
 }
 bool cancel(task & t){
  std::lock_guard<std::mutex> l(mutex);if(t.state!=task::queued)return false;
  unlink(t.urgent?high_head:low_head,t.urgent?high_tail:low_tail,t);t.state=task::cancelled;changed.notify_all();return true;
 }
 bool complete(task & t){std::lock_guard<std::mutex> l(mutex);return t.state==task::done||t.state==task::cancelled||t.state==task::fresh;}
 uint64_t wait(task & t,bool measure=false){
  std::unique_lock<std::mutex> l(mutex);if(t.state==task::done||t.state==task::cancelled||t.state==task::fresh)return 0;
  const auto begin=measure?now():0;changed.wait(l,[&]{return t.state==task::done||t.state==task::cancelled;});return measure?now()-begin:0;
 }
 struct pause {
  pool * owner;
  explicit pause(pool * p):owner(p){if(p){std::lock_guard<std::mutex> l(p->mutex);++p->pauses;}}
  ~pause(){if(owner){{std::lock_guard<std::mutex> l(owner->mutex);--owner->pauses;}owner->changed.notify_all();}}
 };
};
}
