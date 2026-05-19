/**
 * @file threads.h
 * @brief Cross-platform Threading Abstraction — Multi-threading + Concurrency
 *
 * TECHNIQUE: Multi-threading + Concurrency
 *
 * MinGW (the Windows GCC port) has broken std::thread support in versions
 * below 8. This header provides the SAME API using Win32 WINAPI threads on
 * Windows and pthreads on Linux/macOS. The abstraction is production-quality
 * — GCC 8+/Clang/MSVC all use std::thread in their toolchains.
 *
 * PUBLIC API (identical to std::thread / std::mutex):
 *
 *   Thread t([](){ ... });   // launch a thread
 *   t.join();                 // wait for completion
 *
 *   Mutex m;
 *   LockGuard<Mutex> guard(m);  // RAII lock
 *
 *   AtomicInt a{0};
 *   a.fetch_add(1);
 *   a.compare_exchange(expected, desired);
 */

#pragma once

#include <functional>
#include <stdexcept>

#ifdef _WIN32
  // ──────────────────────────────────────────────────────────────────────
  //  WINDOWS implementation using Win32 API
  // ──────────────────────────────────────────────────────────────────────
  // Require Windows Vista+ for full API
  #ifndef _WIN32_WINNT
  #define _WIN32_WINNT 0x0600
  #endif
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>

  // ── Thread ────────────────────────────────────────────────────────────
  class Thread {
  public:
      explicit Thread(std::function<void()> fn) : fn_(std::move(fn)), handle_(nullptr) {
          handle_ = CreateThread(
              nullptr,                    // security attributes
              0,                          // stack size (0 = default)
              &Thread::threadProc,        // start routine
              this,                       // parameter
              0,                          // creation flags
              nullptr                     // thread ID output
          );
          if (!handle_)
              throw std::runtime_error("CreateThread failed");
      }

      Thread(Thread&& other) noexcept : fn_(std::move(other.fn_)), handle_(other.handle_) {
          other.handle_ = nullptr;
      }

      // No copy
      Thread(const Thread&) = delete;
      Thread& operator=(const Thread&) = delete;

      ~Thread() {
          if (handle_) CloseHandle(handle_);
      }

      void join() {
          if (handle_) {
              WaitForSingleObject(handle_, INFINITE);
              CloseHandle(handle_);
              handle_ = nullptr;
          }
      }

      // Returns number of logical processors (hardware_concurrency equivalent)
      static int hardware_concurrency() {
          SYSTEM_INFO si;
          GetSystemInfo(&si);
          return static_cast<int>(si.dwNumberOfProcessors);
      }

  private:
      std::function<void()> fn_;
      HANDLE handle_;

      static DWORD WINAPI threadProc(LPVOID param) {
          auto* self = reinterpret_cast<Thread*>(param);
          self->fn_();
          return 0;
      }
  };

  // ── Mutex ──────────────────────────────────────────────────────────────
  class Mutex {
  public:
      Mutex()  { InitializeCriticalSection(&cs_); }
      ~Mutex() { DeleteCriticalSection(&cs_); }
      Mutex(const Mutex&) = delete;
      Mutex& operator=(const Mutex&) = delete;

      void lock()   { EnterCriticalSection(&cs_); }
      void unlock() { LeaveCriticalSection(&cs_); }

  private:
      CRITICAL_SECTION cs_;
  };

  // ── Condition Variable (spin-sleep for MinGW 6 compat) ────────────────
  class ConditionVariable {
  public:
      ConditionVariable()  {}
      ~ConditionVariable() {}

      void notify_all() {}
      void notify_one() {}

      template<typename Pred>
      void wait(Mutex& m, Pred pred) {
          while (!pred()) {
              m.unlock();
              Sleep(1);
              m.lock();
          }
      }
  };

  // ── Atomic int wrapper ────────────────────────────────────────────────
  struct AtomicInt {
      volatile LONG value_;
      explicit AtomicInt(int v = 0) : value_(v) {}
      int load()      const { return static_cast<int>(value_); }
      void store(int v)     { InterlockedExchange(&value_, v); }
      int fetch_add(int v)  { return static_cast<int>(InterlockedExchangeAdd(&value_, v)); }
      // CAS: returns true if exchange happened
      bool compare_exchange(int& expected, int desired) {
          LONG prev = InterlockedCompareExchange(&value_, desired, expected);
          if (prev == expected) return true;
          expected = prev;
          return false;
      }
      int operator++() { return fetch_add(1) + 1; }
  };

  struct AtomicBool {
      volatile LONG value_;
      explicit AtomicBool(bool v = false) : value_(v ? 1 : 0) {}
      bool load() const { return value_ != 0; }
      void store(bool v) { InterlockedExchange(&value_, v ? 1 : 0); }
  };

  // AtomicDouble — uses Mutex-protected spin for MinGW 6 (no 64-bit interlocked)
  struct AtomicDouble {
      double    value_;
      CRITICAL_SECTION cs_;
      AtomicDouble(double v = 0.0) : value_(v) { InitializeCriticalSection(&cs_); }
      ~AtomicDouble() { DeleteCriticalSection(&cs_); }
      double load() const  { return value_; }  // benign read race (double-word)
      void store(double v) {
          EnterCriticalSection(&cs_);
          value_ = v;
          LeaveCriticalSection(&cs_);
      }
      void fetch_add(double delta) {
          EnterCriticalSection(&cs_);
          value_ += delta;
          LeaveCriticalSection(&cs_);
      }
  };

#else
  // ──────────────────────────────────────────────────────────────────────
  //  POSIX (Linux / macOS) — std::thread works fine
  // ──────────────────────────────────────────────────────────────────────
  #include <thread>
  #include <mutex>
  #include <condition_variable>
  #include <atomic>

  using Thread = std::thread;

  class Mutex {
  public:
      void lock()   { m_.lock(); }
      void unlock() { m_.unlock(); }
      std::mutex& native() { return m_; }
  private:
      std::mutex m_;
  };

  class ConditionVariable {
  public:
      void notify_all() { cv_.notify_all(); }
      void notify_one() { cv_.notify_one(); }
      void wait(Mutex& m) {
          std::unique_lock<std::mutex> lk(m.native(), std::adopt_lock);
          cv_.wait(lk);
          lk.release();
      }
      template<typename Pred>
      void wait(Mutex& m, Pred pred) {
          std::unique_lock<std::mutex> lk(m.native(), std::adopt_lock);
          cv_.wait(lk, pred);
          lk.release();
      }
  private:
      std::condition_variable cv_;
  };

  struct AtomicInt {
      std::atomic<int> value_;
      explicit AtomicInt(int v = 0) : value_(v) {}
      int load()         const { return value_.load(); }
      void store(int v)        { value_.store(v); }
      int fetch_add(int v)     { return value_.fetch_add(v); }
      bool compare_exchange(int& e, int d) { return value_.compare_exchange_weak(e,d); }
      int operator++() { return value_.fetch_add(1) + 1; }
  };
  struct AtomicBool {
      std::atomic<bool> value_;
      explicit AtomicBool(bool v = false) : value_(v) {}
      bool load() const  { return value_.load(); }
      void store(bool v) { value_.store(v); }
  };
  struct AtomicDouble {
      std::atomic<double> value_;
      AtomicDouble(double v = 0.0) : value_(v) {}
      double load() const  { return value_.load(); }
      void store(double v) { value_.store(v); }
      void fetch_add(double delta) {
          double old = value_.load();
          while (!value_.compare_exchange_weak(old, old + delta));
      }
  };
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  RAII LockGuard — works with our portable Mutex
// ─────────────────────────────────────────────────────────────────────────────
template<typename MutexType>
class LockGuard {
public:
    explicit LockGuard(MutexType& m) : m_(m) { m_.lock(); }
    ~LockGuard()                              { m_.unlock(); }
    LockGuard(const LockGuard&) = delete;
private:
    MutexType& m_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Hardware concurrency helper (portable)
// ─────────────────────────────────────────────────────────────────────────────
inline int hardwareConcurrency() {
#ifdef _WIN32
    return Thread::hardware_concurrency();
#else
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return (n < 1) ? 1 : n;
#endif
}
