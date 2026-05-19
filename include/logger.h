/**
 * @file logger.h
 * @brief Thread-Safe Logger — Concurrency + Multi-threading
 *
 * TECHNIQUE: Concurrency, Multi-threading
 * When multiple worker threads train simultaneously they may all try to
 * print to the console at the same time, interleaving their output into
 * garbage.  This logger uses a std::mutex to serialise console writes so
 * each line comes out clean regardless of which thread calls it.
 *
 * Design pattern: Singleton — there is one global log object shared by
 * every thread.  std::mutex prevents simultaneous writes (a "race condition").
 */

#pragma once

// threads.h must come before logger body to get portable Mutex/LockGuard
// (avoids circular include by not including logger.h from threads.h)
#include "threads.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>      // std::put_time
#include <ctime>
#include <chrono>

// Severity levels — callers choose how important a message is
// Note: ERR instead of ERROR to avoid Windows.h macro conflict
enum class LogLevel { DEBUG, INFO, WARNING, ERR };

class Logger {
public:
    // ── Singleton accessor ────────────────────────────────────────────
    // Returns the one global Logger instance.  Constructed on first call.
    static Logger& instance() {
        static Logger log;  // thread-safe in C++11 and later (magic statics)
        return log;
    }

    // Open an optional log file in addition to console output
    void setLogFile(const std::string& path) {
        LockGuard<Mutex> guard(mutex_);       // RAII lock — portable
        file_.open(path, std::ios::app);      // append mode
    }

    // Set the minimum level to print (e.g. suppress DEBUG in production)
    void setLevel(LogLevel lvl) {
        LockGuard<Mutex> guard(mutex_);
        minLevel_ = lvl;
    }

    // Core log function — called by the convenience macros below
    void log(LogLevel lvl, const std::string& msg) {
        if (lvl < minLevel_) return;               // filter by level

        // Build the formatted line BEFORE locking — keeps lock duration short
        std::ostringstream oss;
        oss << "[" << timestamp() << "] [" << levelName(lvl) << "] " << msg;
        std::string line = oss.str();

        // TECHNIQUE: Concurrency — RAII LockGuard prevents simultaneous writes
        LockGuard<Mutex> guard(mutex_);
        std::cout << line << "\n";
        if (file_.is_open()) file_ << line << "\n";
    }

    // Convenience wrappers so callers write LOG_INFO("message") not log(INFO,...)
    void debug  (const std::string& m) { log(LogLevel::DEBUG,   m); }
    void info   (const std::string& m) { log(LogLevel::INFO,    m); }
    void warning(const std::string& m) { log(LogLevel::WARNING, m); }
    void error  (const std::string& m) { log(LogLevel::ERR,    m); }

private:
    Logger() : minLevel_(LogLevel::INFO) {}      // private — enforce singleton
    Logger(const Logger&)            = delete;   // no copy
    Logger& operator=(const Logger&) = delete;   // no assignment

    Mutex         mutex_;    // THE key to thread safety (portable Win32/pthread)
    std::ofstream file_;
    LogLevel     minLevel_;

    // Returns "HH:MM:SS" string for the current wall-clock time
    std::string timestamp() const {
        auto now  = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_info;
#ifdef _WIN32
#if defined(__MINGW32__) || defined(__MINGW64__)
        std::tm* tm_ptr = std::localtime(&t);
        if (tm_ptr) tm_info = *tm_ptr;
#else
        localtime_s(&tm_info, &t);
#endif
#else
        localtime_r(&t, &tm_info);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm_info, "%H:%M:%S");
        return oss.str();
    }

    static std::string levelName(LogLevel l) {
        switch (l) {
            case LogLevel::DEBUG:   return "DBG";
            case LogLevel::INFO:    return "INF";
            case LogLevel::WARNING: return "WRN";
            case LogLevel::ERR:     return "ERR";
        }
        return "???";
    }
};

// ── Convenient global macros ───────────────────────────────────────────────
// Usage: LOG_INFO("Epoch " + std::to_string(e) + " done")
#define LOG_DEBUG(msg)   Logger::instance().debug(msg)
#define LOG_INFO(msg)    Logger::instance().info(msg)
#define LOG_WARN(msg)    Logger::instance().warning(msg)
#define LOG_ERROR(msg)   Logger::instance().error(msg)
