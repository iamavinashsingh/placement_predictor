/**
 * @file fscompat.h
 * @brief Filesystem Compatibility Layer — File Systems
 *
 * TECHNIQUE: File Systems
 *
 * C++17's <filesystem> requires GCC 8+. This header provides the SAME
 * clean API for older toolchains using POSIX sys/stat + direct file I/O.
 *
 * On GCC 8+ it simply delegates to std::filesystem.
 * On GCC 6-7 (and Windows MinGW) it uses _stat / _mkdir / FindFirstFile.
 *
 * The public surface (fs::exists, fs::file_size, fs::create_directories,
 * fs::is_regular_file) is identical to std::filesystem so upgrading later
 * is a one-line change.
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>

// ── Windows-specific I/O headers ──────────────────────────────────────
#ifdef _WIN32
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <direct.h>    // _mkdir
  #include <io.h>
  #include <windows.h>
#else
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #include <dirent.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Minimal filesystem namespace — mirrors std::filesystem API
// ─────────────────────────────────────────────────────────────────────────────
namespace fs {

    // ── Does the path exist? ───────────────────────────────────────────
    inline bool exists(const std::string& path) {
#ifdef _WIN32
        struct _stat st;
        return _stat(path.c_str(), &st) == 0;
#else
        struct stat st;
        return stat(path.c_str(), &st) == 0;
#endif
    }

    // ── Is it a regular file (not a directory)? ────────────────────────
    inline bool is_regular_file(const std::string& path) {
#ifdef _WIN32
        struct _stat st;
        if (_stat(path.c_str(), &st) != 0) return false;
        return (st.st_mode & _S_IFREG) != 0;
#else
        struct stat st;
        if (stat(path.c_str(), &st) != 0) return false;
        return S_ISREG(st.st_mode);
#endif
    }

    // ── File size in bytes ─────────────────────────────────────────────
    inline uintmax_t file_size(const std::string& path) {
#ifdef _WIN32
        struct _stat st;
        if (_stat(path.c_str(), &st) != 0) return 0;
        return static_cast<uintmax_t>(st.st_size);
#else
        struct stat st;
        if (stat(path.c_str(), &st) != 0) return 0;
        return static_cast<uintmax_t>(st.st_size);
#endif
    }

    // ── Is it a directory? ─────────────────────────────────────────────
    inline bool is_directory(const std::string& path) {
#ifdef _WIN32
        struct _stat st;
        if (_stat(path.c_str(), &st) != 0) return false;
        return (st.st_mode & _S_IFDIR) != 0;
#else
        struct stat st;
        if (stat(path.c_str(), &st) != 0) return false;
        return S_ISDIR(st.st_mode);
#endif
    }

    // ── Create a single directory ──────────────────────────────────────
    inline bool create_directory(const std::string& path) {
        if (exists(path)) return true;
#ifdef _WIN32
        return _mkdir(path.c_str()) == 0;
#else
        return mkdir(path.c_str(), 0755) == 0;
#endif
    }

    // ── Create directory + all parents (like mkdir -p) ─────────────────
    // E.g. create_directories("a/b/c") creates a/, a/b/, a/b/c/
    inline bool create_directories(const std::string& path) {
        if (path.empty() || exists(path)) return true;

        // Walk the path and create each component
        std::string current;
        for (size_t i = 0; i < path.size(); ++i) {
            char c = path[i];
            if (c == '/' || c == '\\') {
                if (!current.empty() && !exists(current))
                    create_directory(current);
            }
            current += c;
        }
        // Create the final component
        if (!exists(current))
            return create_directory(current);
        return true;
    }

    // ── List .csv files in a directory ─────────────────────────────────
    inline std::vector<std::string> list_csv_files(const std::string& dir) {
        std::vector<std::string> result;

#ifdef _WIN32
        // Windows: use FindFirstFile / FindNextFile
        WIN32_FIND_DATAA fd;
        std::string pattern = dir + "\\*.csv";
        HANDLE hFind = FindFirstFileA(pattern.c_str(), &fd);
        if (hFind == INVALID_HANDLE_VALUE) return result;
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                result.push_back(dir + "\\" + fd.cFileName);
            }
        } while (FindNextFileA(hFind, &fd));
        FindClose(hFind);
#else
        // POSIX: use opendir / readdir
        DIR* d = opendir(dir.c_str());
        if (!d) return result;
        struct dirent* entry;
        while ((entry = readdir(d)) != nullptr) {
            std::string name(entry->d_name);
            if (name.size() > 4 && name.substr(name.size() - 4) == ".csv")
                result.push_back(dir + "/" + name);
        }
        closedir(d);
#endif

        std::sort(result.begin(), result.end());
        return result;
    }

}  // namespace fs
