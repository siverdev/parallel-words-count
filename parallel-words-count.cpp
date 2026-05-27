/*
    ============================================================
    BUILD
    ============================================================

    Enable MPI/OpenMP/Sequential:
    -----------
    Uncomment:
        #define USE_MPI /
        #define USE_OPENMP /
        #define USE_SEQ

    ============================================================
    RUN
    ============================================================

    OpenMP:
    --------
    .\x64\Release_OMP\parallel-words-count-omp.exe <threads> <path1> [path2] ...

    OpenMP benchmarking
    --------------------
    .\benchmarking\scripts\benchmark_omp.ps1 `
    -exe ".\x64\Release_OMP\parallel-words-count-omp.exe" `
    -min <threads> `
    -max <threads> `
    -runs <runs> `
    <path>

    MPI:
    ----
    mpiexec -np <processes> .\x64\Release_MPI\parallel-words-count-mpi.exe  <path1> [path2] ...

	MPI benchmarking
    -----------------
    .\benchmarking\scripts\benchmark_mpi.ps1 `
    -exe ".\x64\Release_MPI\parallel-words-count-mpi.exe" ` 
    -min <processes> `
    -max <processes> `
    -runs <runs> `
    <path>

	Sequential:
    ------------
	.\x64\Release_SEQ\parallel-words-count-seq.exe <path1> [path2] ...

	Sequential benchmarking:
    ------------------------
    .\benchmarking\scripts\benchmark_seq.ps1 `
    -exe ".\x64\Release_SEQ\parallel-words-count-seq.exe" `
    -runs <runs> `
    <path>



    ============================================================
*/

#include <filesystem>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <cctype>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <iomanip>
#include <cstdlib>
#include <system_error>

//#define USE_MPI
//#define USE_OPENMP
//#define USE_SEQ

#if defined(USE_MPI)
#include <mpi.h>
#elif defined(USE_OPENMP)
#include <omp.h>
#endif

namespace fs = std::filesystem;

// ============================================================
// Sop words
// ============================================================

const std::unordered_set<std::string> stop_words = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "of", "for", "with", "by",
    "is", "was", "were", "are", "be", "been", "i", "you", "he", "she", "it", "we", "they", "me", "my"
};

// ============================================================
// Logging & Output
// ============================================================


std::mutex log_mutex;
std::ofstream result_file;

void log_message(const std::string& msg)
{
    std::lock_guard<std::mutex> guard(log_mutex);
    std::cout << msg << std::endl;
}

void write_result(const std::string& msg)
{
    if (result_file.is_open()) {
        result_file << msg << '\n';
    }
}

std::string human_size(uint64_t bytes)
{
    std::ostringstream oss;
    const double KB = 1024.0;
    const double MB = KB * 1024.0;
    const double GB = MB * 1024.0;

    oss << std::fixed << std::setprecision(2);

    if (bytes >= GB) {
        oss << (bytes / GB) << " GB";
    }
    else if (bytes >= MB) {
        oss << (bytes / MB) << " MB";
    }
    else if (bytes >= KB) {
        oss << (bytes / KB) << " KB";
    }
    else {
        oss << bytes << " B";
    }

    return oss.str();
}

#ifdef USE_MPI
void mpi_log(int rank, const std::string& msg)
{
    log_message("[Process " + std::to_string(rank) + "] " + msg);
}
#endif

// ============================================================
// Configuration & Structures
// ============================================================

uint64_t get_adaptive_chunk_size(uint64_t total_bytes, int workers)
{
    const uint64_t MIN_CHUNK = 1ULL * 1024 * 1024;   // 1MB
    const uint64_t MAX_CHUNK = 16ULL * 1024 * 1024;  // 16MB

    uint64_t base = total_bytes / (workers * 4); // 4 chunks per worker

    if (base < MIN_CHUNK) return MIN_CHUNK;
    if (base > MAX_CHUNK) return MAX_CHUNK;

    return base;

}

struct ChunkTask
{
    fs::path file;
    uint64_t offset;
    uint64_t length;
};

// ============================================================
// Utility
// ============================================================

bool compare_file_size(const fs::path& a, const fs::path& b)
{
    std::error_code ec1, ec2;
    uint64_t s1 = fs::file_size(a, ec1);
    uint64_t s2 = fs::file_size(b, ec2);
    return s1 < s2;
}

template<typename T>
std::vector<T> shuffle(const std::vector<T>& data, int buckets)
{
    int size = static_cast<int>(data.size());
    int fill = size / buckets + std::min(1, size % buckets);

    std::vector<T> shuffled;
    shuffled.reserve(data.size());

    for (int bucket = 0; bucket < buckets; ++bucket) {
        for (int i = 0; i < fill; ++i) {
            int target = i * buckets + bucket;
            if (target < size) {
                shuffled.push_back(data[target]);
            }
        }
    }
    return shuffled;
}

// ============================================================
// File Collection
// ============================================================

void collect_files(const fs::path& path, std::vector<fs::path>& files)
{
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        log_message("Path does not exist: " + path.string());
        return;
    }

    if (fs::is_regular_file(path, ec)) {
        files.push_back(path);
    }
    else if (fs::is_directory(path, ec)) {
        for (const auto& entry : fs::recursive_directory_iterator(path, fs::directory_options::skip_permission_denied)) {
            if (fs::is_regular_file(entry.path(), ec)) {
                files.push_back(entry.path());
            }
        }
    }
}

// ============================================================
// Tokenization
// ============================================================

inline bool is_word_char(char c)
{
    return std::isalpha(static_cast<unsigned char>(c)) || c == '\'';
}

inline char normalize_char(char c)
{
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

inline void trim_quotes(std::string& token)
{
    size_t start = token.find_first_not_of('\'');
    if (start == std::string::npos) {
        token.clear();
        return;
    }
    size_t end = token.find_last_not_of('\'');
    token = token.substr(start, end - start + 1);
}

// ============================================================
// Chunk Creation
// ============================================================
std::vector<ChunkTask> create_chunk_tasks(const std::vector<fs::path>& files, int workers)
{
    std::vector<ChunkTask> tasks;
    for (const auto& file : files) {
        uint64_t size = fs::file_size(file);
        uint64_t chunk_size = get_adaptive_chunk_size(size, workers);
        for (uint64_t offset = 0; offset < size; offset += chunk_size) {
            uint64_t length = std::min(chunk_size, size - offset);
            tasks.push_back({ file, offset, length });
        }
    }
    return tasks;
}

// ============================================================
// Chunk Processing
// ============================================================

bool is_valid_token(const std::string& token) {
    if (token.empty()) return false; 

    if (token.length() > 20) return false;

    // 4+ letters
    int same_char_count = 1;
    for (size_t i = 1; i < token.length(); ++i) {
        if (token[i] == token[0]) same_char_count++;
        else break;
    }
    if (same_char_count >= 5) return false;

    // stop words
    if (stop_words.find(token) != stop_words.end()) return false;

    return true;
}


void process_chunk(const ChunkTask& task, std::unordered_map<std::string, uint64_t>& local_map, std::vector<char>& buffer)
{
    std::ifstream file(task.file, std::ios::binary);
    if (!file) return;

    if (buffer.size() < task.length) {
        buffer.resize(task.length);
    }

    file.seekg(task.offset);
    file.read(buffer.data(), task.length);
    uint64_t bytes_read = static_cast<uint64_t>(file.gcount());

    // ========================================================
    // Boundary Repair (skip partial word at the beginning)
    // ========================================================
    uint64_t start = 0;
    if (task.offset != 0) {
        char prev;
        file.seekg(task.offset - 1);
        file.read(&prev, 1);
        // If previous character was part of a word, skip the rest of this word
        if (is_word_char(prev)) {
            while (start < bytes_read && is_word_char(buffer[start])) {
                start++;
            }
        }
    }

    // ========================================================
    // Tokenization
    // ========================================================
    std::string token;
    for (uint64_t i = start; i < bytes_read; ++i) {
        if (is_word_char(buffer[i])) {
            token.push_back(normalize_char(buffer[i]));
        }
        else {
            if (!token.empty()) {
                trim_quotes(token);
                if (is_valid_token(token)) {
                    ++local_map[token];
                }
                token.clear();
            }
        }
    }

    // ========================================================
    // Overflow Handling (if we ended mid-word, finish it)
    // ========================================================
    if (!token.empty()) {
        uint64_t global_file_end = fs::file_size(task.file);
        uint64_t current_file_pos = task.offset + bytes_read;

        // keep reading from the file until the word ends
        char next_char;
        file.seekg(current_file_pos);
        while (current_file_pos < global_file_end && file.read(&next_char, 1)) {
            if (is_word_char(next_char)) {
                token.push_back(normalize_char(next_char));
                current_file_pos++;
            }
            else {
                break;
            }
        }

        // process the fully reconstructed token
        trim_quotes(token);
        if (is_valid_token(token)) {
            ++local_map[token];
        }
    }
}

// ============================================================
// Merge
// ============================================================

void merge_maps(
    std::unordered_map<std::string, uint64_t>& global_map,
    const std::unordered_map<std::string, uint64_t>& local_map)
{
    for (const auto& pair : local_map) {
        global_map[pair.first] += pair.second;
    }
}

// ============================================================
// MPI Serialization
// ============================================================

#if defined (USE_MPI)

std::string serialize_map(const std::unordered_map<std::string, uint64_t>& map)
{
    std::ostringstream oss;
    for (const auto& pair : map) {
        oss << pair.first << ' ' << pair.second << '\n';
    }
    return oss.str();
}

std::unordered_map<std::string, uint64_t> deserialize_map(const std::string& data)
{
    std::unordered_map<std::string, uint64_t> map;
    std::istringstream iss(data);
    std::string word;
    uint64_t count;

    while (iss >> word >> count) {
        map[word] += count;
    }
    return map;
}

// ============================================================
// MPI VERSION
// ============================================================

int mpi_main(int argc, char* argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();

    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ROOT = 0;

    if (rank == ROOT) {
        result_file.open("output/mpi_result.txt");
        log_message("Mode: MPI");
    }

    std::vector<ChunkTask> all_tasks;

    if (rank == ROOT) {
        if (argc < 2) {
            log_message("Usage: <path1> [path2] ...");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<fs::path> files;
        for (int i = 1; i < argc; ++i) {
            collect_files(argv[i], files);
        }

        log_message("Collected files: " + std::to_string(files.size()));

        std::sort(files.begin(), files.end(), compare_file_size);
        files = shuffle(files, size);
        all_tasks = create_chunk_tasks(files, size);

        log_message("Created chunks: " + std::to_string(all_tasks.size()));
    }

    std::string serialized;

    if (rank == ROOT) {
        std::vector<std::vector<std::string>> partitions(size);

        for (int i = 0; i < static_cast<int>(all_tasks.size()); ++i) {
            const auto& t = all_tasks[i];
            std::ostringstream oss;
            oss << t.file.string() << '|' << t.offset << '|' << t.length;
            partitions[i % size].push_back(oss.str());
        }

        std::ostringstream global;
        for (int p = 0; p < size; ++p) {
            global << "#PROCESS\n";
            for (const auto& line : partitions[p]) {
                global << line << '\n';
            }
        }
        serialized = global.str();
    }

    int serialized_size = static_cast<int>(serialized.size());
    MPI_Bcast(&serialized_size, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    serialized.resize(serialized_size);
    MPI_Bcast(serialized.data(), serialized_size, MPI_CHAR, ROOT, MPI_COMM_WORLD);

    std::vector<ChunkTask> local_tasks;

    {
        std::istringstream iss(serialized);
        std::string line;
        int current_process = -1;

        while (std::getline(iss, line)) {
            if (line == "#PROCESS") {
                ++current_process;
                continue;
            }
            if (current_process != rank) continue;

            std::stringstream ss(line);
            std::string path, offset_str, length_str;

            std::getline(ss, path, '|');
            std::getline(ss, offset_str, '|');
            std::getline(ss, length_str, '|');

            local_tasks.push_back({ path, std::stoull(offset_str), std::stoull(length_str) });
        }
    }

    std::unordered_map<std::string, uint64_t> local_map;
    std::vector<char> process_buffer; // Allocate once per MPI process

    for (const auto& task : local_tasks) {
        process_chunk(task, local_map, process_buffer);
    }

    std::string local_serialized = serialize_map(local_map);
    int local_size = static_cast<int>(local_serialized.size());
    std::vector<int> recv_sizes;

    if (rank == ROOT) {
        recv_sizes.resize(size);
    }

    MPI_Gather(&local_size, 1, MPI_INT,
        rank == ROOT ? recv_sizes.data() : nullptr,
        1, MPI_INT, ROOT, MPI_COMM_WORLD);

    std::vector<int> displacements;
    std::vector<char> recv_buffer;

    if (rank == ROOT) {
        displacements.resize(size);
        int total = 0;
        for (int i = 0; i < size; ++i) {
            displacements[i] = total;
            total += recv_sizes[i];
        }
        recv_buffer.resize(total);
    }

    MPI_Gatherv(
        local_serialized.data(), local_size, MPI_CHAR,
        rank == ROOT ? recv_buffer.data() : nullptr,
        rank == ROOT ? recv_sizes.data() : nullptr,
        rank == ROOT ? displacements.data() : nullptr,
        MPI_CHAR, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        std::unordered_map<std::string, uint64_t> global_map;

        for (int i = 0; i < size; ++i) {
            std::string partial(recv_buffer.data() + displacements[i], recv_sizes[i]);
            auto partial_map = deserialize_map(partial);
            merge_maps(global_map, partial_map);
        }

        log_message("Total unique words: " + std::to_string(global_map.size()));
        write_result("Total unique words: " + std::to_string(global_map.size()));
        write_result("");

        std::vector<std::pair<std::string, uint64_t>> sorted_words(global_map.begin(), global_map.end());

        std::sort(sorted_words.begin(), sorted_words.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
            });

        for (const auto& pair : sorted_words) {
            write_result(pair.first + " : " + std::to_string(pair.second));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        log_message("Execution time: " + std::to_string(duration.count()) + " ms");

        result_file.flush();
        log_message("Results saved to: output/mpi_result.txt");
    }

    MPI_Finalize();
    return 0;
}

#elif defined(USE_OPENMP)

// ============================================================
// OPENMP VERSION
// ============================================================

int omp_main(int argc, char* argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();

    if (argc < 3) {
        log_message("Usage: <threads> <path1> [path2] ...");
        return 1;
    }

    int threads = std::stoi(argv[1]);
    log_message("[OMP] Threads: " + std::to_string(threads));

    std::vector<fs::path> files;
    for (int i = 2; i < argc; ++i) {
        collect_files(argv[i], files);
    }

    log_message("Collected files: " + std::to_string(files.size()));

    std::sort(files.begin(), files.end(), compare_file_size);
    files = shuffle(files, threads);
    std::vector<ChunkTask> tasks = create_chunk_tasks(files, threads);

    log_message("Created chunks: " + std::to_string(tasks.size()));

    std::vector<std::unordered_map<std::string, uint64_t>> local_maps(threads);


#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        auto& local_map = local_maps[tid];

        // Allocate ONE buffer for this specific thread
        std::vector<char> thread_buffer;

#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < static_cast<int>(tasks.size()); ++i) {
            process_chunk(tasks[i], local_map, thread_buffer);
        }
    }

    std::unordered_map<std::string, uint64_t> global_map;

    for (const auto& map : local_maps) {
        merge_maps(global_map, map);
    }

    log_message("Total unique words: " + std::to_string(global_map.size()));
    write_result("Total unique words: " + std::to_string(global_map.size()));
    write_result("");

    std::vector<std::pair<std::string, uint64_t>> sorted_words(global_map.begin(), global_map.end());

    std::sort(sorted_words.begin(), sorted_words.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
        });

    for (const auto& pair : sorted_words) {
        write_result(pair.first + " : " + std::to_string(pair.second));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    log_message("Execution time: " + std::to_string(duration.count()) + " ms");

    result_file.flush();
    log_message("Results saved to: output/omp_result.txt");

    return 0;
}

#else
// ============================================================
// SEQ VERSION
// ============================================================

int seq_main(int argc, char* argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();

    if (argc < 2) {
        log_message("Usage: <path1> [path2] ...");
        return 1;
    }

    std::vector<fs::path> files;
    for (int i = 1; i < argc; ++i) {
        collect_files(argv[i], files);
    }


    log_message("Collected files: " + std::to_string(files.size()));

    std::sort(files.begin(), files.end(), compare_file_size);

    std::vector<ChunkTask> tasks = create_chunk_tasks(files, 1);

    log_message("Created chunks: " + std::to_string(tasks.size()));

    std::unordered_map<std::string, uint64_t> global_map;
    std::vector<char> seq_buffer; // Allocate once

    for (const auto& task : tasks) {
        process_chunk(task, global_map, seq_buffer);
    }
   

    log_message("Total unique words: " + std::to_string(global_map.size()));
    write_result("Total unique words: " + std::to_string(global_map.size()));
    write_result("");

    std::vector<std::pair<std::string, uint64_t>> sorted_words(global_map.begin(), global_map.end());

    std::sort(sorted_words.begin(), sorted_words.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
        });

    for (const auto& pair : sorted_words) {
        write_result(pair.first + " : " + std::to_string(pair.second));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    log_message("Execution time: " + std::to_string(duration.count()) + " ms");

    result_file.flush();
    log_message("Results saved to: output/seq_result.txt");

    return 0;
}

#endif

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[])
{
#if defined(USE_SEQ)
    result_file.open("output/seq_result.txt");
    log_message("Mode: Sequential");
    return seq_main(argc, argv);

#elif defined(USE_MPI)
    return mpi_main(argc, argv);

#else // OPENMP
    result_file.open("output/omp_result.txt");
    log_message("Mode: OpenMP");
    return omp_main(argc, argv);
#endif
}