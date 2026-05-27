# Build

Enable the needed mode in the source code:

```cpp
//#define USE_MPI
//#define USE_OPENMP
//#define USE_SEQ
```

Uncomment only one mode at a time.

---

# Run

## Sequential

```powershell
.\x64\Release_SEQ\parallel-words-count-seq.exe <path1> [path2] ...
```

## OpenMP

```powershell
.\x64\Release_OMP\parallel-words-count-omp.exe <threads> <path1> [path2] ...
```

## MPI

```powershell
mpiexec -np <processes> .\x64\Release_MPI\parallel-words-count-mpi.exe <path1> [path2] ...
```
---

# Benchmarking

## Sequential Benchmark

```powershell
.\benchmarking\scripts\benchmark_seq.ps1 `
-exe ".\x64\Release_SEQ\parallel-words-count-seq.exe" `
-runs <runs> `
<path>
```

## OpenMP Benchmark

```powershell
.\benchmarking\scripts\benchmark_omp.ps1 `
-exe ".\x64\Release_OMP\parallel-words-count-omp.exe" `
-min <threads> `
-max <threads> `
-runs <runs> `
<path>
```
## MPI Benchmark

```powershell
.\benchmarking\scripts\benchmark_mpi.ps1 `
-exe ".\x64\Release_MPI\parallel-words-count-mpi.exe" `
-min <processes> `
-max 12 `
-runs 3 `
.\data\books
```
---

# Output

Results are saved into:

```text
output/seq_result.txt
output/omp_result.txt
output/mpi_result.txt
```

Benchmark scripts generate CSV files with:

- execution time
- unique word count

---
