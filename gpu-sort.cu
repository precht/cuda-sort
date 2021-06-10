/**
 * Authors: Jakub Precht
 * Date:    2019/06/06
 */

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <random>
#include <cassert>

using uint = uint32_t;
using namespace std;

#ifdef __GNUC__ // gcc & clang
#define INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER) // msvc
#define INLINE __forceinline inline
#elif
#define INLINE inline
#endif

void checkCudaError(const char *, unsigned, const char *, cudaError_t);
#define CHECK_RETURN(value) checkCudaError(__FILE__,__LINE__, #value, value)

// ------------------------------ i/o helper functions ------------------------------

template<typename T> INLINE void get(T &&v) { cin >> v; }
template<typename T, typename... Args> INLINE void get(T &&v, Args&&... args) { cin >> v; get(args...); }
template<typename T> INLINE void put(T &&v) { cout << fixed << v << '\n'; }
template<typename T, typename... Args> INLINE void put(T &&v, Args&&... args) { cout << fixed << v << ' '; put(args...); }
template<typename T> INLINE void putc(T &&a, const string s = " ") { { for (auto &v : a) cout << v << s; } cout << '\n'; }
template<typename T> INLINE void print(const int size, T *data)
{
    for (int i = 0; i < size; i++)
        cout << data[i] << ' ';
    cout << '\n';
}

void checkCudaError(const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    cout << endl;
    cerr << "CUDA error: " << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << endl;
    exit(1);
}

// ------------------------------ cpu bubble sort ------------------------------

template<typename T>
void cpuSort(vector<T> &data)
{
    bool swapped = 1;
    while (swapped) {
        swapped = 0;
        for (uint i = 1; i < data.size(); i++) {
            if (data[i - 1] > data[i]) {
                swap(data[i - 1], data[i]);
                swapped = 1;
            }
        }
    }
}

// ------------------------- gpu hybrid sort: block bubble sort + blocks merge -------------------------

#define BLOCK_DIM 128

template<typename T>
__global__ void kernelSortBlock(const uint size, T* g_data)
{
    __shared__ T s_data[BLOCK_DIM << 1];
    const uint idx = threadIdx.x << 1;
    const uint offset = (blockDim.x * blockIdx.x) << 1;
    const uint range = (offset + (BLOCK_DIM << 1) > size ? (size - offset) : (BLOCK_DIM << 1));
    uint fst_idx = idx;
    uint snd_idx = idx + 1;

    if (offset + snd_idx > size)
        return;

    s_data[fst_idx] = g_data[offset + fst_idx];
    s_data[snd_idx] = g_data[offset + snd_idx];

    __syncthreads();

    bool shift = 1;
    T fst_val, snd_val;
    for (uint k = 0; k <= range; k++) {
        shift = !shift;
        fst_idx = idx + shift;
        snd_idx = fst_idx + 1;
        if (snd_idx < range) {
            fst_val = s_data[fst_idx];
            snd_val = s_data[snd_idx];
            if (snd_val < fst_val) {
                s_data[fst_idx] = snd_val;
                s_data[snd_idx] = fst_val;
            }
        }
        __syncthreads();
    }

    g_data[offset + idx] = s_data[idx];
    g_data[offset + idx + 1] = s_data[idx + 1];
}

template<typename T>
__global__ void kernelMergeBlocks(const uint size, const uint total_size, const T *g_in, T *g_out)
{
    const uint g_idx = (blockDim.x * blockIdx.x + threadIdx.x) * size;
    if (g_idx >= total_size)
        return;

    uint lhs_idx = g_idx;
    uint rhs_idx = g_idx + (size >> 1);
    uint out_idx = g_idx;

    const uint lhs_lmt = (rhs_idx > total_size ? total_size : rhs_idx);
    const uint rhs_lmt = (rhs_idx + (size >> 1) > total_size ? total_size : rhs_idx + (size >> 1));

    while (lhs_idx < lhs_lmt && rhs_idx < rhs_lmt) {
        const uint lhs_val = g_in[lhs_idx];
        const uint rhs_val = g_in[rhs_idx];
        if (lhs_val > rhs_val) {
            g_out[out_idx] = rhs_val;
            rhs_idx++;
        }
        else {
            g_out[out_idx] = lhs_val;
            lhs_idx++;
        }
        out_idx++;
    }
    while (lhs_idx < lhs_lmt) {
        g_out[out_idx] = g_in[lhs_idx];
        out_idx++;
        lhs_idx++;
    }
    while (rhs_idx < rhs_lmt) {
        g_out[out_idx] = g_in[rhs_idx];
        out_idx++;
        rhs_idx++;
    }
}

template<typename T>
INLINE void cudaSort(uint size, T *data)
{
    T *d_data, *d_tmp;
    CHECK_RETURN(cudaMalloc(&d_data, (size + 1) * sizeof(T)));
    CHECK_RETURN(cudaMalloc(&d_tmp, (size + 1) * sizeof(T)));
    CHECK_RETURN(cudaMemcpy(d_data, data, size * sizeof(T), cudaMemcpyHostToDevice));

    uint sort_blocks_no = (((size + BLOCK_DIM - 1) / BLOCK_DIM) + 1) >> 1;
    kernelSortBlock << <sort_blocks_no, BLOCK_DIM >> > (size, d_data);
    CHECK_RETURN(cudaDeviceSynchronize());

    uint merge_size = BLOCK_DIM << 2;
    const uint twice_block_dim = BLOCK_DIM << 1;

    while (sort_blocks_no > 1) {
        const uint merge_blocks_no = (sort_blocks_no + twice_block_dim - 1) / twice_block_dim;

        kernelMergeBlocks << <merge_blocks_no, BLOCK_DIM >> > (merge_size, size, d_data, d_tmp);
        CHECK_RETURN(cudaDeviceSynchronize());

        sort_blocks_no = (sort_blocks_no + 1) >> 1;
        merge_size <<= 1;
        swap(d_data, d_tmp);
    }

    CHECK_RETURN(cudaMemcpy(data, d_data, size * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_RETURN(cudaFree(d_data));
    CHECK_RETURN(cudaFree(d_tmp));
}

template<typename T>
INLINE void cudaSort(vector<T> &vec)
{
    cudaSort(vec.size(), vec.data());
}

// ------------------------------ tests ------------------------------

using Time = chrono::system_clock::time_point;
using Duration = chrono::duration<double>;
const auto& getTime = chrono::system_clock::now;

template<typename T>
void runTest(vector<T> &data)
{
    vector<T> &cpu_data = data;
    vector<T> cuda_data = data;

    Time beg_time, end_time;
    Duration cpu_time, cuda_time;

    beg_time = getTime();
    cpuSort(cpu_data);
    //sort(cpu_data.begin(), cpu_data.end());
    end_time = getTime();
    cpu_time = end_time - beg_time;

    beg_time = getTime();
    cudaSort(cuda_data);
    end_time = getTime();
    cuda_time = end_time - beg_time;

    bool is_ok = true;
    for (uint i = 1; i < cuda_data.size(); i++)
        if (cuda_data[i - 1] > cuda_data[i])
            is_ok = false;

    put(data.size(), cpu_time.count(), cuda_time.count(), is_ok ? "correct" : "wrong");
    data = cuda_data;
}

void f(const uint size)
{
    using T = float;
    vector<T> vec(size);

    uint val = vec.size();
    for (auto &x : vec)
        x = val--;

    //random_device rand_dev;
    //mt19937 rand_eng(rand_dev());
    mt19937 rand_eng(123456);

    for (auto &v : vec)
        v = (T)rand_eng();

    runTest(vec);
    //putc(vec);
    cout.flush();
}

int main()
{
    ios_base::sync_with_stdio(false);
    cout.precision(6);


    for (int i = 1; i < 1000000; i <<= 1) {
        f(i);
    }

    CHECK_RETURN(cudaDeviceReset());

    return 0;
}


