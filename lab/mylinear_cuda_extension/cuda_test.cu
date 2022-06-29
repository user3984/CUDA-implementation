
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_SIZE 32

// template <typename scalar_t>
// __global__ void matmul_kernel(
//     const scalar_t* A,
//     const scalar_t* B,
//     scalar_t* C,
//     const int M, 
//     const int K, 
//     const int N,
//     const bool trans_A = false,
//     const bool trans_B = false) 
// {
//     const int row = blockIdx.x * blockDim.x + threadIdx.x;
//     const int col = blockIdx.y * blockDim.y + threadIdx.y;
//     if (row < M && col < N)
//     {
//         scalar_t sum = 0.0;
//         for (int k = 0; k < K; k++)
//         {
//             const int i = trans_A ? (k * M + row) : (row * K + k);
//             const int j = trans_B ? (col * K + k) : (k * N + col);
//             sum += A[i] * B[j];
//         }

//         C[row * N + col]  = sum;
//     }
// }

// use shared memory
template <typename scalar_t>
__global__ void matmul_kernel(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    const int M, 
    const int K, 
    const int N,
    const bool trans_A = false,
    const bool trans_B = false) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int blockRow = threadIdx.x;
    const int blockCol = threadIdx.y;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int val = 0;

    for (int s = 0; s < K; s += BLOCK_SIZE) {
        As[blockRow][blockCol] = (row < M && s + blockCol < K) ? (trans_A ? A[(s + blockCol) * K + row] : A[row * K + s + blockCol]) : 0;
        Bs[blockRow][blockCol] = (col < N && s + blockRow < K) ? (trans_B ? B[col * N + s + blockRow] : B[(s + blockRow) * N + col]) : 0;

        __syncthreads();  // make sure sub-matrices are loaded

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            val += As[blockRow][k] * Bs[k][blockCol];
        }

        // make sure that the preceding computation is done before loading
        // two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col]  = val;
    }

}

int main()
{
    float *input, *weights, *output;

    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    cudaMallocManaged(&input, M * K * sizeof(float));
    cudaMallocManaged(&weights, K * N * sizeof(float));
    cudaMallocManaged(&output, M * N * sizeof(float));

    int sz_input = M * K;
    int sz_weights = K * N;
    int sz_output = M * N;

    // initialize input, weights and output
    for (int i = 0; i < sz_input; ++i) {
        input[i] = 1.0f;
    }

    for (int i = 0; i < sz_weights; ++i) {
        weights[i] = 2.0f;
    }

    for (int i = 0; i < sz_output; ++i) {
        output[i] = 0.0f;
    }

    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 numBlocks((M - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1);
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(input, weights, output, M, K, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    float error = 0.0f;
    for (int i = 0; i < sz_output; ++i) {
        error += abs(output[i] - 2.0f * K);
    }

    std::cout << "Error: " << error << std::endl;
    
    return 0;
}

