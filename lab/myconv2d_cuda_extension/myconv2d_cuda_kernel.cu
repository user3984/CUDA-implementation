
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#define BLOCK_SIZE 32

template <typename scalar_t>
__global__ void conv2d_kernel(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    const int M,
    const int N,
    const int ksize, 
    const int stride,
    const bool trans_A = false,
    const bool trans_B = false) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int blockRow = threadIdx.x;
    const int blockCol = threadIdx.y;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float val = 0;

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

std::vector<torch::Tensor> mylinear_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights)
{
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weights.size(0);

    auto output = torch::zeros({M, N}, torch::TensorOptions().device(torch::kCUDA));

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((M - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_forward", ([&] {
        conv2d_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            weights.data<scalar_t>(),
            output.data<scalar_t>(),
            M,
            K,
            N,
            false,
            true);
        }));
    
    return {output};
}

std::vector<torch::Tensor> mylinear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights)
{
    const int M = grad_output.size(0);
    const int N = grad_output.size(1);
    const int K = weights.size(1);

    auto grad_input = torch::zeros({M, K}, torch::TensorOptions().device(torch::kCUDA));
    auto grad_weights = torch::zeros({N, K}, torch::TensorOptions().device(torch::kCUDA));

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid1((M - 1) / BLOCK_SIZE + 1, (K - 1) / BLOCK_SIZE + 1);
    const dim3 grid2((N - 1) / BLOCK_SIZE + 1, (K - 1) / BLOCK_SIZE + 1);


    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_backward_input", ([&] {
        conv2d_kernel<scalar_t><<<grid1, block>>>(
            grad_output.data<scalar_t>(),
            weights.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            M,
            N,
            K,
            false,
            false);
        }));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "mylinear_cuda_backward_input", ([&] {
        conv2d_kernel<scalar_t><<<grid2, block>>>(
            grad_output.data<scalar_t>(),
            input.data<scalar_t>(),
            grad_weights.data<scalar_t>(),
            N,
            M,
            K,
            true,
            false);
        }));
    
    return {grad_input, grad_weights};
}