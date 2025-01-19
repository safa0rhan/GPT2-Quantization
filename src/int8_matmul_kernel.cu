#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void int8_gemm_shared(const int8_t* __restrict__ A,
                                 const int8_t* __restrict__ B,
                                 int* __restrict__ C,
                                 int M, int N, int K) {
    // A: (M, N)
    // B: (N, K)
    // C: (M, K)
    // We are tiling over the "N" dimension (the shared dimension).

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;  // in [0..M)
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;  // in [0..K)

    // Shared memory tiles, each BLOCK_SIZE x BLOCK_SIZE
    __shared__ int8_t A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int8_t B_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Accumulator
    int sum = 0;

    // We iterate over N in chunks of BLOCK_SIZE
    // Each iteration loads a sub-block of A and B
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Column index in A tile => threadIdx.x
        // Row index in A tile => threadIdx.y

        int tiledColA = t * BLOCK_SIZE + threadIdx.x;  
        int tiledRowA = row;                           

        // Load A tile
        if (tiledRowA < M && tiledColA < N) {
            A_tile[threadIdx.y][threadIdx.x] =
                A[tiledRowA * N + tiledColA]; 
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0;
        }

        int tiledRowB = t * BLOCK_SIZE + threadIdx.y;
        int tiledColB = col;

        if (tiledRowB < N && tiledColB < K) {
            B_tile[threadIdx.y][threadIdx.x] =
                B[tiledRowB * K + tiledColB]; 
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int kSub = 0; kSub < BLOCK_SIZE; kSub++) {
            sum += static_cast<int>(A_tile[threadIdx.y][kSub]) *
                   static_cast<int>(B_tile[kSub][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// The forward CUDA function that PyTorch will call:
torch::Tensor int8_gemm_forward_cuda(torch::Tensor A, torch::Tensor B) {
    // A: [M, N], B: [N, K]
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");

    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    // shape check
    TORCH_CHECK(B.size(0) == N, "B.size(0) must match A.size(1)");

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    auto C = torch::zeros({M, K}, options);

    const int8_t* A_ptr = (const int8_t*) A.data_ptr<int8_t>();
    const int8_t* B_ptr = (const int8_t*) B.data_ptr<int8_t>();
    int*          C_ptr = (int*)         C.data_ptr<int>();

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the tile-based kernel
    int8_gemm_shared<<<grid, block>>>(
        A_ptr, B_ptr, C_ptr,
        M, N, K
    );

    return C;
}


// A naive int8 matmul kernel: 
//   A: (M x N)
//   B: (N x K)
//   C: (M x K), each entry is sum_{e=0..N-1} (A[row,e]*B[e,col])

__global__ void int8MatmulKernel(const int8_t* A, const int8_t* B, int* C,
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        int sum = 0;
        for (int e = 0; e < N; e++) {
            sum += static_cast<int>(A[row*N + e]) *
                   static_cast<int>(B[e*K + col]);
        }
        C[row*K + col] = sum;  // int32 accumulation
    }
}

// The forward CUDA function that PyTorch will call:
torch::Tensor int8_matmul_forward_cuda(torch::Tensor A, torch::Tensor B) {
    // A: [M, N], B: [N, K]  (both int8)
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");

    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    TORCH_CHECK(B.size(0) == N, "B.size(0) must match A.size(1)");

    // output: int32 on the same device
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    auto C = torch::zeros({M, K}, options);

    // Get raw pointers
    const int8_t* A_ptr = (const int8_t*) A.data_ptr<int8_t>();
    const int8_t* B_ptr = (const int8_t*) B.data_ptr<int8_t>();
    int*          C_ptr = (int*) C.data_ptr<int>();

    // Launch config
    dim3 blockSize(16, 16); 
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    int8MatmulKernel<<<gridSize, blockSize>>>(A_ptr, B_ptr, C_ptr, M, N, K);

    return C; // shape [M, K], dtype int32
}
