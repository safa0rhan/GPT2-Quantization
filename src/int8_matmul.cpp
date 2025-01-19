#include <torch/extension.h>
#include <vector>

torch::Tensor int8_matmul_forward_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor int8_gemm_forward_cuda(torch::Tensor A, torch::Tensor B);

torch::Tensor int8_matmul_forward_cpu(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(!A.is_cuda(), "A must be on CPU for CPU fallback");
    TORCH_CHECK(!B.is_cuda(), "B must be on CPU for CPU fallback");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");

    auto M = A.size(0);
    auto N = A.size(1);
    auto K = B.size(1);

    TORCH_CHECK(B.size(0) == N, "B.size(0) must match A.size(1)");

    auto C = torch::zeros({M, K}, torch::dtype(torch::kInt32));

    auto A_acc = A.accessor<int8_t, 2>();
    auto B_acc = B.accessor<int8_t, 2>();
    auto C_acc = C.accessor<int, 2>();

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            int sum = 0;
            for (int n = 0; n < N; n++) {
                sum += static_cast<int>(A_acc[m][n]) *
                       static_cast<int>(B_acc[n][k]);
            }
            C_acc[m][k] = sum;
        }
    }
    return C;
}

torch::Tensor int8_matmul_forward(torch::Tensor A, torch::Tensor B) {
    if (A.is_cuda()) {
        return int8_matmul_forward_cuda(A, B);
    } else {
        return int8_matmul_forward_cpu(A, B);
    }
}

torch::Tensor int8_gemm_forward(torch::Tensor A, torch::Tensor B) {
    if (A.is_cuda()) {
        return int8_gemm_forward_cuda(A, B);
    } else {
        return int8_matmul_forward_cpu(A, B);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_matmul_forward", &int8_matmul_forward, "int8 matmul forward");
    m.def("int8_gemm_forward", &int8_gemm_forward, "int8 GEMM forward");
}