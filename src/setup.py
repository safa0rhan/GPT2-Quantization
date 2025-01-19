from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name='int8_matmul',
    ext_modules=[
        CUDAExtension(
            name='int8_matmul',
            sources=[
                'int8_matmul.cpp',
                'int8_matmul_kernel.cu',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
