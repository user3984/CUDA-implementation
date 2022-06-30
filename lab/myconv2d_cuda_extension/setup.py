from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='myconv2d_cuda',
    ext_modules=[
        CUDAExtension('myconv2d_cuda', [
            'myconv2d_cuda.cpp',
            'myconv2d_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })