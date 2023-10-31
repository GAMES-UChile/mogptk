import unittest
import mogptk
import torch
import time

device = mogptk.gpr.config.device
dtype = mogptk.gpr.config.dtype

kernels = [
    mogptk.gpr.WhiteKernel(),
    mogptk.gpr.ConstantKernel(),
    mogptk.gpr.LinearKernel(),
    mogptk.gpr.PolynomialKernel(degree=2),
    mogptk.gpr.FunctionKernel(phi=lambda x: x.square()),
    mogptk.gpr.ExponentialKernel(),
    mogptk.gpr.SquaredExponentialKernel(),
    mogptk.gpr.RationalQuadraticKernel(),
    mogptk.gpr.PeriodicKernel(),
    mogptk.gpr.LocallyPeriodicKernel(),
    mogptk.gpr.CosineKernel(),
    mogptk.gpr.SincKernel(),
    mogptk.gpr.SpectralKernel(),
    mogptk.gpr.SpectralMixtureKernel(),
    mogptk.gpr.MaternKernel(nu=0.5),
    mogptk.gpr.MaternKernel(nu=1.5),
    mogptk.gpr.MaternKernel(nu=2.5),

    mogptk.gpr.AddKernel(mogptk.gpr.ConstantKernel(), mogptk.gpr.LinearKernel()),
    mogptk.gpr.MulKernel(mogptk.gpr.ConstantKernel(), mogptk.gpr.LinearKernel()),
]

mo_kernels = [
    mogptk.gpr.IndependentMultiOutputKernel(mogptk.gpr.ConstantKernel(), output_dims=2),
    mogptk.gpr.MultiOutputSpectralKernel(output_dims=2),
    mogptk.gpr.MultiOutputSpectralMixtureKernel(Q=2, output_dims=2),
    mogptk.gpr.UncoupledMultiOutputSpectralKernel(output_dims=2),
    mogptk.gpr.MultiOutputHarmonizableSpectralKernel(output_dims=2),
    mogptk.gpr.CrossSpectralKernel(output_dims=2),
    mogptk.gpr.LinearModelOfCoregionalizationKernel(mogptk.gpr.ConstantKernel(), output_dims=2),
    mogptk.gpr.GaussianConvolutionProcessKernel(output_dims=2),
]

class TestKernels(unittest.TestCase):
    def test_diagonals(self):
        X = torch.tensor([[0.0], [1.0], [2.0]], device=device, dtype=dtype)
        for kernel in kernels:
            with self.subTest(kernel.__class__.__name__):
                K = kernel.K(X)
                K_diag = kernel.K_diag(X)
                self.assertTrue(torch.equal(K_diag, K.diagonal()))
        
        X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 2.0]], device=device, dtype=dtype)
        for kernel in mo_kernels:
            with self.subTest(kernel.__class__.__name__):
                K = kernel.K(X)
                K_diag = kernel.K_diag(X)
                self.assertTrue(torch.equal(K_diag, K.diagonal()))

if __name__ == '__main__':
    unittest.main()
