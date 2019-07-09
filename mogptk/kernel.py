from .kernels import MultiOutputSpectralMixture, SpectralMixtureLMC, ConvolutionalGaussian, CrossSpectralMixture

class MOSM:
    def __init__(self, data, components=1):
        self.name = "MOSM"
        self.data = data
        self.components = components

    # TODO: initial parameter estimation algorithms; optimization heuristics

    def build(self):
        # starting parameters
        kernel_set = MultiOutputSpectralMixture(self.data.get_input_dimensions(), self.data.get_output_dimensions())
        for component in range(1, self.components):
            kernel_set += MultiOutputSpectralMixture(self.data.get_input_dimensions(), self.data.get_output_dimensions())
        return kernel_set

class CSM:
    def __init__(self, data, components=1, Rq=1):
        self.name = "CSM"
        self.data = data
        self.components = components
        self.Rq = Rq

    def build(self):
        kernel_set = CrossSpectralMixture(self.data.get_input_dimensions(), self.data.get_output_dimensions(), self.Rq)
        for component in range(1, self.components):
            kernel_set += CrossSpectralMixture(self.data.get_input_dimensions(), self.data.get_output_dimensions(), self.Rq)
        return kernel_set

class SM_LMC:
    def __init__(self, data, components=1, Rq=1):
        self.name = "SM-LMC"
        self.data = data
        self.components = components
        self.Rq = Rq

    def build(self):
        kernel_set = SpectralMixtureLMC(self.data.get_input_dimensions(), self.data.get_output_dimensions(), self.Rq)
        for component in range(1, self.components):
            kernel_set += SpectralMixtureLMC(self.data.get_input_dimensions(), self.data.get_output_dimensions(), self.Rq)
        return kernel_set

class CG:
    def __init__(self, data, components=1):
        self.name = "CG"
        self.data = data
        self.components = components

    def build(self):
        kernel_set = ConvolutionalGaussian(self.data.get_input_dimensions(), self.data.get_output_dimensions())
        for component in range(1, self.components):
            kernel_set += ConvolutionalGaussian(self.data.get_input_dimensions(), self.data.get_output_dimensions())
        return kernel_set

