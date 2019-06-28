from .kernels import MultiOutputSpectralMixture, SpectralMixtureLMC, ConvolutionalGaussian, CrossSpectralMixture

class MOSM:
    def __init__(self, data, components):
        self.data = data
        self.components = components



    def build(self):
        # starting parameters
        kernel_set = MultiOutputSpectralMixture(self.data.get_input_dimensions(), self.data.get_output_dimensions())
        for component in range(1, self.components):
            kernel_set += MultiOutputSpectralMixture(self.data.get_input_dimensions(), self.data.get_output_dimensions())
        return kernel_set

def CSM(data, components, parameters={}):
    kernel_set = CrossSpectralMixture(data.get_input_dimensions(), data.get_output_dimensions())
    for component in range(1, components):
        kernel_set += CrossSpectralMixture(data.get_input_dimensions(), data.get_output_dimensions())
    return kernel_set

def SM_LMC(data, components, parameters={}):
    kernel_set = SpectralMixtureLMC(data.get_input_dimensions(), data.get_output_dimensions())
    for component in range(1, components):
        kernel_set += SpectralMixtureLMC(data.get_input_dimensions(), data.get_output_dimensions())
    return kernel_set

def CG(data, components, parameters={}):
    kernel_set = ConvolutionalGaussian(data.get_input_dimensions(), data.get_output_dimensions())
    for component in range(1, components):
        kernel_set += ConvolutionalGaussian(data.get_input_dimensions(), data.get_output_dimensions())
    return kernel_set

