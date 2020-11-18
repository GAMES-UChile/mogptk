import torch

class Config:
    dtype = torch.float64
    device = torch.device('cpu')
    positive_minimum = 1e-8
config = Config()

def use_single_precision():
    config.dtype = torch.float32

def use_double_precision():
    config.dtype = torch.float64

def use_cpu(n=None):
    if n is None:
        config.device = torch.device('cpu')
    else:
        config.device = torch.device('cpu', n)

def use_gpu(n=None):
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
    elif n is not None and (not isinstance(n, int) or n < 0 or torch.cuda.device_count() <= n):
        logger.error("CUDA GPU '%s' is not available" % (n,))
    elif n is None:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cuda', n)

def print_gpu_information():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return

    print("CUDA is available:")
    current = config.device.index
    for n in range(torch.cuda.device_count()):
        print("%2d  %s%s", n, torch.cuda.get_device_name(n), " (default)" if n == default else "")

def set_positive_minimum(val):
    config.positive_minimum = val
