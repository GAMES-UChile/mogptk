import time
import mogptk
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity
import tracemalloc

use_profiler = True
use_benchmark = False
num_runs = 1

device = mogptk.gpr.config.device
dtype = mogptk.gpr.config.dtype

def generate_data(data_points, input_dims, output_dims):
    X = [np.random.rand(data_points, input_dims) for i in range(output_dims)]
    Y = [np.random.rand(data_points) for i in range(output_dims)]
    return X, Y

def train(model):
    T = torch.empty((model.gpr.X.shape[0],model.gpr.X.shape[0]), device=device, dtype=dtype)
    for i in range(2500):
        T = model.gpr.kernel.K(model.gpr.X, out=T)
        #model.gpr.log_marginal_likelihood()
        #loss.backward()
        #model.gpr.loss()
    #model.train(method='Adam', lr=0.1, iters=250)

for output_dims in [1]:
    for Q in [2]:
        for input_dims in [2]:
            for data_points in [100]:
                x, y = generate_data(data_points, input_dims, output_dims)
                dataset = mogptk.DataSet(x, y)

                #kernel = mogptk.MultiOutputSpectralMixtureKernel(Q=Q, output_dims=output_dims, input_dims=input_dims)
                #kernel.weight.assign(np.random.rand(output_dims,Q))
                #kernel.mean.assign(np.random.rand(output_dims,Q,input_dims))
                #kernel.variance.assign(np.random.rand(output_dims,Q,input_dims))

                #kernel = mogptk.SpectralMixtureKernel(Q=Q, input_dims=input_dims)
                #kernel.magnitude.assign(np.random.rand(Q))
                #kernel.mean.assign(np.random.rand(Q,input_dims))
                #kernel.variance.assign(np.random.rand(Q,input_dims))

                kernel = mogptk.gpr.WhiteKernel(input_dims=input_dims)

                model = mogptk.Model(dataset, kernel)

                print('\noutput_dims=%d Q=%d input_dims=%d data_points=%s num_parameters=%d' % (output_dims, Q, input_dims, data_points, model.num_parameters()))

                if use_profiler:
                    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
                        with record_function("model_inference"):
                            for i in range(num_runs):
                                train(model)
                    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
                elif use_benchmark:
                    t = benchmark.Timer(
                        stmt='train(model)',
                        setup='from __main__ import train',
                        globals={'model': model},
                    )
                    m = t.timeit(num_runs)
                    print('mean=%6.2f median=%6.2f iqr=%6.2f' % (m.mean, m.median, m.iqr))
                else:
                    t = time.time()
                    #tracemalloc.start()
                    train(model)
                    #snapshot = tracemalloc.take_snapshot()
                    #top_stats = snapshot.statistics('lineno')

                    #print("[ Top 10 ]")
                    #for stat in top_stats[:10]:
                    #    print(stat)
                    print(time.time()-t)

