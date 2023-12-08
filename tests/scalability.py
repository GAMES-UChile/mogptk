import time
import sys
import mogptk
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity
import tracemalloc
import matplotlib.pyplot as plt
import gc
from torchviz import make_dot

num_runs = 1
device = mogptk.gpr.config.device
dtype = mogptk.gpr.config.dtype

def generate_data(data_points, input_dims, output_dims):
    X = [np.random.rand(data_points, input_dims) for i in range(output_dims)]
    Y = [np.random.rand(data_points) for i in range(output_dims)]
    return X, Y

def get_model(name, data_points, input_dims, output_dims, Q=1, inference=mogptk.Exact()):
    x, y = generate_data(data_points, input_dims, output_dims)
    dataset = mogptk.DataSet(x, y)

    if name == 'white':
        kernel = mogptk.gpr.WhiteKernel(input_dims=input_dims)
    elif name == 'sm':
        kernel = mogptk.SpectralMixtureKernel(Q=Q, input_dims=input_dims)
        kernel.magnitude.assign(np.random.rand(Q))
        kernel.mean.assign(np.random.rand(Q,input_dims))
        kernel.variance.assign(np.random.rand(Q,input_dims))
    elif name == 'mo_white':
        kernel = mogptk.gpr.WhiteKernel(input_dims=input_dims)
        kernel = mogptk.IndependentMultiOutputKernel(kernel, output_dims=output_dims)
    elif name == 'mosm':
        kernel = mogptk.MultiOutputSpectralMixtureKernel(Q=Q, output_dims=output_dims, input_dims=input_dims)
        kernel.weight.assign(np.random.rand(output_dims,Q))
        kernel.mean.assign(np.random.rand(output_dims,Q,input_dims))
        kernel.variance.assign(np.random.rand(output_dims,Q,input_dims))
    else:
        raise Exception('unknown kernel')

    #mean = mogptk.MultiOutputMean(mogptk.LinearMean(input_dims=input_dims))
    return mogptk.Model(dataset, kernel, inference=inference)

def train(model, iters, jit=False):
    #model.train(method='Adam', lr=0.1, iters=iters, jit=jit)
    #return

    #gpr = torch.jit.trace(model.gpr, ())
    #gpr = model.gpr
    #for i in range(iters):
        #_ = model.gpr.kernel.K(model.gpr.X)
        #_ = model.gpr.log_marginal_likelihood()
        #model.gpr.zero_grad()
        #loss = gpr.forward()
        #loss.backward()

def plot(ax1, x, xlabel, y1_mean, y1_stddev, y2_mean, y2_stddev):
    y1_mean = np.array(y1_mean)
    y1_stddev = np.array(y1_stddev)
    y2_mean = np.array(y2_mean)/1024/1024
    y2_stddev = np.array(y2_stddev)/1024/1024

    ax2 = ax1.twinx()

    ax1.fill_between(x, y1_mean-y1_stddev, y1_mean+y1_stddev, alpha=0.2, facecolor='b')
    ax2.fill_between(x, y2_mean-y2_stddev, y2_mean+y2_stddev, alpha=0.2, facecolor='r')

    ax1.plot(x, y1_mean, 'bs-')
    ax2.plot(x, y2_mean, 'rs-')
    
    y1_min = (y1_mean+y1_stddev).min()
    y1_max = (y1_mean+y1_stddev).max()
    y2_min = (y2_mean+y2_stddev).min()
    y2_max = (y2_mean+y2_stddev).max()
    ax1.set_ylim(0, max(y1_max*1.1, y1_min+y1_max))
    ax2.set_ylim(0, max(y2_max*1.1, y2_min+y2_max))

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Runtime (s)', color='b')
    ax2.set_ylabel('Max. memory (MiB)', color='r')

    ax1.set_title(xlabel, fontsize=16)

def run(ax, xlabel, x, fn_model, fn_train):
    print()
    print(xlabel)
    y1_mean = []
    y2_mean = []
    y1_stddev = []
    y2_stddev = []

    for xi in x:
        y1 = []
        y2 = []
        print('%s' % (xi,), end='', flush=True)

        # stabilize first run
        model = fn_model(xi)
        fn_train(model, xi)

        for i in range(num_runs):
            model = fn_model(xi)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn_train(model, xi)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            y1.append(t1-t0)
            y2.append(torch.cuda.max_memory_allocated())

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print('.', end='', flush=True)

        y1 = np.array(y1)
        y1_mean.append(y1.mean())
        y1_stddev.append(y1.std())

        y2 = np.array(y2)
        y2_mean.append(y2.mean())
        y2_stddev.append(y2.std())

        print(' %g±%g %g±%g' % (y1_mean[-1], y1_stddev[-1], y2_mean[-1], y2_stddev[-1]), flush=True)
    #print('Fit runtime:', np.polyfit(x, y1_mean, 2))
    #print('Fit max. memory:', np.polyfit(x, y2_mean, 2))
    plot(ax, x, xlabel, y1_mean, y1_stddev, y2_mean, y2_stddev)


### Track tensor allocations
#from gpu_profile import MemTracker
#prof = MemTracker()
#prof.track()
#print(torch.cuda.max_memory_allocated()/1024/1024, 'MB')
#model = get_model('mosm', 500, 1, 4)
#prof.track()
#print(torch.cuda.max_memory_allocated()/1024/1024, 'MB')
#train(model, 1)
#prof.track()
#print(torch.cuda.max_memory_allocated()/1024/1024, 'MB')
#exit()


### Visualize computation graph
#yhat = model.gpr.loss()
#make_dot(yhat, show_attrs=True, show_saved=True, params=dict(model.gpr.named_parameters())).render("model", format="png")
#exit()


### Benchmark
#ts = []
#ms = []
#for i in range(10):
#    print('.')
#    model = get_model('mosm', 25, 1, 2, 1)
#    t = time.time()
#    train(model, 100)
#    ts.append(time.time()-t)
#    ms.append(torch.cuda.max_memory_allocated())
#
#    gc.collect()
#    torch.cuda.empty_cache()
#    torch.cuda.reset_peak_memory_stats()
#
#ts = np.array(ts)
#ms = np.array(ms)
#print('t: %g±%g' % (ts.mean(), ts.std()))
#print('m: %g±%g' % (ms.mean(), ms.std()))
#exit()


### White kernel
fig, ax = plt.subplots(3, 1, figsize=(9.6,3*4.8), layout='constrained')

# Iterations
x = [25,50,100,200,400,800]#,1200,1600]
run(ax[0], 'Iterations', x, lambda x: get_model('white', 200, 2, 1), lambda model,x: train(model, x))

# Data points
x = [25,50,100,200,400,800]#,1200,1600]
run(ax[1], 'Data points', x, lambda x: get_model('white', x, 2, 1), lambda model,x: train(model, 250))

# Input dims
x = [1,2,4,8,16,32,64]
run(ax[2], 'Input dims', x, lambda x: get_model('white', 200, x, 1), lambda model,x: train(model, 250))

fig.suptitle('Exact model, White kernel', fontsize=24)
fig.savefig('figs/exact_white.png')



### Spectral mixture kernel
fig, ax = plt.subplots(3, 1, figsize=(9.6,3*4.8), layout='constrained')

# Iterations
x = [25,50,100,200,400,600,800]#,1200,1600]
run(ax[0], 'Iterations', x, lambda x: get_model('sm', 200, 2, 1, 2), lambda model,x: train(model, x))

# Data points
x = [25,50,100,200,400,600,800]#,1200,1600]
run(ax[1], 'Data points', x, lambda x: get_model('sm', x, 2, 1, 2), lambda model,x: train(model, 250))

# Input dims
x = [1,2,4,8,16,32,64]
run(ax[2], 'Input dims', x, lambda x: get_model('sm', 200, x, 1, 2), lambda model,x: train(model, 250))

fig.suptitle('Exact model, SM kernel (Q=2)', fontsize=24)
fig.savefig('figs/exact_sm.png')



### MO White kernel
inducing_points = 10
for name, inference in zip(['Exact', 'Titsias', 'Hensman'], [mogptk.Exact(), mogptk.Titsias(inducing_points), mogptk.Hensman(inducing_points)]):
    if name != 'Exact':
        continue
    data_points = 200
    input_dims = 1
    output_dims = 2
    Q = 2
    iters = 100

    fig, ax = plt.subplots(3, 1, figsize=(1*6.4, 3*4.8), layout='constrained', squeeze=False)

    # Iterations
    x = [100,200,400,800,1200,1600]
    run(ax[0][0], 'Iterations', x, lambda x: get_model('mo_white', data_points, output_dims, input_dims, Q, inference), lambda model,x: train(model, x))

    # Data points
    x = [100,200,400,600,800,1000,1200,1600]
    run(ax[1][0], 'Data points', x, lambda x: get_model('mo_white', x, output_dims, input_dims, Q, inference), lambda model,x: train(model, iters))

    # Output dims
    x = [1,2,3,4,6,8,12,16]
    run(ax[2][0], 'Output dims', x, lambda x: get_model('mo_white', int(1600/x), output_dims, x, Q, inference), lambda model,x: train(model, iters))

    fig.suptitle('%s model, MultiOutput-White kernel' % (name,), fontsize=24)
    fig.savefig('figs/%s_mowhite.png' % (name.lower(),))



### MOSM kernel
fig, ax = plt.subplots(3, 2, figsize=(2*6.4, 3*4.8), layout='constrained')

# Iterations
x = [100,200,400,800,1200,1600]
run(ax[0][0], 'Iterations', x, lambda x: get_model('mosm', 200, 2, 1, 2), lambda model,x: train(model, x))

# Data points
x = [100,200,400,600,800,1000,1200,1600]
run(ax[1][0], 'Data points', x, lambda x: get_model('mosm', x, 2, 1, 2), lambda model,x: train(model, 100))

# Input dims
x = [40,80,160,320,640,1280]
run(ax[2][0], 'Input dims', x, lambda x: get_model('mosm', 200, x, 1, 2), lambda model,x: train(model, 100))

# Output dims
x = [1,2,3,4,6,8,12,16]
run(ax[0][1], 'Output dims', x, lambda x: get_model('mosm', int(1600/x), 2, x, 2), lambda model,x: train(model, 100))

# Components
x = [40,80,160,320,640,1280]
run(ax[1][1], 'Components', x, lambda x: get_model('mosm', 200, 2, 1, x), lambda model,x: train(model, 100))

ax[2][1].axis('off')

fig.suptitle('Exact model, MOSM kernel', fontsize=24)
fig.savefig('figs/exact_mosm.png')
