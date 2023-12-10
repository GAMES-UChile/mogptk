import torch
import matplotlib.pyplot as plt
import numpy as np

mu = torch.tensor([0.0])
var = torch.tensor([1.0])
prior = torch.distributions.normal.Normal(mu, var.sqrt())

# likelihood conditional on prior for loc
var1 = torch.tensor([5.0])
def likelihood(loc):
    return torch.distributions.normal.Normal(loc, var1.sqrt())

# joint p,q
joint = torch.distributions.normal.Normal(mu, (var + var1).sqrt())


def random(n):
    samples = prior.sample([n])
    samples = likelihood(samples).sample()
    return samples.numpy()

def f(t):
    loc = prior.log_prob(t[0]).exp()
    res = likelihood(loc).log_prob(t[1]).exp()
    return res

def gibbs(n, f):
    samples = torch.empty((n,1))

    k = 10
    n *= k
    warmup = int(n/10)
    n += warmup

    # initial starting point
    t0 = torch.tensor([0.0, 0.0])
    x0 = f(t0)
    if warmup == 0:
        samples[0,:] = t0[1]

    step = torch.tensor([1.0, 1.0])
    g = torch.distributions.normal.Normal(torch.zeros(2), step)
    for i in range(1, n):
        t = t0 + g.sample()
        x = f(t)

        if (i-warmup)%k == 0:
            j = int((i-warmup)/k)
            samples[j,:] = t[1]

        r = x/x0
        accept = torch.rand(1) <= r
        if accept:
            t0 = t
            x0 = x
    return samples.numpy()


nbins = 100
nsamples = 1000
x = torch.linspace(-25.0, 25.0, 1000)
plt.plot(x, prior.log_prob(x).exp(), c='m', label='prior')
plt.plot(x, joint.log_prob(x).exp(), c='g', label='joint')


counts, bins = np.histogram(random(nsamples), bins=nbins, density=True)
true_counts = [joint.log_prob(torch.tensor(bins[i])).exp().numpy() for i in range(len(counts))]
diff = np.abs([true_counts[i] - counts[i] for i in range(len(counts))])
print(np.sum(diff)/nbins)
plt.stairs(counts, bins, label='random')

counts, bins = np.histogram(gibbs(nsamples, f), bins=nbins, density=True)
true_counts = [joint.log_prob(torch.tensor(bins[i])).exp().numpy() for i in range(len(counts))]
diff = np.abs([true_counts[i] - counts[i] for i in range(len(counts))])
print(np.sum(diff)/nbins)
plt.stairs(counts, bins, label='gibbs')

plt.xlim(-25, 25)

plt.legend()
plt.show()
