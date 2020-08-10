import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, signal

class bse:
    def __init__(self, space_input, space_output):
        self.offset = np.median(space_input)
        self.x = space_input - self.offset
        self.y = space_output
        self.Nx = len(self.x)
        self.alpha = 1/2/((np.max(self.x)-np.min(self.x))/2)**2
        self.sigma = np.std(self.y)
        self.gamma = 1/2/((np.max(self.x)-np.min(self.x))/self.Nx)**2
        self.theta = 0.01
        self.sigma_n = np.std(self.y)/10
        self.time = np.linspace(np.min(self.x), np.max(self.x), 500)
        self.w = np.linspace(0, self.Nx/(np.max(self.x)-np.min(self.x))/16, 500)
        self.post_mean = None
        self.post_cov = None
        self.post_mean_r = None
        self.post_cov_r = None
        self.post_mean_i = None
        self.post_cov_i = None
        self.time_label = None
        self.signal_label = None

    def neg_log_likelihood(self):
        Y = self.y
        Gram = Spec_Mix(self.x,self.x,self.gamma,self.theta,self.sigma) + 1e-8*np.eye(self.Nx)
        K = Gram + self.sigma_n**2*np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Nx*np.log(2*np.pi))


    def nlogp(self, hypers):
        sigma = np.exp(hypers[0])
        gamma = np.exp(hypers[1])
        theta = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Y = self.y
        Gram = Spec_Mix(self.x,self.x,gamma,theta,sigma)
        K = Gram + sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Nx*np.log(2*np.pi))

    def dnlogp(self, hypers):
        sigma = np.exp(hypers[0])
        gamma = np.exp(hypers[1])
        theta = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Y = self.y
        Gram = Spec_Mix(self.x,self.x,gamma,theta,sigma)
        K = Gram + sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
        h = np.linalg.solve(K,Y).T

        dKdsigma = 2*Gram/sigma
        dKdgamma = -Gram*(outersum(self.x,-self.x)**2)
        dKdtheta = -2*np.pi*Spec_Mix_sine(self.x,self.x, gamma, theta, sigma)*outersum(self.x,-self.x)
        dKdsigma_n = 2*sigma_n*np.eye(self.Nx)

        H = (np.outer(h,h) - np.linalg.inv(K))
        dlogp_dsigma = sigma * 0.5*np.trace(H@dKdsigma)
        dlogp_dgamma = gamma * 0.5*np.trace(H@dKdgamma)
        dlogp_dtheta = theta * 0.5*np.trace(H@dKdtheta)
        dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)
        return np.array([-dlogp_dsigma, -dlogp_dgamma, -dlogp_dtheta, -dlogp_dsigma_n])

    def train(self):
        hypers0 = np.array([np.log(self.sigma), np.log(self.gamma), np.log(self.theta), np.log(self.sigma_n)])
        res = optimize.minimize(self.nlogp, hypers0, args=(), method='L-BFGS-B', jac = self.dnlogp, options={'maxiter': 500, 'disp': True})
        self.sigma = np.exp(res.x[0])
        self.gamma = np.exp(res.x[1])
        self.theta = np.exp(res.x[2])
        self.sigma_n = np.exp(res.x[3])

    def compute_moments(self):
        #posterior moments for time
        cov_space = Spec_Mix(self.x,self.x,self.gamma,self.theta,self.sigma) + 1e-5*np.eye(self.Nx) + self.sigma_n**2*np.eye(self.Nx)
        cov_time = Spec_Mix(self.time,self.time, self.gamma, self.theta, self.sigma)
        cov_star = Spec_Mix(self.time,self.x, self.gamma, self.theta, self.sigma)
        self.post_mean = np.squeeze(cov_star@np.linalg.solve(cov_space,self.y))
        self.post_cov = cov_time - (cov_star@np.linalg.solve(cov_space,cov_star.T))

        #posterior moment for frequency
        cov_real, cov_imag = freq_covariances(self.w,self.w,self.alpha,self.gamma,self.theta,self.sigma, kernel = 'sm')
        xcov_real, xcov_imag = time_freq_covariances(self.w, self.x, self.alpha,self.gamma,self.theta,self.sigma, kernel = 'sm')
        self.post_mean_r = np.squeeze(xcov_real@np.linalg.solve(cov_space,self.y))
        self.post_cov_r = cov_real - (xcov_real@np.linalg.solve(cov_space,xcov_real.T))
        self.post_mean_i = np.squeeze(xcov_imag@np.linalg.solve(cov_space,self.y))
        self.post_cov_i = cov_imag - (xcov_imag@np.linalg.solve(cov_space,xcov_imag.T))
        self.posterior_mean_psd = self.post_mean_r**2 + self.post_mean_i**2 + np.diag(self.post_cov_r + self.post_cov_r)
        return cov_real, xcov_real, cov_space, self.w, self.posterior_mean_psd

    def get_freq_peaks(self):
        x = self.w
        dx = x[1]-x[0]

        y = self.post_mean_r**2 + self.post_mean_i**2 + np.diag(self.post_cov_r + self.post_cov_r)
        ind, _ = signal.find_peaks(y)
        if len(ind) == 0:
            return np.array([]), np.array([]), np.array([])
        ind = ind[np.argsort(y[ind])[::-1]] # sort by biggest peak first

        widths, width_heights, _, _ = signal.peak_widths(y, ind, rel_height=0.5)
        widths *= dx

        positions = x[ind]
        amplitudes = y[ind]
        variances = widths / np.sqrt(8 * np.log(amplitudes / width_heights)) # from full-width half-maximum to Gaussian sigma
        return amplitudes, positions, variances

    def plot_time_posterior(self, flag=None):
        #posterior moments for time
        plt.figure(figsize=(18,6))
        plt.plot(self.x,self.y,'.r', label='observations')
        plt.plot(self.time,self.post_mean, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt(np.diag(self.post_cov))
        plt.fill_between(self.time, self.post_mean - error_bars, self.post_mean + error_bars, color='blue',alpha=0.1, label='95% error bars')
        if flag == 'with_window':
            plt.plot(self.time, 2*self.sigma*np.exp(-self.alpha*self.time**2))
        plt.title('Observations and posterior interpolation')
        plt.xlabel(self.time_label)
        plt.ylabel(self.signal_label)
        plt.legend()
        plt.xlim([min(self.x),max(self.x)])
        plt.tight_layout()
        plt.show()

    def plot_freq_posterior(self):
        #posterior moments for frequency
        plt.figure(figsize=(18,6))
        plt.plot(self.w,self.post_mean_r, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_r)))
        plt.fill_between(self.w, self.post_mean_r - error_bars, self.post_mean_r + error_bars, color='blue',alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (real part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(18,6))
        plt.plot(self.w,self.post_mean_i, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_i)))
        plt.fill_between(self.w, self.post_mean_i - error_bars, self.post_mean_i + error_bars, color='blue',alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (imaginary part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()

    def plot_power_spectral_density(self, how_many, flag=None):
        #posterior moments for frequency
        plt.figure(figsize=(18,6))
        freqs = len(self.w)
        samples = np.zeros((freqs,how_many))
        for i in range(how_many):
            sample_r = np.random.multivariate_normal(self.post_mean_r,(self.post_cov_r+self.post_cov_r.T)/2 + 1e-5*np.eye(freqs))
            sample_i = np.random.multivariate_normal(self.post_mean_i,(self.post_cov_i+self.post_cov_i.T)/2 + 1e-5*np.eye(freqs))
            samples[:,i] = sample_r**2 + sample_i**2
        plt.plot(self.w,samples, color='red', alpha=0.35)
        plt.plot(self.w,samples[:,0], color='red', alpha=0.35, label='posterior samples')
        plt.plot(self.w,self.posterior_mean_psd, color='black', label = '(analytical) posterior mean')
        if flag == 'show peaks':
            peaks, _  = signal.find_peaks(self.posterior_mean_psd)
            plt.stem(self.w[peaks],self.posterior_mean_psd[peaks], markerfmt='ko', label='peaks')
        plt.title('Sample posterior power spectral density')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()
        plt.show()

    def set_labels(self, time_label, signal_label):
        self.time_label = time_label
        self.signal_label = signal_label

    def set_freqspace(self, max_freq, dimension=500):
        self.w = np.linspace(0, max_freq, dimension)


def outersum(a,b):
    # equivalent to np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b) when a and b are arrays
    # speedup approximately 25%
    return np.add.outer(a,b)

def Spec_Mix(x,y, gamma, theta, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.cos(2*np.pi*theta*outersum(x,-y))

def Spec_Mix_sine(x,y, gamma, theta, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.sin(2*np.pi*theta*outersum(x,-y))

def Spec_Mix_spectral(x, y, alpha, gamma, theta, sigma=1):
    magnitude = np.pi * sigma**2 / (np.sqrt(alpha*(alpha + 2*gamma)))
    return magnitude * np.exp(-np.pi**2/(2*alpha)*outersum(x,-y)**2 - 2*np.pi*2/(alpha + 2*gamma)*(outersum(x,y)/2-theta)**2)

def freq_covariances(x, y, alpha, gamma, theta, sigma=1, kernel = 'sm'):
    if kernel == 'sm':
        N = len(x)
        #compute kernels
        K = 1/2*(Spec_Mix_spectral(x, y, alpha, gamma, theta, sigma) + Spec_Mix_spectral(x, y, alpha, gamma, -theta, sigma))
        P = 1/2*(Spec_Mix_spectral(x, -y, alpha, gamma, theta, sigma) + Spec_Mix_spectral(x, -y, alpha, gamma, -theta, sigma))
        real_cov = 1/2*(K + P) + 1e-8*np.eye(N)
        imag_cov = 1/2*(K - P) + 1e-8*np.eye(N)
    return real_cov, imag_cov

def time_freq_SM_re(x, y, alpha, gamma, theta, sigma=1):
    at = alpha/(np.pi**2)
    gt = gamma/(np.pi**2)
    L = 1/at + 1/gt
    return (sigma**2)/(np.sqrt(np.pi*(at+gt))) * np.exp(outersum(-(x-theta)**2/(at+gt), -y**2*np.pi**2/L) ) *np.cos(-np.outer(2*np.pi*(x/at+theta/gt)/(1/at + 1/gt),y))

def time_freq_SM_im(x, y, alpha, gamma, theta, sigma=1):
    at = alpha/(np.pi**2)
    gt = gamma/(np.pi**2)
    L = 1/at + 1/gt
    return (sigma**2)/(np.sqrt(np.pi*(at+gt))) * np.exp(outersum(-(x-theta)**2/(at+gt), -y**2*np.pi**2/L) ) *np.sin(-np.outer(2*np.pi*(x/at+theta/gt)/(1/at + 1/gt),y))

def time_freq_covariances(x, t, alpha, gamma, theta, sigma, kernel = 'sm'):
    if kernel == 'sm':
        tf_real_cov = 1/2*(time_freq_SM_re(x, t, alpha, gamma, theta, sigma) + time_freq_SM_re(x, t, alpha, gamma, -theta, sigma))
        tf_imag_cov = 1/2*(time_freq_SM_im(x, t, alpha, gamma, theta, sigma) + time_freq_SM_im(x, t, alpha, gamma, -theta, sigma))
    return tf_real_cov, tf_imag_cov
