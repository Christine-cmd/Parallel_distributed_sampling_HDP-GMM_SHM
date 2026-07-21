
import numpy as np
import math
from scipy.special import logsumexp, gammaln
from scipy.stats import multivariate_normal

class Gaussian: # a variable of the Gaussian class represents a component
    def __init__(self, X=np.zeros((0, 2)), hyperparams=None):
        self.nk = X.shape[0]
        self.dim = X.shape[1]
        
        if hyperparams is None:
            self._kappa_0 = 1.
            self._nu_0 = 1.0001
            self._mu_0 = np.zeros((1, self.dim))
            self._Psi_0 = 1*np.eye(self.dim)

        else:
            self._mu_0 = hyperparams['mu_0']
            self._kappa_0 = hyperparams['kappa_0']
            self._nu_0 = hyperparams['nu_0']
            self._Psi_0 = hyperparams['Psi_0']

        assert(self._mu_0.shape == (1, self.dim))

        if self._nu_0 <= self.dim:
            self._nu_0 = self.dim

        assert(self._Psi_0.shape == (self.dim, self.dim))

        self.mean = None
        self.covar = None
        if X.shape[0] > 0:
            self.fit(X)
        else:
            self.default()

    def default(self):
        self.mean = np.matrix(np.zeros((1, self.dim)))
        self.covar = 1.0 * np.matrix(np.eye(self.dim))
       
    def recompute_ss(self): # compute sufficient statistics nk, x_bar_k, S_k
        self.nk = self._X.shape[0]
        self.dim = self._X.shape[1]
        if self.nk <= 0:
            self.default()
            return
        # update conjugate posterior parameters # mean_k, covar_k ~ NIW(mu_k, kappa_k, Psi_k, nu_k)
        kappa_k = self._kappa_0 + self.nk
        nu_k = self._nu_0 + self.nk
        x_bar_k = np.matrix(self._sum) / self.nk # x_bar_k
        x_bar_k_mu_0 = x_bar_k - self._mu_0 # x_bar_k - mu_0
        S_k = self._square_sum - self.nk*(x_bar_k.transpose()*x_bar_k) # S_k
        Psi_k = self._Psi_0 + S_k + \
            self._kappa_0 * self.nk * x_bar_k_mu_0.transpose() * x_bar_k_mu_0 / (self._kappa_0 + self.nk) # W_k

        self.mean = (self._kappa_0 * self._mu_0 + self.nk * x_bar_k) / (self._kappa_0 + self.nk) # self.mean = mu_k, as the expectation of the student-t distribution is same as the mean of the normal distribution
        self.covar = (kappa_k + 1) / (kappa_k * (nu_k - self.dim + 1)) * Psi_k # Exp(covar_k)

        # store sufficient statistics
        self.ss = {}
        self.ss['nk'] = self.nk 
        self.ss['x_bar_k'] = x_bar_k  
        self.ss['S_k'] = S_k  

    def fit(self, X):  # fit data
        self._X = X
        self._sum = X.sum(axis=0)
        self._square_sum = np.matrix(X).transpose() * np.matrix(X)
        self.recompute_ss()

    def rm_point(self, x): # remove a point
        """
        remove a point to current Gaussian mixture
        @:param x: data point to be removed
        """
        assert(self._X.shape[0] > 0)
        # Find the index of the point x in self._X
        indices = (abs(self._X - x)).argmin(axis=0)
        indices = np.matrix(indices)
        ind = indices[0, 0]
        for ii in indices:
            if (ii-ii[0] == np.zeros(len(ii))).all():
                ind = ii[0,0]
                break
        tmp = np.matrix(self._X[ind])
        self._sum -= self._X[ind]
        self._X = np.delete(self._X, ind, axis=0)
        self._square_sum -= tmp.transpose() * tmp
        self.recompute_ss()

    def add_point(self, x):
        """
        add a point from current Gaussian mixture
        @:params x: data point to be add
        """
        if self.nk <= 0:
            self._X = np.array([x])
            self._sum = self._X.sum(0)
            self._square_sum = np.matrix(self._X).transpose() * np.matrix(self._X)
        else:
            self._X = np.append(self._X, [x], axis=0)
            self._sum += x
            self._square_sum += np.matrix(x).transpose() * np.matrix(x)
        self.recompute_ss()

    def pdf(self, x):  # compute the prob density of data point x
        # size = len(x)
        # assert size == self.mean.shape[1]
        # assert (size, size) == self.covar.shape
        # det = np.linalg.det(self.covar)
        # assert det != 0
        # norm_const = 1. / (np.power((2*np.pi), float(size)/2) * np.power(det, .5))
        # x_mu = x - self.mean
        # res = math.pow(math.e, -.5 * (x_mu * self.covar.I * x_mu.transpose()))
        # return norm_const * res
        return np.exp(self.logpdf(x))
    
    # use multivariate_t distribution to compute the log pdf
    def logpdf(self, x):
        d = self.dim
        self.nu = self._nu_0 + self.nk
        df = self.nu - d + 1
        assert d == self.mean.shape[1]
        assert (d, d) == self.covar.shape
        x_mu = x - self.mean
        sign, res = np.linalg.slogdet(self.covar)
        lognum = gammaln((d+df)/2.)
        logdenom = gammaln(df/2.) + d/2.*np.log(df*np.pi) + 1/2.*res + (d+df)/2.*np.log(1+1./df * x_mu * self.covar.I * x_mu.transpose())
        return lognum-logdenom
    
