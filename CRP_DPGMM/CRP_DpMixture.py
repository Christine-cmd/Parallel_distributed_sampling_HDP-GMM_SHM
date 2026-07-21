
from matplotlib.pyplot import step
import numpy as np
import scipy.special as ssp
import math
from scipy.special import logsumexp, gammaln
from scipy.stats import multivariate_normal

import torch
from scipy.optimize import linear_sum_assignment
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

def unsupervised_clustering_accuracy(y: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> tuple:
    """Unsupervised Clustering Accuracy
    """
    assert len(y_pred) == len(y)
    u = np.unique(y)
    n_true_clusters = len(u)
    v = np.unique(y_pred)
    n_pred_clusters = len(v)
    map_u = dict(zip(u, range(n_true_clusters)))
    map_v = dict(zip(v, range(n_pred_clusters)))
    inv_map_u = {v: k for k, v in map_u.items()}
    inv_map_v = {v: k for k, v in map_v.items()}
    r = np.zeros((n_pred_clusters, n_true_clusters), dtype=np.int64)
    for y_pred_, y_ in zip(y_pred, y):
        if y_ in map_u:
            r[map_v[y_pred_], map_u[y_]] += 1
    reward_matrix  = np.concatenate((r, r, r), axis=1)
    cost_matrix = reward_matrix.max() - reward_matrix
    row_assign, col_assign = linear_sum_assignment(cost_matrix)

    # Construct optimal assignments matrix
    row_assign = row_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape
    col_assign = col_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape
    assignments = np.concatenate((row_assign, col_assign), axis=1)
    assignments = [[inv_map_v[x], inv_map_u[y%n_true_clusters]] for x, y in assignments]

    optimal_reward = reward_matrix[row_assign, col_assign].sum() * 1.0
    return optimal_reward / y_pred.size, assignments  

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

    # def pdf(self, x):  # compute the prob density of data point x
    #     size = len(x)
    #     assert size == self.mean.shape[1]
    #     assert (size, size) == self.covar.shape
    #     det = np.linalg.det(self.covar)
    #     assert det != 0
    #     norm_const = 1. / (np.power((2*np.pi), float(size)/2) * np.power(det, .5))
    #     x_mu = x - self.mean
    #     res = math.pow(math.e, -.5 * (x_mu * self.covar.I * x_mu.transpose()))
    #     return norm_const * res
    #     #return np.exp(self.logpdf(x))

    # Use a Gaussian distribution to compute the (approximated) log pdf
    # def logpdf(self, x):    # compute the log prob density of data point x 
    #     size = len(x)
    #     assert size == self.mean.shape[1]
    #     assert (size, size) == self.covar.shape
    #     det = np.linalg.det(self.covar)
    #     assert det != 0
    #     norm_const = -np.log(np.power((2*np.pi), float(size)/2) * np.power(det, .5))
    #     x_mu = x - self.mean
    #     res = -.5 * (x_mu * self.covar.I * x_mu.transpose())
    #     return norm_const + res
    
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

class DpMixture:
    def __init__(self, data, hyparams, alpha, sample_alpha=False, alpha_a=100.0, alpha_b=100.0):
        # Other than Gaussian, component model can be any other models
        # with corresponding hyperparameters
        self.hyparams     = hyparams
        self.data         = data # n*d  array-like
        self.assigns      = np.zeros(data.shape[0], dtype=int) # assignments of data points to components
        self.params        = {0: Gaussian(self.data, self.hyparams)} # components dictionary
        self.alpha        = alpha
        self._K           = 1  # number of components
        self.n_samples    = np.zeros(self._K, dtype=int)  # number of samples in each component
        self.n_samples[0] = data.shape[0]
        self.marginallikes  = []
        self.accs = []
        self.sample_alpha = sample_alpha
        self._prior_component = Gaussian(np.zeros((0, data.shape[1])), self.hyparams) # prior component for new components

        if self.sample_alpha:
            self.alpha_a = alpha_a
            self.alpha_b = alpha_b

    def _sample_z(self, i):
        x = self.data[i]
        old_k = self.assigns[i]
        self.params[old_k].rm_point(x)
        assert self.params[old_k].nk >= 0
        # # Delete empty component
        # if self.params[k].rm_point(x)==0:
        #     del self.params[k]
        #     for j,v in self.assigns.items():
        #         self.assigns[j] -= int(v>k) # self.assigns = {sample_index: component_index}

        n_comp = len(self.params)

        # use log-pdf to accelerate computation
        posterior_prob_log = [np.log(self.params[k].nk) + self.params[k].logpdf(x) for k in range(n_comp)] # existing components
        
        new_comp = Gaussian(np.zeros((0, len(x))), self.hyparams) # new component
        posterior_prob_log.append(np.log(self.alpha) + new_comp.logpdf(x))

        posterior_prob = np.exp(posterior_prob_log - logsumexp(posterior_prob_log)) # normalize to get probabilities
        #print("Posterior probabilities:", posterior_prob)

        cdf = np.cumsum(np.array(posterior_prob))
        
        new_k = np.uint8(np.nonzero(cdf >= np.random.random())[0][0]) # draw new assignment from the multinomial distribution

        self.assigns[i] = new_k

        if new_k == n_comp:
            self.params[new_k] = new_comp
            self._K += 1

        self.params[new_k].add_point(x)

    # def _sample_z(self, i: int) -> None:
    #     x = self.data[i]
    #     old_k = self.assigns[i]
    #     self.params[old_k].rm_point(x)

    #     keys = list(self.params.keys())
    #     log_probs = np.empty(len(keys) + 1, dtype=np.float64)

    #     for idx, k in enumerate(keys):
    #         nk = self.params[k].nk
    #         if nk == 0:
    #             log_probs[idx] = -np.inf
    #         else:
    #             log_probs[idx] = math.log(nk) + self.params[k].logpdf(x)

    #     log_probs[-1] = math.log(self.alpha) + self._prior_component.logpdf(x)

    #     probs = np.exp(log_probs - logsumexp(log_probs))
    #     new_comp_index = np.searchsorted(np.cumsum(probs), np.random.random())

    #     if new_comp_index == len(keys):
    #         new_k = self._K
    #         self.params[new_k] = Gaussian(np.zeros((0, self.data.shape[1])), self.hyparams)
    #         self._K += 1
    #     else:
    #         new_k = keys[new_comp_index]

    #     self.assigns[i] = new_k
    #     self.params[new_k].add_point(x)

    def gibbs(self, iterations=1, snapshot_interval=100, y=None):
        for iter in range(iterations):
            print('iteration:', iter)

            if iter % snapshot_interval == 0 or iter == iterations - 1:
                
                print("Alpha:", self.alpha)
                print("Number of components:", self._K)
            if self.sample_alpha:
                alpha = sample_concentration(self.n_samples, self.alpha_a, self.alpha_b)
                self.alpha = alpha
            for i in np.random.permutation(range(len(self.data))):
                self._sample_z(i)
            self.n_samples = np.zeros(self._K, dtype=int)
            for comp_id, comp in self.params.items():
                self.n_samples[comp_id] = comp.nk
            
            # print('n_samples before compact:', self.n_samples)
            # print('assigns before compact:', self.assigns)
            self.compact_params()
            # print('n_samples after compact:', self.n_samples)
            # print('assigns after compact:', self.assigns)
            self.marginallikes.append(self.get_logpdf(self.data))

            if y is not None and (iter % 100 == 0 or iter == iterations - 1):
                acc, assignments = unsupervised_clustering_accuracy(y.numpy().astype(int), self.assigns.astype(int))
                self.accs.append(acc)
                print("Clustering accuracy:", acc)

    def compact_params(self):
        # find unused and used topics
        unused_topics = np.nonzero(np.array(self.n_samples) == 0)[0] # topics (global components) that are not assigned data
        # print("Unused topics:", unused_topics)
        used_topics = np.nonzero(np.array(self.n_samples) != 0)[0] # remove unused topics from params
        # print("Used topics:", used_topics)

        self._K -= len(unused_topics)
        assert(self._K >= 1 and self._K == len(used_topics))

        self.n_samples = np.delete(self.n_samples, unused_topics)
        assert(len(self.n_samples) == self._K)

        new_params = {}
        for k in range(len(used_topics)):
            new_params[k] = self.params.pop(used_topics[k])
        self.params = new_params

        for k in range(self._K):
            self.assigns[np.nonzero(self.assigns == used_topics[k])[0]] = k

    # marginal likelihood of data points X
    def get_logpdf(self, data=None):
        if data is None:
            data = self.data
        weights, dists = dict2mix(self.params)
        tmp = [all_loglike(X, weights, dists) for X in data]
        loglik = np.sum(tmp)
        
        # add likelihood of alpha and gamma
        if self.sample_alpha:
            loglik += (self.alpha_a - 1)*np.log(self.alpha) - self.alpha/self.alpha_b - self.alpha_a*np.log(self.alpha_b) - ssp.gammaln(self.alpha_a)
        return loglik

def dict2mix(dic):
    dists = {}
    idx = 0
    weights = np.zeros(len(dic))
    for key in dic:
        if dic[key].nk > 0:
            weights[idx] = 1.*dic[key].nk
            mlt_norm = multivariate_normal(mean=dic[key].mean.A1, cov=dic[key].covar)
            dists[idx] = mlt_norm
            idx += 1
    weights = np.delete(weights, np.arange(len(dists), len(weights)))
    weights /= np.sum(weights)
    return weights, dists


def mixture_logpdf(x, weights, dists):
    from scipy.special import logsumexp
    loglikelihoods = np.zeros(len(dists), dtype=np.float64)
    for key in dists:
        loglikelihoods[key] = np.log(weights[key]) + dists[key].logpdf(x)
    return logsumexp(loglikelihoods)


def all_loglike(X, weights, dists):
    tmp = 0.
    for x in X:
        tmp += mixture_logpdf(x, weights, dists)
    return tmp

def sample_concentration(nk, alpha_a=100., alpha_b=100., max_iter_alpha=20):
    K = len(nk)
    alpha = np.random.gamma(alpha_a, alpha_b) # alpha is draw from a Gamma prior
        
    for iter in range(max_iter_alpha):
        w_alpha = np.random.beta(alpha+1, np.sum(nk))
        pi = alpha_a + K - 1
        s_alpha = np.random.binomial(1, (alpha_b - np.log(w_alpha)) * np.sum(nk) / (pi + (alpha_b - np.log(w_alpha)) * np.sum(nk)))
        # rate = 1./self._gamma_b - np.log(self._w_gamma)
        rate = (alpha_b - np.log(w_alpha))
        alpha = np.random.gamma(alpha_a + K - s_alpha, 1./rate)
    return alpha