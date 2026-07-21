# In this version, the concentration parameters alpha and gamma are given a gamma prior. Thus no need to specify alpha
# and gamma when initiate an instance of Gibbs sampler.
import numpy as np
import math
import pickle
import scipy.special as ssp
from scipy.stats import multivariate_t
from scipy.special import logsumexp, gammaln
from model_loglikelihood import dict2mix, all_loglike
np.seterr(divide='ignore')


class Gaussian: # a variable of the Gaussian class represents a component in the HDP-GMM model, X denotes the data assigned to this component
    def __init__(self, X=np.zeros((0, 2)), kappa_0=1., nu_0=1.0001, mu_0=None, Psi_0=None, prior_mk=0, mk=0):
        self.nk = X.shape[0]
        self.dim = X.shape[1]
        self.prior_mk = prior_mk  # prior number of tables assigned to this component, obtained from the global state
        self.mk = mk  # number of tables assigned to this component in the current data shard, obtained from the local state

        if mu_0 is None:  # initial mean for the cluster
            self._mu_0 = np.zeros((1, self.dim))
        else:
            self._mu_0 = mu_0
        assert(self._mu_0.shape == (1, self.dim))

        self._kappa_0 = kappa_0  # mean fraction

        self._nu_0 = nu_0  # degrees of freedom
        if self._nu_0 <= self.dim:
            self._nu_0 = self.dim

        if Psi_0 is None:
            self._Psi_0 = 1*np.eye(self.dim)
        else:
            self._Psi_0 = Psi_0
        assert(self._Psi_0.shape == (self.dim, self.dim))

        self.mean = None
        self.covar = None
        if X.shape[0] > 0:
            self.fit(X)
        else:
            self.default()

    def default(self):
        # self.mean = np.matrix(np.zeros((1, self.dim)))
        # self.covar = 10.0 * np.matrix(np.eye(self.dim))

        self.mean = np.matrix(self._mu_0)  
        self.covar = (self._kappa_0+1) / (self._kappa_0*(self._nu_0 - self.dim + 1)) * self._Psi_0
        self.covar = np.matrix(self.covar)  # prior mean of covar
        self.ss = {}  # sufficient statistics
        self.ss['nk'] = 0
        self.ss['sum_x'] = np.zeros((1, self.dim))
        self.ss['sum_xx'] = np.zeros((self.dim, self.dim))
       

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
        # self.ss = {}
        # self.ss['nk'] = self.nk 
        # self.ss['x_bar_k'] = x_bar_k  
        # self.ss['S_k'] = S_k 
         
        # store moments (n_k, sum_x, sum_x*x.t()) instead of exact sufficient statistics
        self.ss = {}
        self.ss['nk'] = self.nk
        self.ss['sum_x'] = self._sum
        self.ss['sum_xx'] = self._square_sum

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
        size = len(x)
        assert size == self.mean.shape[1]
        assert (size, size) == self.covar.shape
        det = np.linalg.det(self.covar)
        assert det != 0
        norm_const = 1. / (np.power((2*np.pi), float(size)/2) * np.power(det, .5))
        x_mu = x - self.mean
        res = math.pow(math.e, -.5 * (x_mu * self.covar.I * x_mu.transpose()))
        return norm_const * res

    # use an approximated Gaussian distribution to replace the actual multivariate_t distribution to compute the log pdf
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
    
    # # use multivariate_t distribution to compute the log pdf
    # def logpdf(self, x):
    #     size = len(x)
    #     # for existing components
    #     if hasattr(self, 'ss'):
    #         assert self.ss != {}
    #         # compute posterior parameters
    #         kappa_k = self._kappa_0 + self.nk
    #         nu_k = self._nu_0 + self.nk
    #         x_bar_k = self.ss['x_bar_k']  # x_bar_k
    #         x_bar_k_mu_0 = x_bar_k - self._mu_0 # x_bar_k - mu_0
    #         S_k = self.ss['S_k']  # S_k
    #         Psi_k = self._Psi_0 + S_k + \
    #             self._kappa_0 * self.nk * x_bar_k_mu_0.transpose() * x_bar_k_mu_0 / (self._kappa_0 + self.nk) # Psi_k
    #         mu_k = (self._kappa_0 * self._mu_0 + self.nk * x_bar_k) / (self._kappa_0 + self.nk) # mu_k
    #         mu_k = np.squeeze(np.array(mu_k))  # convert mu_k to a 1D array
    #         # compute the log pdf using multivariate_t distribution
    #         # logpdf_t = multivariate_t.logpdf(x, loc=mu_k, shape=(Psi_k*(kappa_k+1))/(kappa_k*(nu_k-self.dim+1)), df=nu_k)
    #         # print('logpdf_t is ', logpdf_t)

    #         return multivariate_t.logpdf(x, loc=mu_k, shape=(Psi_k*(kappa_k+1))/(kappa_k*(nu_k-self.dim+1)), df=nu_k)
    #     # for new components
    #     else:
    #         # compute prior parameters
    #         mu_0 = np.squeeze(np.array(self._mu_0))  # convert mu_0 to a 1D array
    #         # print('Psi_0', self._Psi_0)
    #         # print('kappa_0', self._kappa_0)
    #         # print('nu_0', self._nu_0)
    #         # print('scale', (self._Psi_0*(self._kappa_0+1))/(self._kappa_0*(self._nu_0-self.dim+1)))
    #         # logpdf_t = multivariate_t.logpdf(x, loc=mu_0, shape=(self._Psi_0*(self._kappa_0+1))/(self._kappa_0*(self._nu_0-self.dim+1)), df=self._nu_0)
    #         # print('logpdf_t is ', logpdf_t)
            
    #         return multivariate_t.logpdf(x, loc=mu_0, shape=(self._Psi_0*(self._kappa_0+1))/(self._kappa_0*(self._nu_0-self.dim+1)), df=self._nu_0)

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

class WorkerLevelGibbsSampler(object):
    """
    @:param snapshot_interval: the interval for exporting a snapshot of the model
    """
    def __init__(self, snapshot_interval=1, compute_loglik=False):
        self._snapshot_interval = snapshot_interval
        self._flag_compute_loglik = compute_loglik

        self._table_info_title = "Table-information-"
        self._dish_info_title = "Dish-information-"
        self._hyper_parameter_title = "Hyper-parameter-"

        self.params = {}

    """
    @param data: a N-by-D np array object, defines N points of D dimension
    @param K: number of topics, number of broke sticks # number of components in the global level
    @param gamma: the smoothing value for a table to be assigned to a new topic
    @param alpha: the smoothing value for a word to be assigned to a new table

    """
   
    def initialize(self, data, global_state=None, hyperparameter=None): 
        # global_state: a dictionary contains the global id, sufficient statistics, and m_k of each global component {global id: nk, sum_x, sum_xx, mk}
        # initialize the documents
        self._corpus = data
        # initialize the size of the collection, i.e., self._D = total number of documents (subsets of data). 
        self._D, n_samples_per_doc, dim = data.shape

        # self._gamma = np.random.gamma(self._gamma_a, self._gamma_b)
        assert set(hyperparameter) == {'alpha', 'gamma', 'mu_0', 'kappa_0', 'nu_0', 'Psi_0'}
        self._alpha = hyperparameter['alpha']
        self._gamma = hyperparameter['gamma']
   
        '''
        word is the data point, table is the local cluster (component), topic is the global cluster (component)
        must ensure each topics in the inputted global state has been assigned to at least one table, and each table should be assigned at least one data point, otherwise the topic and the table will never be assigned data in the sampling process
        '''
        num_empty_topics = 0
        # based on _m_k, initialize document-level stats and determine samples assigned to each topic initially
        if len(global_state.keys()) >= self._D:
            self._K = len(global_state.keys())
            self._m_k = np.zeros(self._K)
            doc_idx = 0
            for topic_idx in range(len(list(global_state.keys()))):
                global_id = list(global_state.keys())[topic_idx]

                assert set(global_state[global_id]) == {'nk', 'sum_x', 'sum_xx', 'm_k'}
                #self._m_k[idx] = global_state[global_id]['m_k']

                nk = global_state[global_id]['nk']

                if nk > 0:
                    x_bar_k = global_state[global_id]['sum_x'] / nk
                elif nk==0:
                    num_empty_topics += 1
                    x_bar_k = np.zeros((1, dim))
                else:
                    raise ValueError("Unexpected value for nk, nk should be non-negative.")

                x_bar_k_mu_0 = x_bar_k - hyperparameter['mu_0']
                S_k = global_state[global_id]['sum_xx'] - nk*(x_bar_k.transpose()*x_bar_k)

                kappa_k = hyperparameter['kappa_0'] + nk
                nu_k = hyperparameter['nu_0'] + nk
                
                mu_k = (hyperparameter['kappa_0'] * hyperparameter['mu_0'] + nk * x_bar_k) / (hyperparameter['kappa_0'] + nk)  
                # x_bar_k - mu_0
                Psi_k = hyperparameter['Psi_0'] + S_k + \
                    hyperparameter['kappa_0'] * nk * x_bar_k_mu_0.transpose() * x_bar_k_mu_0 / (hyperparameter['kappa_0'] + nk) # W_k

                # initialize Gaussian mixtures based on the global state
                
                if doc_idx <= self._D - 1 and nk == 0:
                    self.params[global_id] = Gaussian(X=self._corpus[doc_idx], kappa_0=kappa_k, nu_0=nu_k, mu_0=mu_k, Psi_0=Psi_k, prior_mk=global_state[global_id]['m_k'], mk=global_state[global_id]['m_k'])
                    doc_idx += 1
                    if global_state[global_id]['m_k'] <= 0:
                        self.params[global_id].mk = 1.
                    # self.params[global_id] = Gaussian(X=self._corpus.reshape(-1,self._corpus.shape[-1]), kappa_0=kappa_k, nu_0=nu_k, mu_0=mu_k, Psi_0=Psi_k)
                else:
                    self.params[global_id] = Gaussian(X=np.zeros((0, dim)),kappa_0=kappa_k, nu_0=nu_k, mu_0=mu_k, Psi_0=Psi_k,prior_mk=global_state[global_id]['m_k'], mk=global_state[global_id]['m_k'])
                #print(self._m_k[global_id])
                for comp in self.params.keys():
                    self._m_k[comp] = self.params[comp].mk
                assert(self._m_k[topic_idx] >= 0)
                    # self.params[global_id].default()
                    # self.params[global_id].ss = {'nk': 0, 'sum_x': global_state[global_id]['sum_x'], 'sum_xx': global_state[global_id]['sum_xx']}

        else:
            self._K = self._D
            self._m_k = np.zeros(self._K)

            for idx in range(len(list(global_state.keys()))):
                global_id = list(global_state.keys())[idx]

                assert set(global_state[global_id]) == {'nk', 'sum_x', 'sum_xx', 'm_k'}
    
                nk = global_state[global_id]['nk']
                
                if nk > 0:
                    x_bar_k = global_state[global_id]['sum_x'] / nk
                elif nk==0:
                    num_empty_topics += 1
                    x_bar_k = np.zeros((1, dim))
                else:
                    raise ValueError("Unexpected value for nk, nk should be non-negative.")

                x_bar_k_mu_0 = x_bar_k - hyperparameter['mu_0']
                S_k = global_state[global_id]['sum_xx'] - nk*(x_bar_k.transpose()*x_bar_k)

                kappa_k = hyperparameter['kappa_0'] + nk
                nu_k = hyperparameter['nu_0'] + nk
                
                mu_k = (hyperparameter['kappa_0'] * hyperparameter['mu_0'] + nk * x_bar_k) / (hyperparameter['kappa_0'] + nk)  
                # x_bar_k - mu_0
                Psi_k = hyperparameter['Psi_0'] + S_k + \
                    hyperparameter['kappa_0'] * nk * x_bar_k_mu_0.transpose() * x_bar_k_mu_0 / (hyperparameter['kappa_0'] + nk) # W_k

                # initialize Gaussian mixtures based on the global state
                
                self.params[global_id] = Gaussian(X=self._corpus[idx], kappa_0=kappa_k, nu_0=nu_k, mu_0=mu_k, Psi_0=Psi_k, prior_mk=global_state[global_id]['m_k'], mk=global_state[global_id]['m_k'])
                
                if global_state[global_id]['m_k'] <= 0:
                    self.params[global_id].mk = 1
                
            for idx in range(len(global_state.keys()),self._D):
                if idx not in list(global_state.keys()):
                    self.params[idx] = Gaussian(X=self._corpus[idx])
                else:
                    max_global_id = max(list(global_state.keys()))
                    self.params[idx-len(global_state.keys())+max_global_id+1] = Gaussian(X=self._corpus[idx])
                self.params[idx].mk = 1

            for comp in self.params.keys():
                self._m_k[comp] = self.params[comp].mk
            assert(self._m_k[idx] >= 0)
            # initialize the word count matrix indexed by topic id and word id, i.e., n_{\cdot \cdot k}^v
            # self._n_kv = np.zeros((self._K, self._V))
            # initialize the word count matrix indexed by topic id and document id, i.e., n_{j \cdot k}
        
        assert(num_empty_topics == data.shape[0]), ('the number of empty topics is not equal to the number of documents in the local worker')

        self._n_kd = np.zeros((self._K, self._D))
        self._t_dv = {}
        self._k_dt = {}
        self._n_dt = {}

        data_assigned = {k: nk for k, nk in zip(self.params.keys(), [self.params[k].nk for k in self.params.keys()])}
        topics_assigned_data = [k for k in data_assigned.keys() if data_assigned[k] > 0]
        assert(len(topics_assigned_data) == self._D), ('the number of topics assigned data is not equal to the number of documents in the local worker')

        # we assume all words in a document belong to one table which is assigned to topic d, where d is the index of document 
        # (subsets of data)
        for d in range(self._D):
            # initialize the table information vector indexed by document and records down which table a word belongs to
            # local-level cluster assignments
            self._t_dv[d] = np.zeros(len(self._corpus[d]), dtype=int).astype(int)

            # self._k_dt records down which topic a table was assigned to  k_dt = k means that the table (local cluster) t in document (subset) d is assigned to topic (global cluster) k
                 
            self._k_dt[d] = np.array([topics_assigned_data[d]]).astype(int)
            self._n_kd[topics_assigned_data[d], d] = data_assigned[topics_assigned_data[d]] # number of words in document d belong to topic k
            #self._k_dt[d] = np.array([d]).astype(int)
            assert(len(self._k_dt[d]) == len(np.unique(self._t_dv[d])))

            # word_count_table records down the number of words sit on every table
            self._n_dt[d] = np.zeros(1, dtype=int) + len(self._corpus[d])
            assert(len(self._n_dt[d]) == len(np.unique(self._t_dv[d])))
            assert(np.sum(self._n_dt[d]) == len(self._corpus[d]))

            # self._n_kd[d, d] = len(self._corpus[d]) # number of words in document d belong to topic k

    def set_global_snapshot(self, global_state, hyperparameter):
        """
        Refresh global dish priors from the parameter server. This is called
        after every synchronization round. It must NOT reset local assignments.
        global_state: {gid: {'n': float, 'sum_x': (D,), 'sum_xx': (D,D), 'm': float}, ...}
        hyperparameter: {'alpha','gamma','mu_0','kappa_0','nu_0','Psi_0'}
        """
        num_local_topics = len(self.params)
        num_global_topics = len(global_state.keys())

        if num_global_topics < num_local_topics:
            print(f'num_global_topics:{num_global_topics}')
            print(f'num_local_topics:{num_local_topics}')
        assert(num_global_topics>=num_local_topics), ('global topics less than local topics')

        
        dim = global_state[0]['sum_x'].shape[1]
        # update the hyperparameters for each topic, donot change the samples assigned to that topic
        # print(f'global_id:{global_state.keys()}')
        for global_id in global_state.keys():
                
            assert set(global_state[global_id]) == {'nk', 'sum_x', 'sum_xx', 'm_k'}

            # # update m_k
            # if global_state[global_id]['m_k'] > 0:
            #     new_m_k[global_id] = global_state[global_id]['m_k']
            # else:
            #     new_m_k[global_id] = self._m_k[global_id]

            nk = global_state[global_id]['nk']

            if nk > 0:
                x_bar_k = global_state[global_id]['sum_x'] / nk
            else:
                x_bar_k = np.zeros((1, dim))
            
            x_bar_k_mu_0 = x_bar_k - hyperparameter['mu_0']
            S_k = global_state[global_id]['sum_xx'] - nk*(x_bar_k.transpose()*x_bar_k)

            kappa_k = hyperparameter['kappa_0'] + nk
            nu_k = hyperparameter['nu_0'] + nk
            
            mu_k = (hyperparameter['kappa_0'] * hyperparameter['mu_0'] + nk * x_bar_k) / (hyperparameter['kappa_0'] + nk)  
            # x_bar_k - mu_0
            Psi_k = hyperparameter['Psi_0'] + S_k + \
                hyperparameter['kappa_0'] * nk * x_bar_k_mu_0.transpose() * x_bar_k_mu_0 / (hyperparameter['kappa_0'] + nk) # W_k
            
            if global_id <= num_local_topics-1:
                #assert(hasattr(self.params[global_id],'_X')), ('local topic is not assigned data')
                if hasattr(self.params[global_id],'_X') and len(self.params[global_id]._X) > 0:
                    data = self.params[global_id]._X
                    exist_mk = self.params[global_id].mk - self.params[global_id].prior_mk # actual number of tables assigned to this topic in the local data shard, obtained from the previous local state
                    new_params = Gaussian(X=data, kappa_0=kappa_k, nu_0=nu_k, mu_0=mu_k, Psi_0=Psi_k, prior_mk=global_state[global_id]['m_k'], mk=exist_mk+global_state[global_id]['m_k'])
                else:
                    new_params = Gaussian(X=np.zeros((0, dim)),kappa_0=kappa_k, nu_0=nu_k, mu_0=mu_k, Psi_0=Psi_k, prior_mk=global_state[global_id]['m_k'], mk=global_state[global_id]['m_k'])
            else:
                new_params = Gaussian(X=np.zeros((0, dim)),kappa_0=kappa_k, nu_0=nu_k, mu_0=mu_k, Psi_0=Psi_k, prior_mk=global_state[global_id]['m_k'], mk=global_state[global_id]['m_k'])
            
            self.params[global_id] = new_params
        new_m_k = np.zeros(num_global_topics)
        for comp in self.params.keys():
            new_m_k[comp] = self.params[comp].mk

        self._m_k = new_m_k
        self._K = new_m_k.shape[0]
        # print('new_mk', new_m_k)
        new_n_kd = np.zeros([self._K, self._D])
        for comp in self.params.keys():
            if hasattr(self.params[comp], '_X'):
                for k in range(self._n_kd.shape[0]):
                    for d in range(self._D):
                        new_n_kd[k,d] = self._n_kd[k,d]
        # print('new_n_kd', new_n_kd)
        self._n_kd = new_n_kd

        # the "empty" topics are retained in the synchronization step, so before the next local sampling iteration, we need to compact the parameters
        self.compact_params()


    '''
    sample concentration parameters alpha and gamma from gamma priors
    '''
    # def sample_concentration_params(self, alpha_a=10., alpha_b=1., gamma_a=1., gamma_b=1., max_iter_alpha=20, max_iter_gamma=20):
    #     self._alpha_a = alpha_a
    #     self._alpha_b = alpha_b
    #     self._gamma_a = gamma_a
    #     self._gamma_b = gamma_b

    #     self._alpha = np.random.gamma(self._alpha_a, self._alpha_b)
    #     self._gamma = np.random.gamma(self._gamma_a, self._gamma_b)

    #     for iter_alpha in range(max_iter_alpha):
    #         self._w_alpha = np.array([np.random.beta(self._alpha+1, np.sum(self._n_dt[j])) for j in range(self._D)])
    #         self._s_alpha = np.array(
    #             [np.random.binomial(1, np.sum(self._n_dt[j]) / (np.sum(self._n_dt[j]) + self._alpha)) for j in
    #             range(self._D)])
    #         # sample alpha from the gamma distribution
    #         #rate = 1./self._alpha_b - np.sum(np.log(self._w_alpha))
    #         rate = (self._alpha_b - np.sum(np.log(self._w_alpha)))
    #         self._alpha = np.random.gamma(self._alpha_a + np.sum(self._m_k) - np.sum(self._s_alpha), 1./rate)
    #         # print "alpha is ", self._alpha
    #     # sample gamma from a gamma distribution
    #     for iter_gamma in range(max_iter_gamma):
    #         self._w_gamma = np.random.beta(self._gamma+1, np.sum(self._m_k))
    #         pi = self._gamma_a + self._K - 1
    #         # np.random.binomial(n,p), draw n samples from the binomial distribution, P(X=1)=p, P(X=0)=1-p, select a Gamma density as the posterior of gamma is a mixture of two Gamma densities
    #         self._s_gamma = np.random.binomial(1, (self._gamma_b - np.log(self._w_gamma)) * np.sum(self._m_k) / (
    #         pi + (self._gamma_b - np.log(self._w_gamma)) * np.sum(self._m_k))) 
    #         # rate = 1./self._gamma_b - np.log(self._w_gamma)
    #         rate = (self._gamma_b - np.log(self._w_gamma))

    #         # in np.random.gamma, it uses the shape–scale parameterization, i.e., np.random.gamma(shape, scale); in Escobar–West, it uses the shape–rate parameterization, so here we use scale = 1./rate
    #         self._gamma = np.random.gamma(self._gamma_a + self._K - self._s_gamma, 1./rate)
            # print "gamma is ", self._gamma

    """
    sample the data to train the parameters
    @param iteration: the maximum number of gibbs sampling iteration
    """
    def sample(self, iteration): 
        # global state: a dictionary contains the global id, sufficient statistics, and m_k of each global component {global id: nk, sum_x, sum_xx, mk}
        # hyparameter: a dictionary contains global and local concentration parameters, gamma and alpha
        # sample the total data
        iter = 0
        # max_iter_alpha = 20
        # max_iter_gamma = 20
        # store log-likelihood for each iteration
        self.log_likelihoods = np.zeros(iteration)
        while iter < iteration:
            iter += 1
            
            # if not hasattr(self,'_alpha'): # user-specified alpha and gamma
            #     print('sampling concentration parameters ...')
            #     self.sample_concentration_params()
            assert(hasattr(self,'_alpha') and hasattr(self,'_gamma')), ('concentration parameters alpha and gamma are not set yet')

            for document_index in np.random.permutation(range(self._D)): # self._D number of documents (datasets)
                # sample customer assignment, see which table it should belong to
                for word_index in np.random.permutation(range(len(self._corpus[document_index]))):
                    # remove x_ji
                    self.update_params(document_index, word_index, -1)  # update_parameters of each global component by substracting x_ji or x_jt # update = +1 or -1, add or remove a data point

                    # get the data at the index position
                    x = self._corpus[document_index][word_index]

                    # compute the log-likelihood log[fk_-xji(xji)], self._K = number of global components
                    f = np.zeros(self._K, dtype=np.float64)
                    flog = np.zeros(self._K, dtype=np.float64)
                    f_new_table = 0.
                    for k in range(self._K):
                        flog[k] = self.params[k].logpdf(x)
                        f[k] = np.exp(flog[k])
                        f_new_table += self._m_k[k]*f[k]

                    base_distribution = Gaussian(X=np.zeros((0, len(x)))) # prior density
                    f_new_topic = np.exp(base_distribution.logpdf(x))
                    f_new_table += self._gamma*f_new_topic
                    # p(x_ji | t_-ji, t_ji=t_new, k)
                    f_new_table /= (np.sum(self._m_k) + self._gamma)    # normalization
                    # assert f_new_table > 0.

                    # compute the prior probability of this word sitting at every table # prior probability of assigning data points to local components
                    # table_probability = np.zeros(len(self._k_dt[document_index]) + 1) # len(self._k_dt[document_index]) number of existing tables, +1: adding a new table
                    # self._k_dt: global assignment variables, dictionary {document(subset) id: [tables (local components) in each document (subset) assigned to topics (global components)]}
                    table_probability_log = np.zeros(len(self._k_dt[document_index]) + 1, dtype=np.float64)
                    for t in range(len(self._k_dt[document_index])):
                        if self._n_dt[document_index][t] > 0: # number of words (data points) in document (subset) [document_index] assigned to table (local component) t
                            # if there are some words sitting on this table,
                            # the probability will be proportional to the population
                            # global assignment (table t in document [document_index] assigned to topic _k_dt[document_index][t] (assigned_topic))
                            assigned_topic = int(self._k_dt[document_index][t]) 
                            assert(assigned_topic >= 0 or assigned_topic < self._K)
                            # table_probability[t] = f[assigned_topic] * self._n_dt[document_index][t]
                            # p(t_ji = t | t_-ji, k)
                            table_probability_log[t] = flog[assigned_topic] + np.log(self._n_dt[document_index][t])
                        else:
                            # if there are no words sitting on this table
                            # note that it is an old table, hence the prior probability is 0, not self._alpha
                            # table_probability[t] = 0.
                            table_probability_log[t] = np.log(0.)
                            
                    # compute the prob of current word sitting on a new table, the prior probability is self._alpha
                    # table_probability[len(self._k_dt[document_index])] = self._alpha * f_new_table
                    # p(t_ji = t_new)
                    table_probability_log[len(self._k_dt[document_index])] = np.log(self._alpha) + np.log(f_new_table)

                    # sample a new table this word should sit in, i.e., sample t_ji based on the probability of local assignment
                    table_probability = np.exp(table_probability_log)
                    assert np.sum(table_probability) > 0.
                    table_probability /= np.sum(table_probability)
                    cdf = np.cumsum(table_probability)
                    # draw a sample (t_ji) from the categorical distribution cat(table_probability)
                    new_table = np.uint8(np.nonzero(cdf >= np.random.random())[0][0])

                    # assign current word to new table
                    self._t_dv[document_index][word_index] = new_table # t_ji = t
 
                    # if current word sits on a new table, we need to get the topic of that table
                    # if t_ji = t_new, sample a global component (k_jt_new) for t_ji (as it is a new table)
                    if new_table == len(self._k_dt[document_index]):
                        # expand the vectors to fit in new table
                        # add one dimension as a new table is activated
                        self._n_dt[document_index] = np.hstack((self._n_dt[document_index], np.zeros(1)))
                        self._k_dt[document_index] = np.hstack((self._k_dt[document_index], np.zeros(1)))                       

                        assert(len(self._n_dt) == self._D and np.all(self._n_dt[document_index] >= 0)) # check each table (local component) has data
                        assert(len(self._k_dt) == self._D and np.all(self._k_dt[document_index] >= 0)) # check each topic (global component) has data
                        assert(len(self._n_dt[document_index]) == len(self._k_dt[document_index]))

                        # compute the probability of this table having every topic # global assignment
                        # topic_probability = np.zeros(self._K + 1)
                        topic_probability_log = np.zeros(self._K + 1, dtype=np.float64)
                        # responsibilities for existing topics (global components)
                        # p(k_jt_new=k | t, k_-jt_new)
                        for k in range(self._K):
                            # topic_probability[k] = self._m_k[k] * f[k]
                            topic_probability_log[k] = np.log(self._m_k[k]) + flog[k] # self._m_k[k] = number of tables (local components) equals topic (global component) k
                        # topic_probability[self._K] = self._gamma * f_new_topic
                        topic_probability_log[self._K] = np.log(self._gamma) + np.log(f_new_topic)

                        # sample a new topic this table should be assigned
                        topic_probability = np.exp(topic_probability_log)
                        assert np.sum(topic_probability) > 0.
                        topic_probability /= np.sum(topic_probability)
                        cdf = np.cumsum(topic_probability)

                        # sample a topic (global component), which can be either an existing topic or a new topic
                        # draw a sample k_jt_new from categorical(topic_probability) 
                        new_topic = np.uint8(np.nonzero(cdf >= np.random.random())[0][0]) 
                        self._k_dt[document_index][new_table] = new_topic

                        # if current table requires a new topic
                        if new_topic == self._K:
                            # expand the matrices to fit in new topic
                            self._K += 1
                            self._n_kd = np.vstack((self._n_kd, np.zeros((1, self._D))))
                            assert(self._n_kd.shape == (self._K, self._D))
                            self._k_dt[document_index][-1] = new_topic
                            self._m_k = np.hstack((self._m_k, np.zeros(1)))
                            assert(len(self._m_k) == self._K)
                            self.params[new_topic] = Gaussian(X=np.zeros((0, len(x)))) # add a new component
                    
                    # first remove the point x_ji, update params, and calulate assignment variables t_ji, k_jt for the point
                    # then add the point x_ji to table t_ji and topic k_jt, finally updata params
                    self.update_params(document_index, word_index, +1)

                # sample table assignment, see which topic it should belong to 
                # sample k_jt (global assignment)
                # print('iteration', iter, 'document', document_index)
                # print('mk before sample k_dt', self._m_k)
                # print('k_dt',self._k_dt)
                # print('t_dv',self._t_dv)
                topic_iter = 0
                for table_index in np.random.permutation(range(len(self._k_dt[document_index]))):
                    topic_iter += 1
                    # if this table is not empty, sample the topic assignment of this table
                    if self._n_dt[document_index][table_index] > 0:
                        old_topic = int(self._k_dt[document_index][table_index])

                        # find the index of the words sitting on the current table  
                        selected_word_index = np.nonzero(self._t_dv[document_index] == table_index)[0] # find all data points in document j with local assignment variable t_ji = [table_index]
                        # find all the data associated with current table X_jt
                        selected_word = np.array([self._corpus[document_index][term]
                                                  for term in list(selected_word_index)])
                        # remove all the data in this table from their cluster
                        for x in selected_word:
                            self.params[old_topic].rm_point(x)

                        # compute the probability of assigning current table every topic
                        topic_probability_log = np.zeros(self._K + 1, dtype=np.float64)
                        # first compute the log-likelihood of a new topic
                        topic_probability_log[self._K] = 0.
                        for x in selected_word:
                            base_distribution = Gaussian(X=np.zeros((0, len(x))))
                            topic_probability_log[self._K] += base_distribution.logpdf(x)
                        topic_probability_log[self._K] += np.log(self._gamma)

                        # compute the likelihood of each existing topic
                        for topic_index in range(self._K):
                            if topic_index == old_topic:
                                if self._m_k[topic_index] <= 1:
                                    # if current table is the only table assigned to current topic,
                                    # it means this topic is probably less useful or less generalizable to other documents,
                                    # it makes more sense to collapse this topic and hence assign this table to other topic.
                                    topic_probability_log[topic_index] = -1e500
                                else:
                                    # if there are other tables assigned to current topic
                                    # topic_probability[topic_index] = 0.
                                    for x in selected_word:
                                        assert self.params[topic_index].logpdf(x) > -np.inf
                                        topic_probability_log[topic_index] += self.params[topic_index].logpdf(x)
                                    # compute the prior if we move this table from this topic
                                    assert self._m_k[topic_index] - 1 > 0
                                    topic_probability_log[topic_index] += np.log(self._m_k[topic_index] - 1)
                            else:
                                # topic_probability[topic_index] = 0.
                                for x in selected_word:
                                    assert self.params[topic_index].logpdf(x) > -np.inf
                                    topic_probability_log[topic_index] += self.params[topic_index].logpdf(x)
                                # if self._m_k[topic_index] <= 0:
                                #     print('topic', topic_index, 'm_k', self._m_k[topic_index])
                                assert self._m_k[topic_index] >= 0, ('m_k less than 0 when sampling k_dt')
                                topic_probability_log[topic_index] += np.log(self._m_k[topic_index])

                        # normalize the distribution and sample new topic assignment for this topic
                        # if len(np.where(topic_probability <= -np.inf)[0]) != 0:
                        #    print np.where(topic_probability <= -np.inf)[0]
                        #    print 'number of topics ', self._K
                        # assert np.all(topic_probability > -np.inf)
                        topic_probability_log_normalized = topic_probability_log - logsumexp(topic_probability_log)  # for numerical stability
                        topic_probability = np.exp(topic_probability_log_normalized)
                        # print('topic_probability_log_normalized is ', topic_probability_log_normalized)
                        # print('topic_probability is ', topic_probability)

                        # topic_probability = np.exp(topic_probability_log)
                        
                        # assert np.sum(topic_probability) > 0.
                        # topic_probability /= np.sum(topic_probability)

                        cdf = np.cumsum(topic_probability)
                        rdm = np.random.random()
                        if len(np.nonzero(cdf >= rdm)[0]) == 0:
                            print('topic probability',topic_probability)
                            print('topic probability_log',topic_probability_log)
                            print('log_sum_exp',logsumexp(topic_probability_log))
                        new_topic = np.uint8(np.nonzero(cdf >= rdm)[0][0])

                        # if the table is assigned to a new topic
                        if new_topic != old_topic:
                            # assign this table to new topic
                            self._k_dt[document_index][table_index] = new_topic

                            # if this table starts a new topic, expand all matrix
                            if new_topic == self._K:
                                self._K += 1
                                self._n_kd = np.vstack((self._n_kd, np.zeros((1, self._D))))
                                assert(self._n_kd.shape == (self._K, self._D))
                                self._m_k = np.hstack((self._m_k, np.zeros(1)))
                                assert(len(self._m_k) == self._K)
                                self.params[new_topic] = Gaussian(X=np.zeros((0, len(selected_word[0]))))

                            # adjust the statistics of all model parameter
                            self._m_k[old_topic] -= 1
                            self._m_k[new_topic] += 1
                            self._n_kd[old_topic, document_index] -= self._n_dt[document_index][table_index]
                            self._n_kd[new_topic, document_index] += self._n_dt[document_index][table_index]
                        # add data point to the cluster
                        for x in selected_word:
                            self.params[new_topic].add_point(x)

            # compact all the parameters, including removing unused topics and unused tables, the compact is done after each iteration of sampling all data
            # print('mk before compact', self._m_k)
            # print('k_dt', self._k_dt)
            # print('n_dt', self._n_dt)
            # print('t_dv', self._t_dv)
            # print('n_kd', self._n_kd)
            
            # for key in self.params.keys():
            #     print('param', key)
            #     print(self.params[key].ss)

            assert(len(self._m_k) == len(self.params.keys())), ('length of m_k and params not equal before compacting')
            for comp in self.params.keys():
                self.params[comp].mk = self._m_k[comp]

            self.compact_params()

            # print('mk after compact', self._m_k)
            # print(self._k_dt)
            # for key in self.params.keys():
            #     print(self.params[key].ss)

            assert(len(self._m_k) == len(self.params.keys())), ('length of m_k and params not equal after compacting')
            self._m_k = np.zeros(self._K)
            for comp in self.params.keys():
                self._m_k[comp] = self.params[comp].mk

            if self._flag_compute_loglik:
                # print "gamma is %.2f, alpha is %.2f" % (self._gamma, self._alpha)
                self.log_likelihoods[iter-1] = self.get_logpdf()
                '''
                if iter >= 2:
                    if self.log_likelihoods[iter-1] < self.log_likelihoods[iter-2]:
                        print "warning: log-likelihood is decreasing..."
                '''
            # output sampling states
            if iter > 0 and iter % self._snapshot_interval == 0:
                print("sampling in progress %2d%%" % (100 * iter / iteration))
                print("total number of topics %i " % self._K)
                if self._flag_compute_loglik:
                    print("gamma is %.2f, alpha is %.2f" % (self._gamma, self._alpha))
                    self.log_likelihoods[iter - 1] = self.get_logpdf()
                    print('model log-likelihood is ', self.log_likelihoods[iter-1])

        
        # finally, order the components
        self.params_ordered = {}
        for i in range(len(self.params.keys())):
            param_id = list(self.params.keys())[i]
            self.params_ordered[i] = self.params[param_id]

            # revise the document-level states based on the ordered components
            # k_dt
            for d in range(self._D):
                tables_at_document_D = self._k_dt[d]
                for table_topic in range(len(tables_at_document_D)):
                    if int(tables_at_document_D[table_topic]) == param_id:
                        self._k_dt[d][table_topic] = float(i)
        self.params = {}
        self.params = self.params_ordered

        # print('params:', self.params.keys())
        # print('ordered_params:', self.params_ordered.keys())
        # print('k_dt:', self._k_dt)
        # print('n_dt', self._n_dt)
        # print('m_k',self._m_k)
        
    """
    @param document_index: the document index to update
    @param word_index: the word index to update
    @param update: the update amount for this document and this word
    @attention: the update table index and topic index is retrieved from self._t_dv and self._k_dt, so make sure these values were set properly before invoking this function
    """
    def update_params(self, document_index, word_index, update): # update_parameters of each global component by substracting x_ji or x_jt # update = +1 or -1, add or remove a data point
        # retrieve the table_id of the current word of current document
        table_id = int(self._t_dv[document_index][word_index]) # t_ji
        # retrieve the topic_id of the table that current word of current document sit on
        topic_id = int(self._k_dt[document_index][table_id]) # k_jt
        # get the data at the word_index of the document_index
        x = self._corpus[document_index][word_index] # x_ji

        self._n_dt[document_index][table_id] += update # number of data points in document [document_index] assigned to table [table_id]
        assert(np.all(self._n_dt[document_index] >= 0))
        # update component parameters by adding or removing x_ji
        if update == -1:
            self.params[topic_id].rm_point(x) 
        elif update == 1:
            self.params[topic_id].add_point(x)

        self._n_kd[topic_id, document_index] += update # _n_kd: n_jk, number of data points in document j assigned to topic (global component) k
        assert(np.all(self._n_kd >= 0))

        # if current table in current document becomes empty
        if update == -1 and self._n_dt[document_index][table_id] == 0: # _n_dt: n_jt, number of data points in document j assigned to table (local component) t,  _n_dt[document_index][table_id] == 0 means that the table is empty and needs to be removed
            # adjust the table counts
            # topic_id: the table [table_id] is assigned to topic [topic_id]
            self._m_k[topic_id] -= 1

        # if a new table is created in current document
        if update == 1 and self._n_dt[document_index][table_id] == 1:
            # adjust the table counts
            self._m_k[topic_id] += 1

        assert(np.all(self._m_k >= 0))
        assert(np.all(self._k_dt[document_index] >= 0))

    def compact_params(self):
        # find unused and used topics
        unused_topics = np.nonzero(self._m_k == 0)[0] # topics (global components) that are not assigned data
        used_topics = np.nonzero(self._m_k != 0)[0]

        self._K -= len(unused_topics)

        if self._K != len(used_topics):
            print('Error in compacting parameters!')
            print('K after subtract', self._K)
            print('unused topics', unused_topics)
            print('used_topics', used_topics)

        assert(self._K >= 1 and self._K == len(used_topics))

        self._n_kd = np.delete(self._n_kd, unused_topics, axis=0)
        assert(self._n_kd.shape == (self._K, self._D))

        self._m_k = np.delete(self._m_k, unused_topics)
        assert(len(self._m_k) == self._K)

        new_params = {}
        for k in range(len(used_topics)):
            new_params[k] = self.params.pop(used_topics[k])
            
        self.params = new_params
        # for k in range(len(used_topics)):
        #     self.params[k] = self.params.pop(used_topics[k])
        # for key in range(len(self.params)):
        #     if key >= self._K and key in unused_topics:
        #         del self.params[key]
        for d in range(self._D):
            # find the unused and used tables
            unused_tables = np.nonzero(self._n_dt[d] == 0)[0]
            used_tables = np.nonzero(self._n_dt[d] != 0)[0]

            self._n_dt[d] = np.delete(self._n_dt[d], unused_tables)
            self._k_dt[d] = np.delete(self._k_dt[d], unused_tables)

            # shift down all the table indices of all words in current document
            # @attention: shift the used tables in ascending order only.
            for t in range(len(self._n_dt[d])):
                self._t_dv[d][np.nonzero(self._t_dv[d] == used_tables[t])[0]] = t

            # shrink down all the topics indices of all tables in current document
            # @attention: shrink the used topics in ascending order only.
            for k in range(self._K):
                self._k_dt[d][np.nonzero(self._k_dt[d] == used_topics[k])[0]] = k

    def get_logpdf(self, data=None):
        if data is None:
            data = self._corpus
        weights, dists = dict2mix(self.params)
        tmp = [all_loglike(X, weights, dists) for X in data]
        loglik = np.sum(tmp)
        
        # add likelihood of alpha and gamma
        if hasattr(self, '_alpha_a'):
            loglik += (self._alpha_a - 1)*np.log(self._alpha) - self._alpha/self._alpha_b - self._alpha_a*np.log(self._alpha_b) - ssp.gammaln(self._alpha_a)
            loglik += (self._gamma_a - 1)*np.log(self._gamma) - self._gamma/self._gamma_b - self._gamma_a*np.log(self._gamma_b) - ssp.gammaln(self._gamma_a)
        return loglik
    
    def create_k_dv(self, transform = False):
        """
        Create _k_dv assignment variable from sampler._k_dt and sampler._t_dv
        
        Returns:
            dict: _k_dv where keys are document indices and values are arrays
                of topic assignments for each word in the document
        """
        self._k_dv = {}
        
        for d in self._t_dv:  # For each document
            if d not in self._k_dt:
                continue
                
            # Get table assignments for this document
            table_assignments = self._k_dt[d].astype(int)
            # Get word-to-table assignments for this document
            word_table_assignments = self._t_dv[d]
            
            # Map each word to its topic through its table assignment
            word_topic_assignments = []
            for t in word_table_assignments:
                # Tables are 0-indexed in _t_dv, so we use t directly
                if t < len(table_assignments):
                    word_topic_assignments.append(table_assignments[t])
                else:
                    # Handle case where table assignment is missing (shouldn't happen in proper HDP)
                    word_topic_assignments.append(-1)  # -1 indicates unassigned
            
            self._k_dv[d] = np.array(word_topic_assignments)

            if transform:
                self._k_dv_array = np.concatenate([v for v in self._k_dv.values()])