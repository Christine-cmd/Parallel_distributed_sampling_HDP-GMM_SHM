import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from sklearn.cluster import KMeans

def mk_hp_prior(data, **kwargs):
    '''init_parameters: initial parameters for the prior distribution, 
    a dictionary including alpha, and parameters of the normal wishart distributio (beta0, m0, covariance, lambda_p)'''
    # hp_prior: hyperparameters for prior distribution
    hp_prior = {}
    D, N = data.shape

    if 'init_with_kmeans' in kwargs:
        assert kwargs['k_means_clusters'] is not None, "k_means_clusters must be provided when init_with_kmeans is True."
        T = kwargs['k_means_clusters']
        hp_prior['m0'] = KMeans(n_clusters=T).fit(data.T).cluster_centers_[::-1]

    else:
        hp_prior['alpha'] = torch.tensor(1)
        hp_prior['beta0'] = torch.tensor(0.01)
        hp_prior['m0'] = torch.mean(data, dim=1).unsqueeze(1)
        covariance0 = torch.cov(data)
        diagonal_elements = torch.diag(covariance0)
        diagonal_covariance = torch.diag(diagonal_elements)
        covariance = diagonal_covariance
        lambda_p = torch.tensor(1)

        if D > 16:
            _ , max_eig = power_method(covariance)
            # eigvals, _ = torch.linalg.eig(covariance)
            # max_eig = torch.max(eigvals[:, 0])
        else:

            max_eig = torch.max(torch.linalg.eig(covariance)[0].real)

        hp_prior['lambda0'] = lambda_p * D + 2
        hp_prior['inv_W0'] = hp_prior['lambda0'] * max_eig * torch.eye(D) * hp_prior['beta0']
        hp_prior['W0'] = torch.inverse(hp_prior['inv_W0'])

    hp_prior.update(kwargs)
    
    return hp_prior
    
def detln(X): # calculate log_determinant
    """
    Calculate the log determinant of matrix X.

    Args:
        X (numpy.ndarray): Input matrix.

    Returns:
        float: Log determinant of X.
    """
    try:
        L = torch.linalg.cholesky(X)
    except torch.linalg.LinAlgError:
        raise ValueError("Error in Choleski decomposition for detln")

    diag_L = torch.diag(L)
    log_det = 2 * torch.sum(np.log(diag_L))
    return log_det

def power_method(A, start=None, precision=1e-10): # return eigenvalues and the maximum eigenvalue
    if start is None:
        start = torch.ones(len(A), 1)
    
    diff = precision + 1
    x = start
    n = torch.norm(x) + diff
    i = 0
    
    while diff > precision:
        i += 1
        y = torch.matmul(A, x)
        n2 = torch.norm(x)
        diff = abs(n2 - n)
        n = n2
        
        if n < 1.0e-200:
            x = torch.zeros(len(A), 1)
            break
        else:
            x = y / n
        
        if i > 100:
            break
    
    n = torch.norm(x)
    if n < 1.0e-200:
        vec = torch.zeros(len(A), 1)
    else:
        vec = x / n
    
    return vec, n

def rand_q_of_z(data, K, opts):
    # q_of_z: N*K responsibilities
    
    N = data.shape[1]

    if opts['algorithm'] == 'vdp':
        q_of_z = torch.zeros(N, K + 1)
    else:
        q_of_z = torch.zeros(N, K)

    q_of_z[:, :K] = torch.rand(N, K)
    
    q_of_z = normalize(q_of_z, 1)
    
    return q_of_z

def normalize(m, dim):
    # set the sum of responsibility of each data point to 1, i.e. sum(q_of_z, dim = 1) = 1
    # Return m normalized along the specified 'dim'
    #
    # e.g.
    # m: i by j by k by ...
    # m = torch.sum(normalize(m, 2), dim=2)
    # m[i, :, k, ...] = torch.ones(1, J, 1, ...)

    dims = [1] * len(m.shape)
    dims[dim] = m.shape[dim]
    m = m / torch.sum(m, dim=dim, keepdim=True).repeat(dims)
    return m

# likelihood function
def log_p_of_x_given_k(data, clusters, hp_posterior): # log(p(x|c)
    D, N = data.shape
    K = len(clusters)
    log_p_of_x_given_k = np.zeros((K, N))
    
    for i in range(K):
        k = clusters[i]
        m = hp_posterior['m'][:, k]
        precision = hp_posterior['W'][:, :, k] @ hp_posterior['lambda'][k]
        d = data - np.tile(m, (1, N))
        log_p_of_x_given_k[i, :] = (-D * 0.5) * np.log(2 * np.pi) + 0.5 * detln(precision) - 0.5 * torch.sum(d * (precision @ d), axis=0)
    
    return log_p_of_x_given_k

# rename the variables to make it consistent with the original manuscript
def mk_q_of_z(data, hp_posterior, hp_prior, opts, omega=None):
    # Compute log_lambda if not provided
    # log_lambda(n,i) is S(n,i) in the paper
    if omega is None:
        omega = mk_omega(data, hp_posterior, hp_prior, opts)
    
    # Compute q_of_z (responsibility)
    q_of_z = torch.exp(normalizeln(omega, 1)) # q_of_z = exp{log(omega)-log(sum(exp{omega}))}
    
    return q_of_z, data, omega

def mk_omega(data, hp_posterior, hp_prior, opts):
    
    if opts['algorithm'] == 'vdp':
        if abs(hp_posterior['gamma'][1, -1] - hp_prior['alpha']) > 1.0e-3:
            print(f"hp_posterior.gamma(2, end): {hp_posterior['gamma'][1, -1]}")
            print(f"hp_prior.alpha: {hp_prior['alpha']}")
            diff = hp_prior['alpha'] - hp_posterior['gamma'][1, -1]
            raise ValueError("must be alpha")
        
    D, N = data.shape
    K = hp_posterior['lambda'].shape[1]

    c0 = hp_posterior['lambda'] + 1
    c0 = c0.repeat(D,1)
    d0 = torch.tensor([_+1 for _ in range(D)])
    d0 = d0.reshape(D,1)
    d0 = d0.repeat(1,K)
    psi_sum = torch.sum(torch.special.psi(c0-d0 * 0.5), dim=0)
    
    omega = torch.zeros(N, K)

    for k in range(K):
        
        E_log_p_of_z_given_other_z_k = (
            torch.special.psi(hp_posterior['gamma'][0, k])
            - torch.special.psi(torch.sum(hp_posterior['gamma'][:, k], dim=0))
            + torch.sum(torch.special.psi(hp_posterior['gamma'][1, :k]) - torch.special.psi(torch.sum(hp_posterior['gamma'][:, :k], dim=0)))
        )
        
        Precision = hp_posterior['W'][:, :, k] * hp_posterior['lambda'][0][k]  # D*D
        
        E_log_p_of_x = -0.5 * D * torch.log(torch.tensor(np.pi)) - 0.5 * detln(hp_posterior['inv_W'][:,:,k]) + 0.5 * psi_sum[k] - 0.5 * D / hp_posterior['beta'][0,k] # 1*1 
        
        d = data - hp_posterior['m'][:, k].unsqueeze(1).repeat(1,N) # D*N
        
        E_log_p_of_x = -torch.sum(0.5 * d * (Precision @ d), dim=0) + E_log_p_of_x 
        E_log_p_of_x = E_log_p_of_x.unsqueeze(0) # 1*N
        
        
        omega[:, k] = E_log_p_of_x + E_log_p_of_z_given_other_z_k

    
    omega[:, -1] = omega[:, -1] - torch.log(1 - torch.exp(torch.special.psi(hp_prior['alpha']) - torch.special.psi(1 + hp_prior['alpha']))) 
        
    return omega

def log_sum_exp(x, dim, y=None):
    """
    Compute log(sum(exp(x))) along a specified dimension in a numerically stable way.
    
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to perform the operation.
        y (Optional): This parameter is not used in the original function.
        
    Returns:
        torch.Tensor: Result of log(sum(exp(x))) along the specified dimension.
    """
    # Find the maximum value along the specified dimension
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    
    dims = torch.ones(1,len(x.shape))
    dims[0][dim] = x.shape[dim]
    dims = dims.numpy()
    dims = dims.astype(int)
    # Subtract the maximum value from x for numerical stability
    x = x - x_max.repeat(dims[0][0],dims[0][1])
    
    # Compute the log-sum-exp
    val = x_max + torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))
    return val # N*1

def lpt2lcpt(lpt, dimension):
    """
    Create a log conditional probability table from a log probability table.

    Args:
        lpt (torch.Tensor): Input log probability tensor.
        dimension (int): Dimension along which to normalize (1 or 2).

    Returns:
        torch.Tensor: Log conditional probability tensor.
    """
    assert dimension in [1, 2], "dimension must be either 1 or 2."

    # Adjust dimensions to Python's 0-based indexing
    dimension = dimension - 1
    the_other_dimension = 1 - dimension

    # Permute the tensor to bring the specified dimension to the front
    # lpt N*K
    lpt = lpt.permute(dimension, the_other_dimension)

    # Calculate log_sum_exp along the second dimension (originally specified dimension)
    log_sum_exp_lpt = log_sum_exp(lpt, dim=1)  # Mx1

    # Subtract log_sum_exp_lpt from lpt
    lcpt = lpt - log_sum_exp_lpt.repeat(1,lpt.shape[1])

    # Permute back to the original dimension order
    lcpt = lcpt.permute(dimension, the_other_dimension)

    return lcpt

def normalizeln(M ,dimension):
    
    M = lpt2lcpt(M, dimension)
    
    return M

# we only need to store this summary statistics in the online learning process
def compute_summary_stats(data, q_of_z, opts):
# given a batach of data and the corresponding responsibilities, compute the summary statistics
# data: D*N
# q_of_z: N*K
    sum_stat = {}
    D, N = data.shape
    K = q_of_z.shape[1]

    if opts['algorithm'] == 'vdp':
        true_Nk = q_of_z.sum(dim=0)  # 1*K  Nc is expected number of observations in cluster c
        true_Nk = true_Nk.unsqueeze(0)
        q_of_z[:, -1] = 0 # add a new component without any data to represent the infinite sum of inactive components

    Nk = q_of_z.sum(dim=0).unsqueeze(0) # 1*K
    I = torch.where(torch.squeeze(Nk) > 1e-50)[0]
    inv_Nk = torch.zeros(1, K)
    inv_Nk[:, I] = 1.0 / Nk[:, I] # 1*K

    x_bar_k = torch.mm(data, q_of_z)*inv_Nk.repeat(D,1) # D*K
    Sk = torch.zeros(D, D, K) 
    
    for k in range(K):
        x_minus_x_bar = data - x_bar_k[:, k].unsqueeze(1).repeat(1, N) # D*N
        Sk[:,:,k] = torch.mm((q_of_z[:, k].unsqueeze(1).repeat(1,D).t() * x_minus_x_bar), x_minus_x_bar.t())*inv_Nk[0,k].repeat(D,1)  # (D*N .* D*N) * N*D = D*D

    sum_stat['Nk'] = Nk
    sum_stat['true_Nk'] = true_Nk
    sum_stat['x_bar_k'] = x_bar_k
    sum_stat['Sk'] = Sk
    
    return sum_stat

def mk_hp_posterior(hp_prior, sum_stat, opts):
#     the last component of q_of_z represents the infinite rest of components
#     the last component is the prior.
#     q_of_z: N*K
#     q_of_z(:,end) is the rest of responsibilities.

# sum_stat['Nk'] = Nk 1*K
# sum_stat['x_bar_k'] = x_bar_k D*K
# sum_stat['Sk'] = Sk D*D*K

    threshold_for_N = 1.0e-200
    
    D, K = sum_stat['x_bar_k'].shape
    
    #Nc = Nc.unsqueeze(0)  # 1*K true_Nc != Nc for VDP
    
    #sum_x = torch.mm(data['given_data'] , q_of_z)  # r_nk*xn in PRML (dot mul) D*N @ N*K = D*K
    
    I = torch.where(torch.squeeze(sum_stat['Nk']) > threshold_for_N)[0]
    
    inv_Nk = torch.zeros(1, K)
    
    inv_Nk[:, I] = 1.0 / sum_stat['Nk'][:, I] # 1*K
    
    hp_posterior = {}
    hp_posterior['beta'] = hp_prior['beta0'] + sum_stat['Nk']  # beta0 in PRML
    hp_posterior['m'] = (sum_stat['Nk']*sum_stat['x_bar_k'] + hp_prior['beta0'] * hp_prior['m0'].repeat(1, K)) / hp_posterior['beta'].repeat(D, 1) # D*K
    hp_posterior['W'] = torch.zeros(D, D, K)
    hp_posterior['inv_W'] = torch.zeros(D, D, K)
    hp_posterior['lambda'] = hp_prior['lambda0'] + sum_stat['Nk']  # lambda0 is nu_0 in PRML
    
    for k in range(K):
        #x_minus_x_bar = data - sum_stat['x_bar_k'][:, k].unsqueeze(1).repeat(1, N) # D*N  (x_n-x_bar) in PRML
        x_bar_minus_m0 = sum_stat['x_bar_k'][:, k].unsqueeze(1) - hp_prior['m0']  # (x_bar - m0) in PRML
        # hp_prior['m0'] should have the size D*1 instead of D
        
        hp_posterior['inv_W'][:,:,k] = hp_prior['inv_W0'] + \
                            sum_stat['Sk'][:,:,k] * sum_stat['Nk'][0][k] + \
                            sum_stat['Nk'][0][k] * hp_prior['beta0'] * torch.mm(x_bar_minus_m0, x_bar_minus_m0.t()) / hp_posterior['beta'][0][k] 
        
        hp_posterior['W'][:, :, k] = torch.inverse(hp_posterior['inv_W'][:,:,k].squeeze())
    
    if opts['algorithm'] == 'vdp':
        hp_posterior['gamma'] = torch.zeros(2, K) # 2*K
        hp_posterior['gamma'][0, :] = 1 + sum_stat['true_Nk']
        hp_posterior['gamma'][1, :] = hp_prior['alpha'] + sum_stat['true_Nk'].sum() - torch.cumsum(sum_stat['true_Nk'], dim=1)
    
    #hp_posterior['q_of_z'] = q_of_z  # q_of_z is a N by K matrix where N is  # of given_data
    
    return hp_posterior

# compute the free energy using the summary statistics and the prior, do not use the posterior
def mk_free_energy(data, hp_posterior, hp_prior, sum_stat, opts, fc=None, omega=None):
    
    if fc is None and omega is None:
        fc = mk_E_log_q_p_eta(data, hp_posterior, hp_prior, sum_stat, opts) # accounts for the parameters of the normal-Wishart distribution in the ELBO
        omega = mk_omega(data, hp_posterior, hp_prior, opts)

    N, K = omega.shape
    
    if opts['algorithm'] == 'vdp':
        len_gamma = hp_posterior['gamma'].shape[1]
        
        # accounts for the stick length in the ELBO
        E_log_p_of_V = (torch.special.gammaln(hp_posterior['gamma'].sum(dim=0)) 
                        - torch.special.gammaln(1 + hp_prior['alpha']) 
                        - torch.special.gammaln(hp_posterior['gamma']).sum(dim=0) 
                        + torch.special.gammaln(hp_prior['alpha']) 
                        + ((hp_posterior['gamma'][0, :] - 1) 
                           * (torch.special.psi(hp_posterior['gamma'][0, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0)))) 
                        + ((hp_posterior['gamma'][1, :] - hp_prior['alpha']) 
                           * (torch.special.psi(hp_posterior['gamma'][1, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0)))))
        extra_term = E_log_p_of_V.sum()
        
    else:
        raise ValueError('Unknown algorithm')

    free_energy = extra_term + fc.sum() - log_sum_exp(omega, 1).sum()
    
    return free_energy, omega

def mk_E_log_q_p_eta(data, hp_posterior, hp_prior, sum_stat, opts):
    # returns E_q(eta)[log q(eta)/p(eta)] % Eq.(10.74) and Eq.(10.77) in PRML, eta is the parameters of the normal-Wishart distribution
    # fc : 1 by K
    D = hp_posterior['m'].size(0)
    K = hp_posterior['lambda'].size(1)
    log_det_inv_W = torch.zeros(1, K) # log determinant
    term_eta = torch.zeros(2, K)
    
    for k in range(K):
        log_det_inv_W[0, k] = detln(hp_posterior['inv_W'][:,:,k])
        d = hp_posterior['m'][:, k].unsqueeze(1) - hp_prior['m0']  # D*1 d = (m_k - m0) in PRML 
        term_eta[0, k] = torch.sum(hp_posterior['W'][:, :, k] * (hp_prior['beta0'] * torch.mm(d, d.t())))
        term_eta[1, k] = torch.sum(hp_posterior['W'][:, :, k] * hp_prior['inv_W0']) - D
    
    #print(term_eta)
    # E_q(mu)[log q(mu)/p(mu)]    p(mu) = N(mu|m0, inv(beta0 * precision)), q(mu) = N(mu|m, inv(beta * precision))
    E_log_q_p_mean = (
        0.5 * D * (hp_prior['beta0'] / hp_posterior['beta'] 
                   - torch.log(hp_prior['beta0'] / hp_posterior['beta']) 
                   - 1)
        + 0.5 * hp_posterior['lambda'] * term_eta[0, :]
    )
    
    psi_sum = torch.sum(torch.special.psi(((hp_posterior['lambda'] + 1).repeat(D, 1) - torch.arange(1, D + 1).unsqueeze(1).repeat(1, K))* 0.5),dim=0) # 1*K
    
    E_log_q_p_cov = (
        0.5 * hp_prior['lambda0'] * (log_det_inv_W - detln(hp_prior['inv_W0'])) # 1*K
        + 0.5 * sum_stat['Nk'] * psi_sum
        + 0.5 * hp_posterior['lambda'] * term_eta[1, :].unsqueeze(0)
        + gamma_multivariate_ln(hp_prior['lambda0'].unsqueeze(0) * 0.5, D)
        - gamma_multivariate_ln(hp_posterior['lambda'] * 0.5, D)
    )
    
    # print(f"E_log_q_p_mean:{E_log_q_p_mean}")
    # print(f"E_log_q_p_cov:{E_log_q_p_cov}")
    # print(f"psi_sum:{psi_sum}")
    
    if torch.any(E_log_q_p_mean < -1.0e-3):
        print(E_log_q_p_mean)
        raise ValueError('E_log_q_p_mean is negative.')
    
    if torch.any(E_log_q_p_cov < -1.0e-3):
        print(E_log_q_p_cov)
        raise ValueError('E_log_q_p_cov is negative.')
    
    fc = E_log_q_p_mean + E_log_q_p_cov
    
    return fc


def gamma_multivariate_ln(x, p):
    # x: array(1, K)
    # p: scalar
    #
    # x must be greater than (p-1)/2
    # x should be greater than p/2
    #
    # Gamma_p(x) = pi^(p(p-1)/4) prod_(j=1)^p Gamma(x+(1-j)/2)
    # log Gamma_p(x) = p(p-1)/4 log pi + sum_(j=1)^p log Gamma(x+(1-j)/2)
    
    c0 = torch.tensor([_+1 for _ in range(p)])
    c0 = c0.reshape(p,1)

    K = len(x)
    gammaln_val = torch.special.gammaln(x.repeat(p, 1) + 0.5 * (1 - c0.repeat(1,K)))
    val = p * (p - 1) * 0.25 * torch.log(torch.tensor(np.pi)) + torch.sum(gammaln_val, dim=0)     
    val = val.unsqueeze(0) # 1*K
    
    return val

# Calculate the variational free energy (minus ELBO)
def my_disp(*args):
    # Initialize a persistent variable to keep track of whether the function is disabled
    if not hasattr(my_disp, 'is_disabled'):
        my_disp.is_disabled = False

    # Check if the second argument is provided (to disable the function)
    if len(args) == 2:
        my_disp.is_disabled = args[1]
        return

    # If not disabled, print the input arguments
    if not my_disp.is_disabled:
        print(*args)
        
def disp_status(free_energy, sum_stats, opts):
    if opts['algorithm'] == 'vdp':
        Nk = sum_stats['true_Nk']
    else:
        Nk = sum_stats['Nk']
    Nk = Nk.flatten()
    formatted_Nk = ', '.join(map(lambda x: f"{x:.2f}", Nk.numpy()))
    my_disp(f"F={free_energy:.5g};    Nk=[{formatted_Nk}];")

def greedy_split(data, hp_posterior, hp_prior, sum_stat, q_of_z, opts):
    """
    Perform the greedy algorithm for optimizing free energy.
    
    Args:
        data (dict): Data dictionary containing necessary data.
        hp_posterior (dict): Posterior hyperparameters.
        hp_prior (dict): Prior hyperparameters.
        opts (dict): Options for the algorithm.
        
    Returns:
        tuple: Updated free energy, posterior hyperparameters, and data.
    """

    free_energy, _ = mk_free_energy(data, hp_posterior, hp_prior, sum_stat, opts)
    my_disp('### greedy splitting ###')
    #disp_status(free_energy, sum_stat, opts)
    i = 0
    while True:
        #my_disp('finding the best one....\t')
        # return the responsibilities (local parameters), summary statistics, global parameters after splitting component k 
        # only the parameters for the two splitted components are updated, the parameters for other components are not changed
        new_free_energy, new_hp_posterior, new_data, new_sum_stat, k = find_best_splitting(data, hp_posterior, hp_prior, sum_stat, opts)
        
        if k == -1:
            # k = -1 means no splitting will decrease the free energy (increase the ELBO)
            return free_energy, hp_posterior, data, q_of_z, sum_stat
            #break
        
        #my_disp(f'finding the best one.... done')  
        #my_disp(f"component {k} was split.")
        
        #disp_status(new_free_energy, new_sum_stat, opts)
        # update the free energy, the posterior, the summary statistics for the other components (except for the two splitted components)
        # free_energy, hp_posterior, data, q_of_z, sum_stat
        new_free_energy, new_hp_posterior, new_data, new_q_of_z, new_sum_stat = update_posterior(new_data, new_hp_posterior, hp_prior, new_sum_stat, opts, ite = opts['ite'], do_sort = 1)
        
        # split cpmponents using a greedy manner, until free energy does not decrease
        if not free_energy_decreased(free_energy, new_free_energy, 0, opts) or i >= opts['max_iter_split']:
            my_disp('free_energy not decreased after splitting')
            disp_status(free_energy, sum_stat, opts)
            return free_energy, hp_posterior, data, q_of_z, sum_stat

        free_energy = new_free_energy
        hp_posterior = new_hp_posterior
        data = new_data
        q_of_z = new_q_of_z
        sum_stat = new_sum_stat

        i += 1
    
    #disp_status(free_energy, sum_stat, opts)
    
    #return free_energy, hp_posterior, data, q_of_z, sum_stat

def find_best_splitting(data, hp_posterior, hp_prior, sum_stat, opts):
    
    k_max =  opts['max_split'] # find potential components to be splitted, maximum 10 trials
    
    K = sum_stat['Nk'].shape[1]
    
    candidates = torch.nonzero(sum_stat['Nk'][0,:] >= opts['min_split_size']).flatten() # 
    if len(candidates) == 0:
        return 0, hp_posterior, data, -1

    q_of_z, _ , _ = mk_q_of_z(data, hp_posterior, hp_prior, opts)

    new_free_energy = torch.ones(1, candidates.max().item()+1) * float('inf')
    
    fc = mk_E_log_q_p_eta(data, hp_posterior, hp_prior, sum_stat, opts)
    
    omega = mk_omega(data, hp_posterior, hp_prior, opts)
    
    new_data = {}
    new_q_of_z_cell = {}

    #print(f"candidates:{candidates}")

    for k in candidates[:min(k_max, len(candidates))]:
        k = k.item()
        #my_disp(f'Splitting component {k}...')
        # choose component k to split, and split it into two components, create the new responsibilities with K+1 components (K components before splitting)
        # new_q_of_z (N*K+1) is the new responsibilities after splitting
        new_data0, new_q_of_z, info = split(k, data, q_of_z, hp_posterior, hp_prior, opts) 

        new_data[k] = new_data0
        
        new_k = info['new_k']

        # after splitting, only update the posterior (global parameters) for the two splitted components

        relating_n = torch.nonzero(torch.sum(new_q_of_z[:, [k, new_k]], dim=1) > 0.5).flatten()

        if len(relating_n) == 0:
            continue

        new_K = new_q_of_z.shape[1] 

        # responsibilities of the two splitted components and the additional (last) component, note in python the index starts from 0, thus the last component is the K-1 component
        sub_q_of_z = new_q_of_z[relating_n][:,[k, new_k, new_K - 1]]

        sub_data = new_data[k][:, relating_n]
        sub_sum_stat = compute_summary_stats(sub_data, sub_q_of_z, opts)
        #print(f"sub_sum_stat:{sub_sum_stat}")
        
        #  only update the posterior (global parameters) for the two splitted components
        sub_hp_posterior = mk_hp_posterior(hp_prior, sub_sum_stat, opts)
        # use coordinate ascent to update the local and global parameters, as well as the summary statistics, only for the two splitted components
        try:
            sub_f, sub_hp_posterior, _, sub_q_of_z, sub_sum_stat = update_posterior(sub_data, sub_hp_posterior, hp_prior, sub_sum_stat, opts, ite = 10, do_sort = 1)
            #print(f"sub_q_of_z_after update:{sub_q_of_z.shape}")
        except ValueError:
            continue

        if sub_q_of_z.shape[1] < 3:
            continue
        elif sub_q_of_z.shape[1] > 3:
            new_q_of_z = torch.cat([new_q_of_z, torch.zeros(new_q_of_z.shape[0], sub_q_of_z.shape[1]-3)], dim=1)
        if len(torch.nonzero(torch.sum(sub_q_of_z, dim=0) < 1.0e-1)) > 1: # torch.sum(sub_q_of_z, dim=0) < 1.0e-4
            continue
        
        # increase the dimension of omega and fc to include the new components, which are used to compute the free energy
        new_omega = omega.clone()
        
        sub_omega = mk_omega(new_data[k], sub_hp_posterior, hp_prior, opts)
        
        insert_indices = [k, new_k] + list(range(new_K-1, new_K + sub_q_of_z.shape[1] - 3)) # 两个list相加即把两个list的元素合并 insert_indices = [k, new_k, new_K-1]

        # Ensure new_omega has enough columns
        required_columns = max(insert_indices) + 1
        if new_omega.size(1) < required_columns:
            new_omega = torch.cat([new_omega, torch.zeros(new_omega.size(0), required_columns - new_omega.size(1))], dim=1)

        new_omega[:][:, insert_indices] = sub_omega
        
        new_fc = fc.clone()

        if new_fc.size(1) < required_columns:
            new_fc = torch.cat([new_fc, torch.zeros(new_fc.size(0), required_columns - new_fc.size(1))], dim=1)
        new_fc[0,insert_indices] = mk_E_log_q_p_eta(sub_data, sub_hp_posterior, hp_prior, sub_sum_stat, opts)
        # new_free_energy[0,k] is the free energy of splitting the kth component
        #print(f"new_fc:{new_fc}")
        #print(f"new_omega:{new_omega}")
        new_free_energy[0,k], _ = mk_free_energy(new_data[k], sub_hp_posterior, hp_prior, sub_sum_stat, opts, new_fc, new_omega)
        new_q_of_z[relating_n[:,None], :] = 0

        # print(f"relating_n:{relating_n.shape}")
        # print(f"range i:{len(relating_n.numpy())}")
        # print(f"insert_indices:{insert_indices}")
        # print(f"sub_q_of_z:{sub_q_of_z.shape}")
        # print(f"new_q_of_z:{new_q_of_z.shape}")

        for i in range(len(relating_n.numpy())):
            new_q_of_z[relating_n[i], insert_indices] = sub_q_of_z[i,:]

        new_q_of_z_cell[k] = new_q_of_z
        
    # split the component that miminizes the free energy
    free_energy, k = torch.min(new_free_energy, dim=1)
    
    if torch.isinf(free_energy):
        print('No splitting will decrease the free energy.')
        return 0, hp_posterior, data, sum_stat, -1
    
    data = new_data[k.item()]
    q_of_z = new_q_of_z_cell[k.item()]
    sum_stat = compute_summary_stats(data, q_of_z, opts)
    hp_posterior = mk_hp_posterior(hp_prior, sum_stat, opts)
    
    return free_energy.item(), hp_posterior, data, sum_stat, k.item()


def split(k, data, q_of_z, hp_posterior, hp_prior, opts):
    # q_of_z: N*K

    new_data = data
    
    # split component k into two components and assign the data points and corresponding responsibilities to the new components
    if opts['init_of_split'] == 'pc':  # principal eigenvector
        arg1_data = new_data
        D = arg1_data.shape[0]    
        expected_cov = hp_posterior['inv_W'][:,:,k] / (hp_posterior['lambda'][0][k]-D-1) # expected covariance matrix of the component k, the covariance matrix follow an inverse-Wishart distribution
        dir = divide_by_principal_component(arg1_data, expected_cov, hp_posterior['m'][:,k])
        # split component k into two components along the principal component
        q_of_z_k1 = torch.zeros(q_of_z.shape[0], 1) # N*1
        q_of_z_k2 = q_of_z[:,k].unsqueeze(1).clone()   # N*1
        
        I = (dir >= 0).nonzero(as_tuple=True)[0] # 1*Number of non-negative elements in dir
        
        q_of_z_k1[I] = q_of_z[I, k].unsqueeze(1).clone()       
        q_of_z_k2[I] = 0

    else:
        q_of_z_k = q_of_z[:, k]
        if opts['init_of_split'] == 'rnd':  # random
            r = torch.rand(q_of_z.shape[0], 1)
        elif opts['init_of_split'] == 'rnd_close':  # make close clusters
            r = 0.5 + (torch.rand(q_of_z.shape[0], 1) - 0.5) * 0.01
        elif opts['init_of_split'] == 'close_f':  # one is almost zero
            r = 0.98 + torch.rand(q_of_z.shape[0], 1) * 0.01
        else:
            raise ValueError('Unknown algorithm')

        q_of_z_k1 = q_of_z_k * r
        q_of_z_k2 = q_of_z_k * (1 - r)

    new_q_of_z = torch.zeros(q_of_z.size(0), q_of_z.size(1) + 1)  # N*(K+1) add a new component in the responsibilities
    
    # Create a true copy of q_of_z
    q_z_copy = q_of_z.clone()
    
    # Fill new_q_of_z with values from q_z_copy
    # change the responsibilities of the kth component to the new responsibilities q_of_z_k1
    # add a new component with the responsibilities q_of_z_k2 to the (end-1) of the responsibilities (remember the end of q_of_z is always the additional component)
    new_q_of_z[:, :-2] = q_z_copy[:, :-1]
    new_q_of_z[:, -1] = q_z_copy[:, -1]
    
    new_q_of_z[:, k] = q_of_z_k1.squeeze()

    new_k = new_q_of_z.size(1) - 2 # the new component is the second last component, note in python the index starts from 0
    
    new_q_of_z[:, new_k] = q_of_z_k2.squeeze()
    
    info = {'new_k': new_k}

    return new_data, new_q_of_z, info

def divide_by_principal_component(data, covariance, mean):
    """
    data: Tensor of shape (features, samples)
    covariance: Covariance matrix of shape (features, features)
    mean: Mean vector of shape (features,)
    """
    D, N = data.shape

    if D <= 16:
        eigvals, eigvecs = torch.linalg.eig(covariance)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        
        principal_component_idx = torch.argmax(eigvals)

        principal_component = eigvecs[:, principal_component_idx]

    else:
        principal_component, _ = power_method(covariance)

    centered_data = data - mean.unsqueeze(1)
    direction = torch.sum(centered_data * principal_component.unsqueeze(1), dim=0)
    # direction结果与matlab数值相同，正负号相反，可能是特征值计算方法的问题？
    direction = -direction
    
    return direction

def cumulant_func(betak, mk, Wk, lambdak): # parameters of the normal-Wishart distribution for the kth component
    ''' Evaluate cumulant function at given params (normal-Wishart). log marginal likelihood = -1*cumulant

    Returns
    --------
    cumulant function : scalar real value of cumulant function at provided args
    '''
    
    log_det_Wk = detln(Wk) # log|inv_W| = log|W^-1| = -log|W|
    D = mk.shape[0]
    gamma_term = 0
    for i in np.arange(1, D + 1):
        term1 = torch.special.gammaln(0.5 * (lambdak + 1 - i))
        gamma_term += term1

    cumulant = - 0.5 * D * np.log(2 * np.pi) - 0.25 * D * (D - 1) * np.log(np.pi) - 0.5 * D * np.log(2) * lambdak - gamma_term + 0.5 * D * np.log(betak) + 0.5 * lambdak * log_det_Wk #(log_det_Wk is minus log_det_B)
    if not torch.is_tensor(cumulant):
        cumulant = torch.tensor(cumulant)
    if cumulant.shape == torch.Size([]):
        cumulant = cumulant.unsqueeze(0)
    return cumulant

def calcHardMergeGap(data, q_of_z, hp_posterior, hp_prior, opts, kA, kB):
    ''' Calculate ratio of marginal likelihoods before and after merging two components. also, add an additional component at the end

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
    '''
    cA = cumulant_func(hp_posterior['beta'][:,kA], hp_posterior['m'][:,kA], hp_posterior['W'][:,:,kA], hp_posterior['lambda'][:,kA])
    cB = cumulant_func(hp_posterior['beta'][:,kB], hp_posterior['m'][:,kB], hp_posterior['W'][:,:,kB], hp_posterior['lambda'][:,kB])
    cPrior = cumulant_func(hp_prior['beta0'], hp_prior['m0'], hp_prior['W0'], hp_prior['lambda0'])

    q_of_z_AB = torch.cat([q_of_z[:,kA].unsqueeze(1)+ q_of_z[:,kB].unsqueeze(1), q_of_z[:,-1].unsqueeze(1)], dim=1) # N*3
    
    sum_stat_AB = compute_summary_stats(data, q_of_z_AB, opts)
    posterior_AB = mk_hp_posterior(hp_prior, sum_stat_AB, opts)
    cAB = cumulant_func(posterior_AB['beta'][:,0], posterior_AB['m'][:,0], posterior_AB['W'][:,:,0], posterior_AB['lambda'][:,0])
    #print(f"cA:{cA}, cB:{cB}, cPrior:{cPrior}, cAB:{cAB}")
    return cA + cB - cPrior - cAB # as log marginal likelihood = -1*cumulant, so here we use cA + cB - cPrior - cAB where c() is the cumulant function

def mk_merge_pairs(data, q_of_z, hp_posterior, hp_prior, sum_stat, opts):
    ''' Randomly select kA, and find the best component kB to merge with kA based on the ratio of marginal likelihoods before and after merging

    Returns
    ---------
    best_pair : tuple of 3 integers, indices of components to merge, the gap of the best pair
    '''
    k_max = opts['max_merge']
    true_K = sum_stat['Nk'].shape[1] - 1 # ignore the last component (the additional component)
    best_pairs = {}
    for kA in np.arange(max(true_K-k_max,0), true_K):
        # for each kA, find the best kB to merge with kA
        merge_pairs = []
        gap_list =[]
        for kB in np.arange(max(true_K-k_max,0), true_K):
            if kB == kA:
                continue
            gap = calcHardMergeGap(data, q_of_z, hp_posterior, hp_prior, opts, kA, kB)
            gap_list.append(gap.item())
            merge_pairs.append((kA, kB, gap))

        _ , idx = torch.max(torch.tensor(gap_list), dim=0, keepdim=False)

        best_pairs[kA] = merge_pairs[idx]

    return best_pairs # return the best pair to merge for each component

def find_best_merge(data, q_of_z, hp_posterior, hp_prior, sum_stat, opts):
    ''' Find the best pair of components to merge that minimize the free energy after merging

    Returns
    ---------
    indices of components to merge, the free energy after merging the two components
    '''
    # best pairs is a dictionary including the best kB to merge with kA for each kA in range(0, true_K)
    # best pairs = {kA:tuple (kA, kB, gap)} for kA in range(true_K)
    best_pairs = mk_merge_pairs(data, q_of_z, hp_posterior, hp_prior, sum_stat, opts) 
    free_energy_merge_set = []
    merge_component_set = []
    # find the best pair of components to merge
    # use a greedy approach to find the pair of components to merge, which minimize the free energy
    for comp_A in best_pairs.keys(): # randomly select kA and find the best kB to merge with kA
        best_pair = best_pairs[comp_A]
        # best_pair is a tuple (kA, kB, gap)
        kA = best_pair[0]
        kB = best_pair[1]
        
        # create the new responsibilities after merging the two components, q_of_z_AB = q_of_z[:,kA] + q_of_z[:,kB]
        q_of_z_mergeAB = q_of_z[:,kA].unsqueeze(1)+ q_of_z[:,kB].unsqueeze(1) # N*1
        new_q_of_z = q_of_z.clone()
        new_q_of_z[:, kA] = q_of_z_mergeAB.squeeze() # replace the responsibilities of component kA with the merged responsibilities
        new_q_of_z = torch.cat([new_q_of_z[:,:kB], new_q_of_z[:,kB+1:]], dim=1) # delete the kB component

        new_sum_stat = compute_summary_stats(data, new_q_of_z, opts)
        new_hp_posterior = mk_hp_posterior(hp_prior, new_sum_stat, opts)
        free_energy_merge, hp_posterior_merge, data, q_of_z_merge, sum_stat_merge = update_posterior(data, new_hp_posterior, hp_prior, new_sum_stat, opts, ite = opts['ite'], do_sort = 1)
        free_energy_merge_set.append(free_energy_merge.item())
        merge_component_set.append((kA, kB))

    free_energy_merge_best, idx = torch.min(torch.tensor(free_energy_merge_set), dim=0) # find the best pair of components to merge, which minimize the free energy
    kA, kB = merge_component_set[idx]

    # return the best pairs to merge and the free energy after merging the two components
    # also, return the updated parameters after merging the two components
    return kA, kB, free_energy_merge_best, hp_posterior_merge, q_of_z_merge, sum_stat_merge 

def do_merge(data, q_of_z, hp_posterior, hp_prior, sum_stat, free_energy, opts):
    # merge component ka and kb into a new component, accept the merge if the free energy decreases (ELBO increases), also use a greedy manner
    # iteratively merge until the free energy does not decrease or the maximum number of iterations is reached
    my_disp('### merge step ###')
    i = 0
    while True:
        
        if sum_stat['Nk'].shape[1] < 3:
            my_disp('less than 2 components')
            return free_energy, hp_posterior, data, q_of_z, sum_stat
        # find the best pair of components to merge that minimize the free energy after merging
        kA, kB, free_energy_merge_best, hp_posterior_merge, q_of_z_merge, sum_stat_merge = find_best_merge(data, q_of_z, hp_posterior, hp_prior, sum_stat, opts)
        #my_disp(f'merging components {kA} and {kB} minimize free energy, the free energy after merging is {free_energy_merge_best.item():.5g}')
        if not free_energy_decreased(free_energy, free_energy_merge_best, 0, opts) or i >= opts['max_iter_merge']: # if the free energy does not decrease after merging the two components, reject the merge and stopt the merging process
            my_disp('free_energy not decreased after merging')
            disp_status(free_energy, sum_stat, opts)
            return free_energy, hp_posterior, data, q_of_z, sum_stat
        
        # if the free energy decreases after merging the two components, accept the merge and reset the parameters for the next merge iteration
        free_energy = free_energy_merge_best
        hp_posterior = hp_posterior_merge
        q_of_z = q_of_z_merge
        sum_stat = sum_stat_merge

        i += 1 # times of iteration
    
    return free_energy, hp_posterior, data, q_of_z, sum_stat

# use coordinate ascent to update the responsibilities (local parameters) and the Beta and Normal-Wishart parameters (global parameters), as well as the summary statistics
# include 4 steps
# 1. compute the free energy (mk_free_energy)
# 2. update the responsibilities (mk_q_of_z) (with sort_q_of_z)
# 3. update the summary statistics (compute_summary_stats)
# 4. update the global parameters (mk_hp_posterior)

def update_posterior(data, hp_posterior, hp_prior, sum_stat, opts, ite=10, do_sort = 1):
    #my_disp('### updating posterior ...')
    free_energy = float('inf')
    
    i = 0
    last_Nk = 0 # Nk of the last iteration
    start_sort = 0
    
    while True:
        i += 1
        #print(f"sum_stat in update_posterior:{sum_stat}")
        new_free_energy, omega = mk_free_energy(data, hp_posterior, hp_prior, sum_stat, opts)
        #print(f"new_free_energy:{new_free_energy}")
        #disp_status(new_free_energy, hp_posterior, opts)
        
        if (not torch.isinf(torch.tensor(ite)).item() and i >= ite) or (torch.isinf(torch.tensor(ite)).item() and not free_energy_decreased(free_energy, new_free_energy, 0, opts)):
            free_energy = new_free_energy
            if do_sort and opts['do_sort'] and not start_sort:
                start_sort = 1
            else:
                break
        
        last_Nk = sum_stat['Nk']
        free_energy = new_free_energy
        q_of_z, data, _ = mk_q_of_z(data, hp_posterior, hp_prior, opts, omega)
        
        
        if opts['algorithm'] == 'vdp' and torch.sum(q_of_z[:, -1]) > 1.0e-20:
            q_of_z = torch.cat((q_of_z, torch.zeros(q_of_z.shape[0], 1)), dim=1) # 增加一个component代表infinite inactive components
        
        if start_sort:
            q_of_z, _ = sort_q_of_z(data, q_of_z, opts)
        
        if opts['algorithm'] == 'vdp': 
            I = torch.sum(q_of_z, dim=0) > 1.0e-5
            indices = torch.nonzero(I).flatten()
            indices = torch.cat([indices, torch.tensor([q_of_z.shape[1]-1])], dim=0)
            q_of_z = q_of_z[:, indices]
            q_of_z = normalize(q_of_z, 1)

        #print(f"sub_q_of_z_shape:{q_of_z.shape}")
        sum_stat = compute_summary_stats(data, q_of_z, opts)
        #print(f"sub_Nk:{sum_stat['Nk']}")
        hp_posterior = mk_hp_posterior(hp_prior, sum_stat, opts)
    
    # my_disp('### updating posterior ... done.')
    # disp_status(free_energy, sum_stat, opts)
    return free_energy, hp_posterior, data, q_of_z, sum_stat 

def sort_q_of_z(data, q_of_z, opts):
    
    #my_disp('sorting...')
    
    Nk = torch.sum(q_of_z, dim=0)  # 1*K
    
    if opts['algorithm'] == 'vdp':
        sorted_indices = torch.argsort(Nk[:-1], descending=True)
        sorted_indices = torch.cat((sorted_indices, torch.tensor([Nk.size(0) - 1])))
    else:
        sorted_indices = torch.argsort(Nk, descending=True)
    
    q_of_z = q_of_z[:, sorted_indices]
    
    #my_disp('sorting... done.')
    
    return q_of_z, sorted_indices

def free_energy_decreased(free_energy, new_free_energy, warn_when_increasing, opts): # minimize variation free energy, equivalent to maximize ELBO
    '''return True is new_free_energy < free_energy, otherwise return False'''
    diff = new_free_energy - free_energy
    
    if torch.abs(diff / free_energy) < opts['threshold']:
        return False
    
    elif diff > 0:
        if warn_when_increasing:
            if torch.abs(diff / free_energy) > 1.0e-3:
                raise ValueError(f"the free energy increased. the diff is {new_free_energy - free_energy}")
            else:
                print(f"Warning: the free energy increased. the diff is {new_free_energy - free_energy}")
        return False
    elif diff == 0:
        return False
    else:
        return True
    
def cluster_assignments(data, hp_posterior, hp_prior, opts, omega=None):
        # z: batch_size * latent_dim --> z.t(): latent_dim * batch_size
        q_of_z, _, _ = mk_q_of_z(data, hp_posterior, hp_prior, opts, omega=None)
        # Here, responsibility is a 2D array of size N x K. here N is batch size, K active clusters
        # Each entry resp[n, k] gives the probability that data atom n is assigned to cluster k under the posterior.
        resp = q_of_z[:,:-1]

        # To convert to hard assignments
        # Here, Z is a 1D array of size N, where entry Z[n] is an integer in the set {0, 1, 2, … K-1, K}.
        # Z represents for each atom n (in total N), which cluster it should belongs to accroding to the probability
        Z = resp.argmax(axis=1)
        return resp, Z

# mk_log_likelihood is not required for vdpgmm p(x|X, theta) = sum_{k=1 to T}(E_q(vk)*E_q(p(x|theta_k)) + (1-sum_{k=1 to T}(E_p(vk)))*E_p(p(x|theta0)) where T is the number of active components
def mk_log_likelihood(data, hp_posterior, hp_prior, opts):
    """
    Compute the log likelihood.
    
    Args:
        data (tensor): D*N the new data, whose likelihood is conditioned on the posterior given previous data.
        hp_posterior (dict): Hyperparameters for the posterior.
        hp_prior (dict): Hyperparameters for the prior.
        opts (dict): Additional options.

    Returns:
        torch.Tensor: Log likelihood.
    """
    D, N = data.shape
    K = hp_posterior['m'].shape[1] 
    log_likelihood = torch.zeros(K, N)

    E_pi = mk_E_pi(hp_posterior, hp_prior)

    # activated components
    for k in range(K):
        mu = hp_posterior['m'][:, k].unsqueeze(1)
        f = hp_posterior['lambda'][:,k] + 1 - D
        Sigma = (hp_posterior['beta'][:,k] + 1) / hp_posterior['beta'][:,k] / f * hp_posterior['inv_W'][:,:,k]
        log_likelihood[k, :] = torch.log(E_pi[k]) + logmvtpdf(data, mu, f, Sigma)
    
    log_likelihood = log_sum_exp(log_likelihood, 0)  # 1 by N
    log_likelihood = torch.sum(log_likelihood, dim=1)
    
    return log_likelihood

# E_q(pi(vk)), pi(vk) = (1 - sum_{i=1 to k-1}(vi))*vk, vi~Beta(gamma(i,1), gamma(i,2)), E_q(vi) = gamma(i,1)/(gamma(i,1) + gamma(i,2))
def mk_E_pi(hp_posterior, hp_prior):
    """
    Compute the expected value of the mixing proportions obtained from the stick-breaking process.
    
    Args:
        hp_posterior (dict): Posterior hyperparameters.
        hp_prior (dict): Prior hyperparameters.
        opts (dict): Options for the algorithm.
        
    Returns:
        torch.Tensor (1*K): Expected value of the mixing proportions.
    """
    K = hp_posterior['m'].shape[1]
    E_pik = torch.zeros(K)
    # activated components
    Ka = K - 1 # activated components
    print(f"Ka:{Ka}")
    E_vk = torch.zeros(Ka)
    
    for k in range(Ka):
        prod_term = 1
        E_vk[k] = hp_posterior['gamma'][0, k] / torch.sum(hp_posterior['gamma'][:, k], dim=0)
        if k > 0:
            for i in range(k):
                prod_term *= (1 - E_vk[i])
            E_pik[k] = prod_term * E_vk[k]
        else:
            E_pik[k] = E_vk[k]
    
    # inactivated components (1-sum{k=1,Ka}E_p[pik(vk)]) p(vk)~Beta(1,alpha) E_p(vk) = 1/(1+alpha)
    E_v_prior = 1/(1+hp_prior['alpha'].item())
    E_p_vk = torch.zeros(Ka)
    for k in range(Ka):
        prod_term = 1
        if k > 0:
            for i in range(k):
                prod_term *= (1 - E_v_prior)
            E_p_vk[k] = prod_term * E_v_prior
        else:
            E_p_vk[k] = E_v_prior
    
    E_pik[-1] = 1 - torch.sum(E_p_vk, dim=0)
    return E_pik # 1*K with an additional component at the end

def logmvtpdf(data, mu, f, Sigma):
# log pdf of multivariate t-student dist. (the marginal probability of x given the component is a multivariate t-distribution)
# data : D by N
    D, N = data.shape
    c = torch.special.gammaln((D+f)*0.5) - (D*0.5)*torch.log(f*np.pi) - torch.special.gammaln(f*0.5) - 0.5*detln(Sigma)
    diff = data - mu.repeat(1,N) # D*N
    logpdf = c - (f+D)*0.5 * torch.log(1 + torch.sum(diff*np.linalg.solve((f*Sigma), diff), dim = 0)) # 1*N

    return logpdf

def mkopts_vdp(**kwargs):
    opts = {
        'algorithm': 'vdp',
        'init_of_split': 'pc',
        'initial_K': 1,
        'do_sort': 1,
        'do_greedy_split': 1,
        'do_split': 0,
        'do_merge': 1,
        'get_q_of_z': 0,
        'get_E_pi': 0,
        'get_log_likelihood': 0,
        'max_iter_merge': 3,
        'max_iter_split': 3,
        'max_merge': 5,
        'max_split': 5,
        'ite': 10,
        'threshold': 1e-5,
        'min_split_size': 2
    }
    
    # Override default options with provided keyword arguments
    opts.update(kwargs)
    
    return opts

def online_vdpgmm_suffstat(data, hp_prior = None, opts = {}, previous_results={}):

    if len(opts) == 0:
        opts = mkopts_vdp()

    if hp_prior == None:
        hp_prior = mk_hp_prior(data, opts)

    if 'hp_posterior' in opts:
        if opts['get_q_of_z']:
            results['q_of_z'] = mk_q_of_z(data, opts['hp_posterior'], hp_prior, opts)
        if opts['get_log_likelihood']:
            results['log_likelihood'] = mk_log_likelihood(data, opts['hp_posterior'], hp_prior, opts)
        return results

    if 'q_of_z' in previous_results:
        q_of_z = previous_results['q_of_z']
    else:
        q_of_z = rand_q_of_z(data, opts['initial_K'], opts)
    
    if 'sum_stat' in previous_results:
        sum_stat = previous_results['sum_stat']
    else:
        sum_stat = compute_summary_stats(data, q_of_z, opts)
    
    hp_posterior = mk_hp_posterior(hp_prior, sum_stat, opts)

    if opts['do_greedy_split']:
        free_energy, hp_posterior, data, q_of_z, sum_stat = greedy_split(data, hp_posterior, hp_prior, sum_stat, q_of_z, opts)
    else:
        raise ValueError('Unknown algorithm')
    
    if opts['do_merge'] and sum_stat['Nk'].shape[1] > 2: # if there is more than one component, we can do merge step
        free_energy, hp_posterior, data, q_of_z, sum_stat = do_merge(data, q_of_z, hp_posterior, hp_prior, sum_stat, free_energy, opts)
    else:
        print('merge step is not implemented')
    
    # match the hard assignment and responsibilities with the order of input data (as the dataset is shuffled)
    q_of_z, comps = cluster_assignments(data, hp_posterior, hp_prior, opts)
    
    #disp_status(free_energy, sum_stat, opts)
    results = {
        'algorithm': opts['algorithm'],
        'free_energy': free_energy,
        'hp_prior': hp_prior,
        'sum_stat': sum_stat,
        'q_of_z': q_of_z,
        'hp_posterior': hp_posterior,
        'K': sum_stat['Nk'].shape[1],
        'true_K': sum_stat['true_Nk'].shape[1]-1,
        'opts': opts,
        'hard_assign': comps
    }

    # if opts['get_q_of_z']:
    #     results['q_of_z'] = mk_q_of_z(data, hp_posterior, hp_prior, opts)

    if opts['get_log_likelihood']:
        results['log_likelihood'] = mk_log_likelihood(data, hp_posterior, hp_prior, opts)

    if opts['get_E_pi']:
        results['E_pi'] = mk_E_pi(hp_posterior, hp_prior)

    return results
