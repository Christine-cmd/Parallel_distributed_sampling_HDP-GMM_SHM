# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:04:08 2024

@author: DELL
"""

import numpy as np
import time
import torch
import torch.nn.functional as F
import pandas as pd

## Initialize the prior

def mk_hp_prior(data, opts):
    hp_prior = {}

    if 'xi0' in opts:
        hp_prior['xi0'] = opts['xi0']
    else:
        hp_prior['xi0'] = 0.01

    if 'eta_p' in opts:
        eta_p = opts['eta_p']
    else:
        eta_p = 1

    if 'alpha' in opts:
        hp_prior['alpha'] = opts['alpha']
    else:
        hp_prior['alpha'] = 1

    D, N = data['given_data'].shape

    if opts.get('use_kd_tree', False):
        sum_xx = torch.sum(torch.stack([tree['sum_xx'] for tree in data['kdtree']], dim=2), dim=2)
        sum_x = torch.sum(torch.stack([tree['sum_x'] for tree in data['kdtree']], dim=1), dim=1)
        covariance = sum_xx / N - torch.ger(sum_x, sum_x) / (N * N)
    else:
        covariance = torch.cov(data['given_data'])
        #print(covariance)
        #cov1 = np.cov(data['given_data'])
        #print(cov1)

    hp_prior['m0'] = torch.mean(data['given_data'], dim=1)
    hp_prior['m0'] = hp_prior['m0'].unsqueeze(1) # D*1

    if D > 16:
        [dummy, max_eig] = power_method(covariance);
        # eigvals, _ = torch.linalg.eig(covariance)
        # max_eig = torch.max(eigvals[:, 0])
    else:

        max_eig = torch.max(torch.linalg.eig(covariance)[0].real)

    hp_prior['eta0'] = eta_p * D
    hp_prior['B0'] = covariance
    
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
    # q_of_z: N*K
    
    if opts['use_kd_tree']:
        N = len(data['kdtree'])
    else:
        N = data['given_data'].shape[1]

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

def log_p_of_x_given_c(data, clusters, hp_posterior):
    D, N = data.shape
    K = len(clusters)
    log_p_of_x_given_c = np.zeros((K, N))
    
    for i in range(K):
        c = clusters[i]
        m = hp_posterior['m'][:, c]
        precision = hp_posterior['inv_B'][:, :, c] @ hp_posterior['eta'][c]
        d = data - np.tile(m, (1, N))
        log_p_of_x_given_c[i, :] = (-D * 0.5) * np.log(2 * np.pi) + 0.5 * detln(precision) - 0.5 * torch.sum(d * (precision @ d), axis=0)
    
    return log_p_of_x_given_c

## Update responsibilities q_of_z
def mk_q_of_z(data, hp_posterior, hp_prior, opts, log_lambda=None):
    # Compute log_lambda if not provided
    # log_lambda(n,i) is S(n,i) in the paper
    if log_lambda is None:
        log_lambda = mk_log_lambda(data, hp_posterior, hp_prior, opts)
    
    # Compute q_of_z (responsibility)
    q_of_z = torch.exp(normalizeln(log_lambda, 1))
    
    return q_of_z, data, log_lambda

def mk_log_lambda(data, hp_posterior, hp_prior, opts):
    
    if opts['algorithm'] == 'vdp':
        if abs(hp_posterior['gamma'][1, -1] - hp_prior['alpha']) > 1.0e-3:
            print(f"hp_posterior.gamma(2, end): {hp_posterior['gamma'][1, -1]}")
            print(f"hp_prior.alpha: {hp_prior['alpha']}")
            diff = hp_prior['alpha'] - hp_posterior['gamma'][1, -1]
            raise ValueError("must be alpha")
        
    D, N = data['given_data'].shape
    K = hp_posterior['eta'].shape[1]

    c0 = hp_posterior['eta'] + 1
    c0 = c0.repeat(D,1)
    d0 = torch.tensor([_+1 for _ in range(D)])
    d0 = d0.reshape(D,1)
    d0 = d0.repeat(1,K)
    psi_sum = torch.sum(torch.special.psi(c0-d0 * 0.5), dim=0)
    
    log_lambda = torch.zeros(N, K)

    for c in range(K):
        if opts['algorithm'] == 'vdp':
            E_log_p_of_z_given_other_z_c = (
                torch.special.psi(hp_posterior['gamma'][0, c])
                - torch.special.psi(torch.sum(hp_posterior['gamma'][:, c], dim=0))
                + torch.sum(torch.special.psi(hp_posterior['gamma'][1, :c]) - torch.special.psi(torch.sum(hp_posterior['gamma'][:, :c], dim=0)))
            )
        else:
            raise ValueError('unknown algorithm')


        Precision = 0.5 * hp_posterior['inv_B'][:, :, c] * hp_posterior['eta'][0][c]  # D*D
        
        E_log_p_of_x = -0.5 * D * torch.log(torch.tensor(np.pi)) - 0.5 * detln(hp_posterior['B'][:,:,c]) + 0.5 * psi_sum[c] - 0.5 * D / hp_posterior['xi'][0,c]
        # 1*1
        
        d = data['given_data'] - hp_posterior['m'][:, c].unsqueeze(1).repeat(1,N) # D*N
        
        E_log_p_of_x = -torch.sum(d * (Precision @ d), dim=0) + E_log_p_of_x 
        E_log_p_of_x = E_log_p_of_x.unsqueeze(0) # 1*N
        
        
        log_lambda[:, c] = E_log_p_of_x + E_log_p_of_z_given_other_z_c

    if opts['algorithm'] == 'vdp':
        log_lambda[:, -1] = log_lambda[:, -1] - torch.log(1 - torch.exp(torch.special.psi(torch.tensor(hp_prior['alpha'])) - torch.special.psi(torch.tensor(1 + hp_prior['alpha'])))) 
        
    return log_lambda

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

## Update the variational parameters
# mk_log_likelihood is not required for vdpgmm
def mk_log_likelihood(data, hp_posterior, hp_prior, opts):
    """
    Compute the log likelihood.
    
    Args:
        hp_posterior (dict): Hyperparameters for the posterior.
        hp_prior (dict): Hyperparameters for the prior.
        opts (dict): Additional options.

    Returns:
        torch.Tensor: Log likelihood.
    """
    given_data = data['given_data']
    D, N = given_data.shape
    K = hp_posterior['m'].shape[1]
    log_likelihood = torch.zeros(K, N)

    E_pi = mk_E_pi(hp_posterior, hp_prior, opts)

    for c in range(K):
        mu = hp_posterior['m'][:, c]
        f = hp_posterior['eta'][c] + 1 - D
        Sigma = (hp_posterior['xi'][c] + 1) / hp_posterior['xi'][c] / f * hp_posterior['B'][c]
        log_likelihood[c, :] = log_no_w(E_pi[c]) + logmvtpdf(given_data, mu, f, Sigma)

    log_likelihood = log_sum_exp(log_likelihood, 0)  # 1 by N
    log_likelihood = torch.sum(log_likelihood, dim=1)
    
    return log_likelihood

def mk_hp_posterior(data, q_of_z, hp_prior, opts):
#     the last component of q_of_z represents the infinite rest of components
#     the last component is the prior.
#     q_of_z: N*K
#     q_of_z(:,end) is the rest of responsibilities.

    threshold_for_N = 1.0e-200
    
    K = q_of_z.size(1)
    
    D, N = data['given_data'].size()
    
    if opts['algorithm'] == 'vdp':
        
        true_Nc = q_of_z.sum(dim=0)  # 1*K  Nc is expected number of observations in cluster c
        true_Nc = true_Nc.unsqueeze(0)
        
        q_of_z[:, -1] = 0 # add a new component without any data to represent the infinite sum of inactive components
    
    Nc = q_of_z.sum(dim=0)
    

    Nc = Nc.unsqueeze(0)  # 1*K true_Nc != Nc for VDP
    
    sum_x = torch.mm(data['given_data'] , q_of_z)  # r_nk*xn in PRML (dot mul) D*N @ N*K = D*K
    
    I = torch.where(torch.squeeze(Nc) > threshold_for_N)[0]
    
    inv_Nc = torch.zeros(1, K)
    
    inv_Nc[:, I] = 1.0 / Nc[:, I] # 1*K
    
    hp_posterior = {}
    hp_posterior['eta'] = hp_prior['eta0'] + Nc  # eta0 is nu_0 in PRML
    hp_posterior['xi'] = hp_prior['xi0'] + Nc  # xi0 is beta0 in PRML
    means = sum_x * inv_Nc.repeat(D,1)  # D*K .* D*K   x_bar in PRML
    
    hp_posterior['inv_B'] = torch.zeros(D, D, K)
    hp_posterior['B'] = torch.zeros(D, D, K)
    

    for c in range(K):
             
        v = data['given_data'] - means[:, c].unsqueeze(1).repeat(1, N)  # D*N  v is (x_n-x_bar) in PRML
        v0 = means[:, c].unsqueeze(1) - hp_prior['m0']  # v0 is (x_bar - m0) in PRML
        # hp_prior['m0'] should have the size D*1 instead of D
        
        hp_posterior['B'][:,:,c] = hp_prior['B0'] + \
                            torch.mm((q_of_z[:, c].unsqueeze(1).repeat(1,D).t() * v), v.t()) + \
                            Nc[0][c] * hp_prior['xi0'] * torch.mm(v0, v0.t()) / hp_posterior['xi'][0][c] 
             
        #  torch.mm((q_of_z[:, c].unsqueeze(1).repeat(1,D).t() * v), v.t()) ---- (D*N .* D*N) @ N*D = D*D
        #  size(Nc) = 1*K Nc[c] = Nc[0,c]
        #  hp_posterior['B'][:,:,c] --- D*D
        
    
    for c in range(K):
        
        hp_posterior['inv_B'][:, :, c] = torch.inverse(hp_posterior['B'][:,:,c].squeeze())
    
    hp_posterior['m'] = (sum_x + hp_prior['xi0'] * hp_prior['m0'].repeat(1, K)) / (Nc + hp_prior['xi0']).repeat(D, 1) # D*K
    
    if opts['algorithm'] == 'vdp':
        hp_posterior['gamma'] = torch.zeros(2, K)
        hp_posterior['gamma'][0, :] = 1 + true_Nc
        hp_posterior['gamma'][1, :] = hp_prior['alpha'] + true_Nc.sum() - torch.cumsum(true_Nc, dim=1)
    
    hp_posterior['Nc'] = Nc
    if opts['algorithm'] == 'vdp':
        hp_posterior['true_Nc'] = true_Nc
    hp_posterior['q_of_z'] = q_of_z  # q_of_z is a N by K matrix where N is  # of given_data
    
    return hp_posterior

## Calculate the variational free energy (minus ELBO)
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
        
def disp_status(free_energy, hp_posterior, opts):
    if opts['algorithm'] == 'vdp':
        Nc = hp_posterior['true_Nc']
    else:
        Nc = hp_posterior['Nc']
    
    my_disp(f"F={free_energy:.5g};    Nc=[{', '.join(map(str, Nc.numpy()))}];")

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

def mk_E_log_q_p_eta(data, hp_posterior, hp_prior, opts):
    # returns E_q(eta)[log q(eta)/p(eta)] % Eq.(10.74) and Eq.(10.77) in PRML
    # fc : 1 by K
    D = hp_posterior['m'].size(0)
    K = hp_posterior['eta'].size(1)
    log_det_B = torch.zeros(1, K) # log determinant
    term_eta = torch.zeros(2, K)
    
    for c in range(K):
        log_det_B[0, c] = detln(hp_posterior['B'][:,:,c])
        d = hp_posterior['m'][:, c].unsqueeze(1) - hp_prior['m0']  # D*1 d = (m_k - m0) in PRML 
        term_eta[0, c] = torch.sum(hp_posterior['inv_B'][:, :, c] * (hp_prior['xi0'] * torch.mm(d, d.t())))
        term_eta[1, c] = torch.sum(hp_posterior['inv_B'][:, :, c] * hp_prior['B0']) - D
    
    #print(term_eta)
    E_log_q_p_mean = (
        0.5 * D * (hp_prior['xi0'] / hp_posterior['xi'] 
                   - torch.log(hp_prior['xi0'] / hp_posterior['xi']) 
                   - 1)
        + 0.5 * hp_posterior['eta'] * term_eta[0, :]
    )
    
   
    
    psi_sum = torch.sum(
        torch.special.psi(((hp_posterior['eta'] + 1).repeat(D, 1) - torch.arange(1, D + 1).unsqueeze(1).repeat(1, K))* 0.5) ,
        dim=0
    ) # 1*K
    
    E_log_q_p_cov = (
        0.5 * hp_prior['eta0'] * (log_det_B - detln(hp_prior['B0'])) # 1*K
        + 0.5 * hp_posterior['Nc'] * psi_sum
        + 0.5 * hp_posterior['eta'] * term_eta[1, :].unsqueeze(0)
        + gamma_multivariate_ln(torch.tensor(hp_prior['eta0']).unsqueeze(0) * 0.5, D)
        - gamma_multivariate_ln(hp_posterior['eta'] * 0.5, D)
    )
    
    # print(f"E_log_q_p_mean:{E_log_q_p_mean}")
    # print(f"E_log_q_p_cov:{E_log_q_p_cov}")
    # print(f"psi_sum:{psi_sum}")
    
    if torch.any(E_log_q_p_mean < -1.0e-5):
        print(E_log_q_p_mean)
        raise ValueError('E_log_q_p_mean is negative.')
    
    if torch.any(E_log_q_p_cov < -1.0e-5):
        print(E_log_q_p_cov)
        raise ValueError('E_log_q_p_cov is negative.')
    
    fc = E_log_q_p_mean + E_log_q_p_cov
    
    return fc

def mk_free_energy(data, hp_posterior, hp_prior, opts, fc=None, log_lambda=None):
    
    if fc is None and log_lambda is None:
        fc = mk_E_log_q_p_eta(data, hp_posterior, hp_prior, opts)
        log_lambda = mk_log_lambda(data, hp_posterior, hp_prior, opts)

    N, K = log_lambda.shape
    
    if opts['algorithm'] == 'vdp':
        len_gamma = hp_posterior['gamma'].shape[1]
        
        E_log_p_of_V = (torch.special.gammaln(hp_posterior['gamma'].sum(dim=0)) 
                        - torch.special.gammaln(torch.tensor(1 + hp_prior['alpha'])) 
                        - torch.special.gammaln(hp_posterior['gamma']).sum(dim=0) 
                        + torch.special.gammaln(torch.tensor(hp_prior['alpha'])) 
                        + ((hp_posterior['gamma'][0, :] - 1) 
                           * (torch.special.psi(hp_posterior['gamma'][0, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0)))) 
                        + ((hp_posterior['gamma'][1, :] - hp_prior['alpha']) 
                           * (torch.special.psi(hp_posterior['gamma'][1, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0)))))
        extra_term = E_log_p_of_V.sum()
        
    else:
        raise ValueError('Unknown algorithm')

    free_energy = extra_term + fc.sum() - log_sum_exp(log_lambda, 1).sum()
    
    return free_energy, log_lambda

## Update component number, K, using a greedy approach

def greedy(data, hp_posterior, hp_prior, opts):
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
    free_energy, log_lambda0 = mk_free_energy(data, hp_posterior, hp_prior, opts)
    
    #disp_status(free_energy, hp_posterior, opts)
    
    while True:
        #my_disp('finding the best one....\t')
        
        new_free_energy, new_hp_posterior, new_data, c = find_best_splitting(data, hp_posterior, hp_prior, opts)
        
        if c == -1:
            return free_energy, hp_posterior, data
            #break
        
        #my_disp(f'finding the best one.... done')  
        #my_disp(f"component {c} was split.")
        
        #disp_status(new_free_energy, new_hp_posterior, opts)
        
        new_free_energy, new_hp_posterior, new_data, _ = update_posterior2(new_data, new_hp_posterior, hp_prior, opts, 1, opts['ite'])
        
        if not free_energy_decreased(free_energy, new_free_energy, 0, opts):
            return free_energy, hp_posterior, data
            #break
        free_energy = new_free_energy
        hp_posterior = new_hp_posterior
        data = new_data
    
    disp_status(free_energy, hp_posterior, opts)
    
    return free_energy, hp_posterior, data


def find_best_splitting(data, hp_posterior, hp_prior, opts):
    
    c_max = 10 # find potential components to be splitted, maximum 3 trials
    
    K = hp_posterior['m'].shape[1]
    
    candidates = torch.nonzero(hp_posterior['Nc'][0,:] > 2).flatten()

    if len(candidates) == 0:
        return 0, hp_posterior, data, -1

    q_of_z, _ , _ = mk_q_of_z(data, hp_posterior, hp_prior, opts)

    new_free_energy = torch.ones(1, candidates.max().item()+1) * float('inf')
    
    fc = mk_E_log_q_p_eta(data, hp_posterior, hp_prior, opts)
    
    log_lambda = mk_log_lambda(data, hp_posterior, hp_prior, opts)
    
    new_data = []
    new_q_of_z_cell = {}
    for c in candidates[:min(c_max, len(candidates))]:
        c = c.item()
        #my_disp(f'Splitting component {c}...')
        
        new_data0, new_q_of_z, info = split(c, data, q_of_z, hp_posterior, hp_prior, opts)
        new_data.append(new_data0)
        
        new_c = info['new_c']
        relating_n = torch.nonzero(torch.sum(new_q_of_z[:, [c, new_c]], dim=1) > 0.5).flatten()
        if len(relating_n) == 0:
            continue
        
        new_K = new_q_of_z.shape[1] 
        
        sub_q_of_z = new_q_of_z[relating_n][:,[c, new_c, new_K - 1]]

        sub_data = {}
        if opts['use_kd_tree']:
            sub_data['kdtree'] = new_data[c]['kdtree'][relating_n]
        else:
            sub_data['given_data'] = new_data[c]['given_data'][:, relating_n]
        
        sub_hp_posterior = mk_hp_posterior(sub_data, sub_q_of_z, hp_prior, opts)

        sub_f, sub_hp_posterior, _, sub_q_of_z = update_posterior2(sub_data, sub_hp_posterior, hp_prior, opts, 0, 10, 1)

        if sub_q_of_z.shape[1] < 3:
            continue
        if len(torch.nonzero(torch.sum(sub_q_of_z, dim=0) < 1.0e-10)) > 1:
            continue
        
        new_log_lambda = log_lambda.clone()
        sub_log_lambda = mk_log_lambda(new_data[c], sub_hp_posterior, hp_prior, opts)
        
        insert_indices = [c, new_c] + list(range(new_K-1, new_K + sub_q_of_z.shape[1] - 3)) # 两个list相加即把两个list的元素合并

        # Ensure new_log_lambda has enough columns
        required_columns = max(insert_indices) + 1
        if new_log_lambda.size(1) < required_columns:
            new_log_lambda = torch.cat([new_log_lambda, torch.zeros(new_log_lambda.size(0), required_columns - new_log_lambda.size(1))], dim=1)
        
        # print(f"new_log_lambda:{new_log_lambda}")
        # print(f"insert_indices:{insert_indices}")
        # print(f"sub_log_lambda:{sub_log_lambda}")

        new_log_lambda[:][:, insert_indices] = sub_log_lambda
        
        new_fc = fc.clone()

        # print(f"new_fc:{new_fc}")
        # test_fc = mk_E_log_q_p_eta(sub_data, sub_hp_posterior, hp_prior, opts)
        # print(f"test_fc:{test_fc}")

        if new_fc.size(1) < required_columns:
            new_fc = torch.cat([new_fc, torch.zeros(new_fc.size(0), required_columns - new_fc.size(1))], dim=1)
        new_fc[0,insert_indices] = mk_E_log_q_p_eta(sub_data, sub_hp_posterior, hp_prior, opts)
        # print(f"new_fc:{new_fc}")
        # print(f"new_free_energy:{new_free_energy}")
        new_free_energy[0,c] = mk_free_energy(new_data[c], sub_hp_posterior, hp_prior, opts, new_fc, new_log_lambda)[0]
        new_q_of_z[relating_n[:,None], :] = 0
#         print(f"relating_n_shape:{relating_n.shape}")
#         print(f"sub_q_of_z_shape:{sub_q_of_z.shape}")
        for i in range(len(relating_n.numpy())):
            new_q_of_z[relating_n[i], insert_indices] = sub_q_of_z[i,:]

        new_q_of_z_cell[c] = new_q_of_z
        
        # print(f"new_q_of_z:{new_q_of_z}")
        # print(f"new_q_of_z_cell[c]:{new_q_of_z_cell[c]}")
        
    #print(f"new_q_of_z:{new_q_of_z}")
    free_energy, c = torch.min(new_free_energy, dim=1)
    
    # print(f"candidates:{candidates[:min(c_max, len(candidates))]}")
    # print(f"c:{c}")
    if torch.isinf(free_energy):
        return 0, hp_posterior, data, -1
    
    data = new_data[c.item()]
    
    hp_posterior = mk_hp_posterior(data, new_q_of_z_cell[c.item()], hp_prior, opts)
    
    return free_energy.item(), hp_posterior, data, c.item()


def split(c, data, q_of_z, hp_posterior, hp_prior, opts):
    # q_of_z: N*K

    new_data = data
    
    if opts['init_of_split'] == 'pc':  # principal eigenvector

        arg1_data = new_data['given_data']

        dir = divide_by_principal_component(arg1_data, \
                                      hp_posterior['B'][:,:,c]/hp_posterior['eta'][0][c], \
                                      hp_posterior['m'][:,c])
        q_of_z_c1 = torch.zeros(q_of_z.shape[0], 1) # N*1
        q_of_z_c2 = q_of_z[:,c].unsqueeze(1).clone()   # N*1
        
        I = (dir >= 0).nonzero(as_tuple=True)[0] # 1*Number of non-negative elements in dir
        
        q_of_z_c1[I] = q_of_z[I, c].unsqueeze(1).clone()
        
        q_of_z_c2[I] = 0

        
    else:
        q_of_z_c = q_of_z[:, c]
        if opts['init_of_split'] == 'rnd':  # random
            r = torch.rand(q_of_z.shape[0], 1)
        elif opts['init_of_split'] == 'rnd_close':  # make close clusters
            r = 0.5 + (torch.rand(q_of_z.shape[0], 1) - 0.5) * 0.01
        elif opts['init_of_split'] == 'close_f':  # one is almost zero
            r = 0.98 + torch.rand(q_of_z.shape[0], 1) * 0.01
        else:
            
            raise ValueError('Unknown algorithm')

    
        q_of_z_c1 = q_of_z_c * r
        q_of_z_c2 = q_of_z_c * (1 - r)

    new_q_of_z = torch.zeros(q_of_z.size(0), q_of_z.size(1) + 1)  # N*(K+1)
    
    # Create a true copy of q_of_z
    q_z_copy = q_of_z.clone()
    
    # Fill new_q_of_z with values from q_z_copy
    new_q_of_z[:, :-2] = q_z_copy[:, :-1]
    new_q_of_z[:, -1] = q_z_copy[:, -1]
    
    new_q_of_z[:, c] = q_of_z_c1.squeeze()
    
    new_c = new_q_of_z.size(1) - 2 # python counts from 0
    
    new_q_of_z[:, new_c] = q_of_z_c2.squeeze()
    
    info = {'new_c': new_c}

    return new_data, new_q_of_z, info

def divide_by_principal_component(data, covariance, mean):
    """
    data: Tensor of shape (features, samples)
    covariance: Covariance matrix of shape (features, features)
    mean: Mean vector of shape (features,)
    """
    N = data.shape[1]

    if data.shape[0] <= 16:
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

def free_energy_decreased(free_energy, new_free_energy, warn_when_increasing, opts): # minimize variation free energy, equivalent to maximize ELBO
    diff = new_free_energy - free_energy
    
    if torch.abs(diff / free_energy) < opts['threshold']:
        return 0
    
    elif diff > 0:
        if warn_when_increasing:
            if torch.abs(diff / free_energy) > 1.0e-3:
                raise ValueError(f"the free energy increased. the diff is {new_free_energy - free_energy}")
            else:
                print(f"Warning: the free energy increased. the diff is {new_free_energy - free_energy}")
        return 0
    elif diff == 0:
        return 0
    else:
        return 1

def update_posterior2(data, hp_posterior, hp_prior, opts, upkdtree=0, ite=float('inf'), do_sort = 0):
    #my_disp('### updating posterior ...')
    free_energy = float('inf')
    
    i = 0
    last_Nc = 0
    start_sort = 0
    
    while True:
        i += 1
        new_free_energy, log_lambda = mk_free_energy(data, hp_posterior, hp_prior, opts)
        #print(f"new_free_energy:{new_free_energy}")
        #disp_status(new_free_energy, hp_posterior, opts)
        
        if (not torch.isinf(torch.tensor(ite)).item() and i >= ite) or (torch.isinf(torch.tensor(ite)).item() and not free_energy_decreased(free_energy, new_free_energy, 0, opts)):
            free_energy = new_free_energy
            if do_sort and opts['do_sort'] and not start_sort:
                start_sort = 1
            else:
                break
        
        last_Nc = hp_posterior['Nc']
        free_energy = new_free_energy
        q_of_z, data, _ = mk_q_of_z(data, hp_posterior, hp_prior, opts, log_lambda)
        
        freq = opts['recursive_expanding_frequency']
        
        # if opts['use_kd_tree'] and upkdtree and (freq == 1 or i % freq == 1):
        #     data, q_of_z = expand_recursively_until_convergence(data, q_of_z, hp_posterior, hp_prior, opts)
        #print(f'q_of_z:{q_of_z}')
        if opts['algorithm'] == 'vdp' and torch.sum(q_of_z[:, -1]) > 1.0e-20:
            q_of_z = torch.cat((q_of_z, torch.zeros(q_of_z.shape[0], 1)), dim=1) # 增加一个component代表infinite inactive components
        
        if start_sort:
            q_of_z, _ = sort_q_of_z(data, q_of_z, opts)

        
        if opts['algorithm'] == 'vdp' and torch.sum(q_of_z[:, -2]) < 1.0e-10:
            indices = [i for i in range(q_of_z.shape[1]) if i != q_of_z.shape[1] - 2]
            q_of_z = q_of_z[:, indices]
        
        hp_posterior = mk_hp_posterior(data, q_of_z, hp_prior, opts)
    
    #my_disp('### updating posterior ... done.')
    return free_energy, hp_posterior, data, q_of_z 

def sort_q_of_z(data, q_of_z, opts):
    
    #my_disp('sorting...')
    
    if opts['use_kd_tree']:
        Nc = torch.matmul(torch.tensor([d['N'] for d in data['kdtree']]), q_of_z)  # 1*K
    else:
        Nc = torch.sum(q_of_z, dim=0)  # 1*K
    
    if opts['algorithm'] == 'vdp':
        sorted_indices = torch.argsort(Nc[:-1], descending=True)
        sorted_indices = torch.cat((sorted_indices, torch.tensor([Nc.size(0) - 1])))
    else:
        sorted_indices = torch.argsort(Nc, descending=True)
    
    q_of_z = q_of_z[:, sorted_indices]
    
    #my_disp('sorting... done.')
    
    return q_of_z, sorted_indices

def mkopts_vdp():
    opts = {}
    opts['alpha'] = 1
    opts['algorithm'] = 'vdp'
    opts['use_kd_tree'] = 0
    opts['initial_K'] = 1
    opts['do_greedy'] = 1
    opts['do_split'] = 0
    opts['do_merge'] = 0
    opts['get_E_pi'] = 0
    opts['get_log_likelihood'] = 0
    opts['ite'] = 100
    opts['threshold'] = 1e-5
    return opts

## main function
def vdpgmm(given_data, hp_prior = None, opts=mkopts_vdp()):
    start_time = time.time()
    if opts is None:
        opts = {}

    if isinstance(given_data, np.matrix):  # Check if given_data is a sparse matrix
        given_data = given_data.toarray()  # Convert sparse matrix to dense

    opts.setdefault('algorithm', 'vdp')
    opts.setdefault('do_sort', '0')
    opts.setdefault('get_q_of_z', 1)
    opts.setdefault('get_E_pi', 0)
    opts.setdefault('get_log_likelihood', 1)
    opts.setdefault('threshold', 1.0e-5)
    opts.setdefault('initial_depth', 3)
    opts.setdefault('initial_K', 1)
    opts.setdefault('ite', np.inf)
    opts.setdefault('do_split', 0)
    opts.setdefault('do_merge', 0)
    opts.setdefault('do_greedy', 1)
    opts.setdefault('init_of_split', 'pc')
    opts.setdefault('recursive_expanding_depth', 2)
    opts.setdefault('recursive_expanding_threshold', 1.0e-1)
    opts.setdefault('recursive_expanding_frequency', 3)
    print('final opts:', opts)

    if 'seed' in opts:
        np.random.seed(opts['seed'])
    else:
        seed = np.random.get_state()
        results = {'seed': seed}

    data = {'given_data': given_data}

    if hp_prior == None:
        hp_prior = mk_hp_prior(data, opts)

    if 'hp_posterior' in opts:
        opts['use_kd_tree'] = 0
        if opts['get_q_of_z']:
            results['q_of_z'] = mk_q_of_z(data, opts['hp_posterior'], hp_prior, opts)
        if opts['get_log_likelihood']:
            results['log_likelihood'] = mk_log_likelihood(data, opts['hp_posterior'], hp_prior, opts)
        return results

    if 'q_of_z' in opts:
        q_of_z = opts['q_of_z']
    else:
        q_of_z = rand_q_of_z(data, opts['initial_K'], opts)

    hp_prior['q_of_z'] = q_of_z
    hp_posterior = mk_hp_posterior(data, q_of_z, hp_prior, opts)
    
    #return hp_prior, hp_posterior, data, opts

    if opts['do_greedy']:
        free_energy, hp_posterior, data = greedy(data, hp_posterior, hp_prior, opts)
    else:
        free_energy, hp_posterior, data = split_merge(data, hp_posterior, hp_prior, opts)

    disp_status(free_energy, hp_posterior, opts)
    results = {
        'algorithm': opts['algorithm'],
        'elapsed_time': time.time() - start_time,
        'free_energy': free_energy,
        'hp_prior': hp_prior,
        'hp_posterior': hp_posterior,
        'K': hp_posterior['eta'].shape[1],
        'opts': opts
    }

    if opts['get_q_of_z']:
        results['q_of_z'], _, _ = mk_q_of_z(data, hp_posterior, hp_prior, opts)

    if opts['get_log_likelihood']:
        results['log_likelihood'] = mk_log_likelihood(data, hp_posterior, hp_prior, opts)

    if opts['get_E_pi']:
        results['E_pi'] = mk_E_pi(hp_posterior, hp_prior, opts)

    return results

#import os
#print(os.listdir('.'))
# Test
#features = pd.read_csv("TF_6features_test.csv")
#features = features.astype("float32")


# # Convert DataFrame to PyTorch tensors
# X = torch.tensor(features.values[:6,:])

# input_dim = X.shape[1]


# result = onlinevdpgm(X, opts=mkopts_vdp())
# # print(f"hp_prior:{result['hp_prior']}")
# # print(f"hp_posterior:{result['hp_posterior']}")