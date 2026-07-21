import numpy as np
import torch
import torch.nn.functional as F
from vdpgm import mkopts_vdp, sort_q_of_z, normalizeln, lpt2lcpt, log_sum_exp, normalize, detln, my_disp, disp_status, gamma_multivariate_ln, free_energy_decreased

# def mkopts_vdp():
#     opts = {}
#     opts['alpha'] = 1
#     opts['algorithm'] = 'vdp'
#     opts['use_kd_tree'] = 0
#     opts['initial_K'] = 1
#     opts['do_greedy'] = 1
#     opts['do_split'] = 0
#     opts['do_merge'] = 0
#     opts['get_E_pi'] = 0
#     opts['get_log_likelihood'] = 0
#     opts['ite'] = 10
#     opts['threshold'] = 1e-3
#     return opts

def mk_hp_posterior_ol(data, q_of_z, hp_prior, init_prior, opts):

    # The last component of q_of_z represents the infinite rest of components
    # The last component is the prior.
    # q_of_z: N*K
    # q_of_z[:, -1] is the rest of responsibilities.

    threshold_for_N = 1.0e-30
    K = q_of_z.shape[1]

    D, N = data['given_data'].shape

    if opts['algorithm'] == 'vdp':
        true_Nc = torch.sum(q_of_z, dim=0)  # 1*K  Nc is expected number of observations in cluster c
        true_Nc = true_Nc.unsqueeze(0)
        # q_of_z[:, -1] = 0  # Commented out as it's not in use

    Nc = torch.sum(q_of_z, dim=0, keepdim=True)  # 1*K true_Nc != Nc for VDP

    sum_x = torch.matmul(data['given_data'], q_of_z)  # r_nk * xn in PRML xkBar Product Nk  D*N * N*K

    I = torch.where(torch.squeeze(Nc) > threshold_for_N)[0]
    
    inv_Nc = torch.zeros(1, K)  
    inv_Nc[:, I] = 1.0 / Nc[:, I] # 1*K

    len_Nc = Nc.shape[1]
    len_prior = hp_prior['eta'].shape[1]

    if len_Nc < len_prior:
        num = len_prior - len_Nc
        for _ in range(num):
            hp_prior['eta'] = torch.cat([hp_prior['eta'][:,:-2], hp_prior['eta'][:,-1:]], dim=1)
            hp_prior['xi'] = torch.cat([hp_prior['xi'][:,:-2], hp_prior['xi'][:,-1:]], dim=1)
            hp_prior['m'] = torch.cat([hp_prior['m'][:, :-2], hp_prior['m'][:, -1:]], dim=1)
            hp_prior['B'] = torch.cat([hp_prior['B'][:, :, :-2], hp_prior['B'][:, :, -1:]], dim=2)
            hp_prior['inv_B'] = torch.cat([hp_prior['inv_B'][:, :, :-2], hp_prior['inv_B'][:, :, -1:]], dim=2)
            
    elif len_Nc > len_prior:
        num = len_Nc - len_prior
        for _ in range(num):
            hp_prior['eta'] = torch.cat([hp_prior['eta'], torch.tensor([init_prior['eta0']]).unsqueeze(0)], dim = 1)
            hp_prior['xi'] = torch.cat([hp_prior['xi'], torch.tensor([init_prior['xi0']]).unsqueeze(0)], dim = 1)
            hp_prior['m'] = torch.cat([hp_prior['m'], init_prior['m0']], dim=1)
            hp_prior['B'] = torch.cat([hp_prior['B'], init_prior['B0'].unsqueeze(2)], dim=2)
            hp_prior['inv_B'] = torch.cat([hp_prior['inv_B'], torch.inverse(init_prior['B0']).unsqueeze(2)], dim=2)

    hp_posterior = {}
    hp_posterior['eta'] = hp_prior['eta'] + Nc # 1*K + 1*K
    hp_posterior['xi'] = hp_prior['xi'] + Nc
    
    means = sum_x * inv_Nc.repeat(D,1) # D*K .* D*1
    
    hp_posterior['m'] = torch.zeros(D, K)
    hp_posterior['inv_B'] = torch.zeros(D, D, K)
    hp_posterior['B'] = torch.zeros(D, D, K)

    v0 = torch.zeros(D,K)
    for c in range(K): 
        #print(means[:, c].unsqueeze(1).shape)
        #print(hp_prior['m'][:,c].unsqueeze(1).shape)
        v = data['given_data'] - means[:, c].unsqueeze(1).repeat(1, N)  # D*N  v is (x_n-x_bar) in PRML
        v0[:,c] = means[:, c] - hp_prior['m'][:,c]  # v0 is (x_bar - m0) in PRML
        
 
        # print(f"v:{v}")
        # print(f"v0[:,c]:{v0[:,c]}")      
        # hp_prior['m0'] should have the size D*1 instead of D
        
        hp_posterior['B'][:,:,c] = hp_prior['B'][:,:,c] + \
                            torch.mm((q_of_z[:, c].unsqueeze(1).repeat(1,D).t() * v), v.t()) + \
                            Nc[0][c] * hp_prior['xi'][0][c] * torch.mm(v0[:, c].unsqueeze(1), v0[:, c].unsqueeze(1).t()) / hp_posterior['xi'][0][c]

#         if torch.isnan(hp_posterior['B'][:,:,c]).any().item():
#             print("Find Nan")
#             print(f"v0[:,c]:{v0[:,c]}")
#             print(f"hp_prior['B'][:,:,c]:{hp_prior['B'][:,:,c]}")
#             print("Dignosis 2nd term")
#             print(f"second term:{torch.mm((q_of_z[:, c].unsqueeze(1).repeat(1,D).t() * v), v.t())}")
#             print(q_of_z[:, c].unsqueeze(1).repeat(1,D).t() * v)
#             print(torch.mm((q_of_z[:, c].unsqueeze(1).repeat(1,D).t() * v), v.t()))
#             print("Dignosis 3rd term")
#             print(f"third term:{Nc[0][c] * hp_prior['xi'][0][c] * torch.mm(v0[:, c].unsqueeze(1), v0[:, c].unsqueeze(1).t()) / hp_posterior['xi'][0][c]}")
            
        # print(f"c:{c}")
        
        # print(f"2nd term:{torch.mm((q_of_z[:, c].unsqueeze(1).repeat(1,D).t() * v), v.t())}")
        # print(f"3rd term:{Nc[0][c] * hp_prior['xi'][0][c] * torch.mm(v0, v0.t()) / hp_posterior['xi'][0][c]}")
        # print(f"Nc[0][c]:{Nc[0][c]}")
        # print(f"hp_prior['xi'][0][c]:{hp_prior['xi'][0][c]}")
        # print(f"torch.mm(v0[:, c].unsqueeze(1), v0[:, c].unsqueeze(1).t()):{torch.mm(v0[:, c].unsqueeze(1), v0[:, c].unsqueeze(1).t())}")
        # print(f"hp_posterior['xi'][0][c]:{hp_posterior['xi'][0][c]}")

        
        hp_posterior['inv_B'][:,:,c] = torch.inverse(hp_posterior['B'][:,:,c])
        
        hp_posterior['m'][:,c] = (sum_x[:,c] + hp_prior['xi'][0][c]*hp_prior['m'][:,c]) / hp_posterior['xi'][0][c]   
        #  torch.mm((q_of_z[:, c].unsqueeze(1).repeat(1,D).t() * v), v.t()) ---- (D*N .* D*N) @ N*D = D*D
        #  size(Nc) = 1*K Nc[c] = Nc[0,c]
        #  hp_posterior['B'][:,:,c] --- D*D

    if opts['algorithm'] == 'vdp':
        hp_posterior['gamma'] = torch.zeros(2, K)
        hp_posterior['gamma'][0, :] = hp_prior['gamma'][0, :] + true_Nc
        
        hp_posterior['gamma'][1, :] = hp_prior['gamma'][1, :] + true_Nc.sum() - torch.cumsum(true_Nc, dim=1)
    else:
        raise ValueError('Unknown algorithm')

    hp_posterior['Nc'] = Nc
    
    if opts['algorithm'] == 'vdp':
        hp_posterior['true_Nc'] = true_Nc

    hp_posterior['q_of_z'] = q_of_z

    return hp_posterior, hp_prior

# def detln(X): # calculate log_determinant
#     """
#     Calculate the log determinant of matrix X.

#     Args:
#         X (numpy.ndarray): Input matrix.

#     Returns:
#         float: Log determinant of X.
#     """
#     try:
#         L = torch.linalg.cholesky(X+1e-6*torch.eye(X.shape[0]))
#     except torch.linalg.LinAlgError:
#         print(X)
#         raise ValueError("Error in Choleski decomposition for detln")

#     diag_L = torch.diag(L)
#     log_det = 2 * torch.sum(np.log(diag_L))
#     return log_det

# def gamma_multivariate_ln(x, p):
#     # x: array(1, K)
#     # p: scalar
#     #
#     # x must be greater than (p-1)/2
#     # x should be greater than p/2
#     #
#     # Gamma_p(x) = pi^(p(p-1)/4) prod_(j=1)^p Gamma(x+(1-j)/2)
#     # log Gamma_p(x) = p(p-1)/4 log pi + sum_(j=1)^p log Gamma(x+(1-j)/2)
    
#     c0 = torch.tensor([_+1 for _ in range(p)])
#     c0 = c0.reshape(p,1)

#     K = len(x)
#     gammaln_val = torch.special.gammaln(x.repeat(p, 1) + 0.5 * (1 - c0.repeat(1,K)))
#     val = p * (p - 1) * 0.25 * torch.log(torch.tensor(np.pi)) + torch.sum(gammaln_val, dim=0)     
#     val = val.unsqueeze(0) # 1*K
#     return val

def mk_E_log_q_p_eta_ol(data, hp_posterior, hp_prior, init_prior, opts):
    # returns E_q(eta)[log q(eta)/p(eta)] % Eq.(10.74) and Eq.(10.77) in PRML
    # fc : 1 by K
    D = hp_posterior['m'].shape[0]
    K = hp_posterior['eta'].shape[1]
    log_det_B = torch.zeros(1, K) # log determinant
    term_eta = torch.zeros(2, K)
    
    # in case the dimension does not match
    len_Nc = K
    len_prior = hp_prior['eta'].shape[1]

    if len_Nc < len_prior:
        num = len_prior - len_Nc
        for _ in range(num):
            hp_prior['eta'] = torch.cat([hp_prior['eta'][:,:-2], hp_prior['eta'][:,-1:]], dim=1)
            hp_prior['xi'] = torch.cat([hp_prior['xi'][:,:-2], hp_prior['xi'][:,-1:]], dim=1)
            hp_prior['m'] = torch.cat([hp_prior['m'][:, :-2], hp_prior['m'][:, -1:]], dim=1)
            hp_prior['B'] = torch.cat([hp_prior['B'][:, :, :-2], hp_prior['B'][:, :, -1:]], dim=2)
            hp_prior['inv_B'] = torch.cat([hp_prior['inv_B'][:, :, :-2], hp_prior['inv_B'][:, :, -1:]], dim=2)
            
    elif len_Nc > len_prior:
        num = len_Nc - len_prior
        for _ in range(num):
            hp_prior['eta'] = torch.cat([hp_prior['eta'], torch.tensor([init_prior['eta0']]).unsqueeze(0)], dim = 1)
            hp_prior['xi'] = torch.cat([hp_prior['xi'], torch.tensor([init_prior['xi0']]).unsqueeze(0)], dim = 1)
            hp_prior['m'] = torch.cat([hp_prior['m'], init_prior['m0']], dim=1)
            hp_prior['B'] = torch.cat([hp_prior['B'], init_prior['B0'].unsqueeze(2)], dim=2)
            hp_prior['inv_B'] = torch.cat([hp_prior['inv_B'], torch.inverse(init_prior['B0']).unsqueeze(2)], dim=2)

    for c in range(K):
        log_det_B[0, c] = detln(hp_posterior['B'][:,:,c])
        d = hp_posterior['m'][:, c].unsqueeze(1) - hp_prior['m'][:, c].unsqueeze(1)  # D*1 d = (m_k - m0) in PRML 
        term_eta[0, c] = torch.sum(hp_posterior['inv_B'][:, :, c] * (hp_prior['xi'][0][c] * torch.mm(d, d.t())))
        term_eta[1, c] = torch.sum(hp_posterior['inv_B'][:, :, c] * hp_prior['B'][:,:,c]) - D
    
    #print(term_eta)
    E_log_q_p_mean = (
        0.5 * D * (hp_prior['xi'] / hp_posterior['xi'] 
                   - torch.log(hp_prior['xi'] / hp_posterior['xi']) 
                   - 1)
        + 0.5 * hp_posterior['eta'] * term_eta[0, :]
    )
    
   
    
    psi_sum = torch.sum(
        torch.special.psi(((hp_posterior['eta'] + 1).repeat(D, 1) - torch.arange(1, D + 1).unsqueeze(1).repeat(1, K))* 0.5) ,
        dim=0
    ) # 1*K

    #print(f"E_log_q_p_mean:{E_log_q_p_mean}")
    
    
    E_log_q_p_cov0 = torch.zeros(K,K)
    for c in range(K):
        E_log_q_p_cov0[:,c] = (
            0.5 * hp_prior['eta'][0][c] * (log_det_B - detln(hp_prior['B'][:,:,c])) # 1*K
            + 0.5 * hp_posterior['Nc'] * psi_sum
            + 0.5 * hp_posterior['eta'][:, c] * term_eta[1, c]#.unsqueeze(0)
            + gamma_multivariate_ln(hp_prior['eta'][:,c] * 0.5, D)
            - gamma_multivariate_ln(hp_posterior['eta'][:,c] * 0.5, D)
        ).squeeze()
    
    # print(f"0.5 * hp_posterior['Nc'] * psi_sum:{0.5 * hp_posterior['Nc'] * psi_sum}")
    # print(f"detln(hp_prior['B'][:,:,c]):{detln(hp_prior['B'][:,:,c])}")
    # print(f"gamma_multivariate_ln(hp_prior['eta'][:,c] * 0.5, D):{gamma_multivariate_ln(hp_prior['eta'][:,c] * 0.5, D)}")
    # print(f"gamma_multivariate_ln(hp_posterior['eta'][:,c] * 0.5, D):{gamma_multivariate_ln(hp_posterior['eta'][:,c] * 0.5, D)}")
        
    E_log_q_p_cov = torch.sum(E_log_q_p_cov0, dim = 0) 

    # print(f"E_log_q_p_cov0:{E_log_q_p_cov0}")
    # print(f"E_log_q_p_cov:{E_log_q_p_cov}")
    # print(f"E_log_q_p_mean:{E_log_q_p_mean}")
    # print(f"E_log_q_p_cov:{E_log_q_p_cov}")
    # print(f"psi_sum:{psi_sum}")
    # print(E_log_q_p_cov)
    if torch.any(E_log_q_p_mean < -1.0e-5):
        print(E_log_q_p_mean)
        raise ValueError('E_log_q_p_mean is negative.')
    
    # if torch.any(E_log_q_p_cov < -1.0e-5):
    #     print(E_log_q_p_cov)
    #     raise ValueError('E_log_q_p_cov is negative.')
    
    fc = E_log_q_p_mean + E_log_q_p_cov
    
    return fc

# def log_sum_exp(x, dim, y=None):
#     """
#     Compute log(sum(exp(x))) along a specified dimension in a numerically stable way.
    
#     Args:
#         x (torch.Tensor): Input tensor.
#         dim (int): Dimension along which to perform the operation.
#         y (Optional): This parameter is not used in the original function.
        
#     Returns:
#         torch.Tensor: Result of log(sum(exp(x))) along the specified dimension.
#     """
#     # Find the maximum value along the specified dimension
#     x_max, _ = torch.max(x, dim=dim, keepdim=True)
    
#     dims = torch.ones(1,len(x.shape))
#     dims[0][dim] = x.shape[dim]
#     dims = dims.numpy()
#     dims = dims.astype(int)
#     # Subtract the maximum value from x for numerical stability
#     x = x - x_max.repeat(dims[0][0],dims[0][1])
    
#     # Compute the log-sum-exp
#     val = x_max + torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))
#     return val # N*1

def mk_log_lambda_ol(data, hp_posterior, hp_prior, init_prior, opts):
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

    log_lambda[:, -1] = log_lambda[:, -1] - torch.log(1 - torch.exp(torch.special.psi(torch.tensor(init_prior['alpha'])) - torch.special.psi(torch.tensor(1 + init_prior['alpha'])))) 
        
    return log_lambda

def mk_free_energy_ol(data, hp_posterior, hp_prior, init_prior, opts, fc=None, log_lambda=None):

    if fc is None and log_lambda is None:
        fc = mk_E_log_q_p_eta_ol(data, hp_posterior, hp_prior, init_prior, opts)
        log_lambda = mk_log_lambda_ol(data, hp_posterior, hp_prior, init_prior, opts)

    N, K = log_lambda.shape
    
    if opts['algorithm'] == 'vdp':
        len_gamma = hp_posterior['gamma'].shape[1]
        
        E_log_p_of_V = (torch.special.gammaln(hp_posterior['gamma'].sum(dim=0)) 
                        - torch.special.gammaln(hp_prior['gamma'].sum(dim=0)) 
                        - torch.special.gammaln(hp_posterior['gamma']).sum(dim=0) 
                        + torch.special.gammaln(hp_prior['gamma']).sum(dim=0)  
                        + ((hp_posterior['gamma'][0, :] - 1) 
                           * (torch.special.psi(hp_posterior['gamma'][0, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0)))) 
                        + ((hp_posterior['gamma'][1, :] - hp_prior['gamma'][1, :]) 
                           * (torch.special.psi(hp_posterior['gamma'][1, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0)))))
        extra_term = E_log_p_of_V.sum()

        # print(f"(hp_posterior['gamma'][0, :] - 1):{(hp_posterior['gamma'][0, :] - 1)}")
        # print(f"(torch.special.psi(hp_posterior['gamma'][0, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0))):{(torch.special.psi(hp_posterior['gamma'][0, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0)))}")
    
        # print(f"(hp_posterior['gamma'][1, :] - hp_prior['gamma'][1, :]):{(hp_posterior['gamma'][1, :] - hp_prior['gamma'][1, :])}")
        # print(f"(torch.special.psi(hp_posterior['gamma'][1, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0))):{(torch.special.psi(hp_posterior['gamma'][1, :]) - torch.special.psi(hp_posterior['gamma'].sum(dim=0)))}")
        # print(f"extra_term:{extra_term}")
        # print(f"E_log_p_of_V:{E_log_p_of_V}") 
    else:
        raise ValueError('Unknown algorithm')

    free_energy = extra_term + fc.sum() - log_sum_exp(log_lambda, 1).sum()

    # print(f"fc:{fc}")
    # print(f"extra_term:{extra_term}")
    # print(f"fc.sum():{fc.sum()}")
    # print(f"log_sum_exp(log_lambda, 1).sum():{log_sum_exp(log_lambda, 1).sum()}")
    return free_energy, log_lambda

# def my_disp(*args):
#     # Initialize a persistent variable to keep track of whether the function is disabled
#     if not hasattr(my_disp, 'is_disabled'):
#         my_disp.is_disabled = False

#     # Check if the second argument is provided (to disable the function)
#     if len(args) == 2:
#         my_disp.is_disabled = args[1]
#         return

#     # If not disabled, print the input arguments
#     if not my_disp.is_disabled:
#         print(*args)
        
# def disp_status(free_energy, hp_posterior, opts):
#     if opts['algorithm'] == 'vdp':
#         Nc = hp_posterior['true_Nc']
#     else:
#         Nc = hp_posterior['Nc']

#     my_disp(f"F={free_energy:.5g};    Nc=[{', '.join(map(str, Nc.numpy()))}];")

# def free_energy_decreased(free_energy, new_free_energy, warn_when_increasing, opts): # minimize variation free energy, equivalent to maximize ELBO
#     diff = new_free_energy - free_energy
    
#     if torch.abs(diff / free_energy) < opts['threshold']:
#         return 0
    
#     elif diff > 0:
#         if warn_when_increasing:
#             if torch.abs(diff / free_energy) > 1.0e-3:
#                 raise ValueError(f"the free energy increased. the diff is {new_free_energy - free_energy}")
#             else:
#                 print(f"Warning: the free energy increased. the diff is {new_free_energy - free_energy}")
#         return 0
#     elif diff == 0:
#         return 0
#     else:
#         return 1
    
# def lpt2lcpt(lpt, dimension):
#     """
#     Create a log conditional probability table from a log probability table.

#     Args:
#         lpt (torch.Tensor): Input log probability tensor.
#         dimension (int): Dimension along which to normalize (1 or 2).

#     Returns:
#         torch.Tensor: Log conditional probability tensor.
#     """
#     assert dimension in [1, 2], "dimension must be either 1 or 2."

#     # Adjust dimensions to Python's 0-based indexing
#     dimension = dimension - 1
#     the_other_dimension = 1 - dimension

#     # Permute the tensor to bring the specified dimension to the front
#     # lpt N*K
#     lpt = lpt.permute(dimension, the_other_dimension)

#     # Calculate log_sum_exp along the second dimension (originally specified dimension)
#     log_sum_exp_lpt = log_sum_exp(lpt, dim=1)  # Mx1

#     # Subtract log_sum_exp_lpt from lpt
#     lcpt = lpt - log_sum_exp_lpt.repeat(1,lpt.shape[1])

#     # Permute back to the original dimension order
#     lcpt = lcpt.permute(dimension, the_other_dimension)

#     return lcpt

# def normalizeln(M ,dimension):
    
#     M = lpt2lcpt(M, dimension)
    
#     return M

# def normalize(m, dim):
#     # set the sum of responsibility of each data point to 1, i.e. sum(q_of_z, dim = 1) = 1
#     # Return m normalized along the specified 'dim'
#     #
#     # e.g.
#     # m: i by j by k by ...
#     # m = torch.sum(normalize(m, 2), dim=2)
#     # m[i, :, k, ...] = torch.ones(1, J, 1, ...)

#     dims = [1] * len(m.shape)
#     dims[dim] = m.shape[dim]
#     m = m / torch.sum(m, dim=dim, keepdim=True).repeat(dims)
#     return m

def mk_q_of_z_ol(data, hp_posterior, hp_prior, init_prior, opts, log_lambda=None):
    # Compute log_lambda if not provided
    # log_lambda(n,i) is S(n,i) in the paper
    if log_lambda is None:
        log_lambda = mk_log_lambda_ol(data, hp_posterior, hp_prior, init_prior, opts)
    
    # Compute q_of_z (responsibility)
    q_of_z = torch.exp(normalizeln(log_lambda, 1))
    
    return q_of_z, data, log_lambda

def update_posterior2_ol(data, hp_posterior, hp_prior, init_prior, opts, upkdtree=0, ite=float('inf'), do_sort = 0):
    #my_disp('### updating posterior ...')
    free_energy = float('inf')
    
    i = 0
    last_Nc = 0
    start_sort = 0
    
    while True:
        i += 1
        new_free_energy, log_lambda = mk_free_energy_ol(data, hp_posterior, hp_prior, init_prior, opts)
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
        q_of_z, data, _ = mk_q_of_z_ol(data, hp_posterior, hp_prior, init_prior, opts, log_lambda)
        
        #print(f"q_of_z:{q_of_z}")
        hp_posterior, hp_prior = mk_hp_posterior_ol(data, q_of_z, hp_prior, init_prior, opts)
        
        if torch.isnan(hp_posterior['B']).any().item():
            print(f"hp_posterior['B']:{hp_posterior['B']}")
            
    #my_disp('### updating posterior ... done.')
    return free_energy, hp_posterior, data, q_of_z 

# def sort_q_of_z(data, q_of_z, opts):
    
#     #my_disp('sorting...')
    
#     if opts['use_kd_tree']:
#         Nc = torch.matmul(torch.tensor([d['N'] for d in data['kdtree']]), q_of_z)  # 1*K
#     else:
#         Nc = torch.sum(q_of_z, dim=0)  # 1*K
    
#     if opts['algorithm'] == 'vdp':
#         sorted_indices = torch.argsort(Nc[:-1], descending=True)
#         sorted_indices = torch.cat((sorted_indices, torch.tensor([Nc.size(0) - 1])))
#     else:
#         sorted_indices = torch.argsort(Nc, descending=True)
    
#     q_of_z = q_of_z[:, sorted_indices]
    
#     #my_disp('sorting... done.')
    
#     return q_of_z, sorted_indices

def online_vdpgmm(data, hp_prior, init_prior, opts = mkopts_vdp()):
    """
    Conduct VDPGMM in an incremental manner.

    Args:
        data (torch.Tensor): current batch of observations size = D*N
        hp_prior (dict): current prior (posterior of the previous batch)
        init_prior (dict): initial prior between each batch, used to set the parameters of inactivated components
        
    Returns:
        hp_posterior (dict): current posterior estimated using the prior and current batch
        
    """  
    data0 = {'given_data': data}
    new_q_z = None
    
    while True:
        # reset q_of_z    
        if new_q_z == None:
            q_of_z = torch.rand(data.shape[1], hp_prior['Nc'].shape[1]) # N*K
            q_of_z = normalize(q_of_z,1)
            #new_q_z = q_of_z
        
        hp_posterior, hp_prior = mk_hp_posterior_ol(data0, q_of_z, hp_prior, init_prior, opts)

        free_energy, log_lambda = mk_free_energy_ol(data0, hp_posterior, hp_prior, init_prior, opts)

        while True:
            new_free_energy, new_hp_posterior, data0, q_of_z = update_posterior2_ol(data0, hp_posterior, hp_prior, init_prior, opts, 1, opts['ite'])

            if not free_energy_decreased(free_energy, new_free_energy, 0, opts):
                break

            free_energy = new_free_energy

            hp_posterior = new_hp_posterior

        padding0 = (0, hp_posterior['Nc'].shape[1] - hp_prior['Nc'].shape[1])
        hp_prior_Nc_padded = torch.nn.functional.pad(hp_prior['Nc'], padding0)
        hp_posterior['Nc'] += hp_prior_Nc_padded

        padding1 = (0, hp_posterior['true_Nc'].shape[1] - hp_prior['true_Nc'].shape[1])
        hp_prior_Nc_padded = torch.nn.functional.pad(hp_prior['true_Nc'], padding0)
        hp_posterior['true_Nc'] += hp_prior_Nc_padded

        # reset hp_prior
        hp_prior = hp_posterior.copy()

        K = hp_posterior['Nc'].shape[1]        
        M, I = torch.max(hp_posterior['q_of_z'], dim=1)

        if hp_posterior['Nc'][0,-2] >= 1 and (I == K-1).any():
            #print("additional component")
            # if the additional component is activated, then add a new additional component and start the next iteration
            # reset Nc
            hp_prior['Nc'] = hp_prior['Nc'] - torch.sum(hp_prior['q_of_z'], keepdim = True, dim=0)
            hp_prior['true_Nc'] = hp_prior['true_Nc'] - torch.sum(hp_prior['q_of_z'], keepdim = True, dim=0)
            
            new_q_z = torch.zeros(data.shape[1], K + 1)
            new_q_z[:,:-1] = normalize(hp_prior['q_of_z'],1)

            hp_prior['eta'] = torch.cat([hp_prior['eta'], torch.tensor([init_prior['eta0']]).unsqueeze(0)], dim = 1)
            hp_prior['xi'] = torch.cat([hp_prior['xi'], torch.tensor([init_prior['xi0']]).unsqueeze(0)], dim = 1)
            hp_prior['m'] = torch.cat([hp_prior['m'], init_prior['m0']], dim=1)
            hp_prior['B'] = torch.cat([hp_prior['B'], init_prior['B0'].unsqueeze(2)], dim=2)
            hp_prior['inv_B'] = torch.cat([hp_prior['inv_B'], torch.inverse(init_prior['B0']).unsqueeze(2)], dim=2)

            hp_prior['gamma'] = torch.cat((hp_prior['gamma'], torch.tensor([[1], [init_prior['alpha']]])), dim=1)
            hp_prior['Nc'] = torch.cat([hp_prior['Nc'],torch.tensor([0]).unsqueeze(0)], dim = 1)
            hp_prior['true_Nc'] = torch.cat([hp_prior['true_Nc'],torch.tensor([0]).unsqueeze(0)], dim = 1)

            q_of_z = new_q_z
            #print(f"new_q_z:{new_q_z}")

        else:
            # if the additional component is not activated, break the loop
            #print('no additional component')
        
            # renormalize q_of_z with K-1 dimensions
            q_of_z0 = hp_posterior['q_of_z'][:,:K-1]
            q_of_z0 = torch.cat([q_of_z0, torch.zeros(q_of_z0.shape[0],1)], dim = 1)
            q_of_z0 = normalize(q_of_z0, 1)

            hp_prior['q_of_z'] = q_of_z0
            hp_prior['eta'] = torch.cat([hp_prior['eta'][:,:-1], torch.tensor([init_prior['eta0']]).unsqueeze(0)], dim=1)
            hp_prior['xi'] = torch.cat([hp_prior['xi'][:,:-1], torch.tensor([init_prior['xi0']]).unsqueeze(0)], dim=1)
            hp_prior['m'] = torch.cat([hp_prior['m'][:, :-1], init_prior['m0']], dim=1)
            hp_prior['B'] = torch.cat([hp_prior['B'][:, :, :-1], init_prior['B0'].unsqueeze(2)], dim=2)
            hp_prior['inv_B'] = torch.cat([hp_prior['inv_B'][:, :, :-1], torch.inverse(init_prior['B0']).unsqueeze(2)], dim=2)
            hp_prior['gamma'] = torch.cat([hp_prior['gamma'][:, :-1], torch.tensor([1, init_prior['alpha']]).unsqueeze(1)], dim=1)
            
            # i = 0
            # while i < (hp_prior['Nc'].shape[1] - 1):
            #     #print('Pruning!!!!!')
            #     if not (I == i).any() or hp_prior['Nc'][0][i] <= 1e-2:
            #         hp_prior['q_of_z'] = torch.cat([hp_prior['q_of_z'][:,:i], hp_prior['q_of_z'][:,i+1:]], dim=1)
            #         hp_prior['q_of_z'] = normalize(hp_prior['q_of_z'], 1)
            #         hp_prior['eta'] = torch.cat([hp_prior['eta'][:,:i], hp_prior['eta'][:,i+1:]], dim=1)
            #         hp_prior['xi'] = torch.cat([hp_prior['xi'][:,:i], hp_prior['xi'][:,i+1:]], dim=1)
            #         hp_prior['m'] = torch.cat([hp_prior['m'][:,:i], hp_prior['m'][:,i+1:]], dim=1)
            #         hp_prior['B'] = torch.cat([hp_prior['B'][:,:,:i], hp_prior['B'][:,:,i+1:]], dim=2)
            #         hp_prior['inv_B'] = torch.cat([hp_prior['inv_B'][:,:,:i], hp_prior['inv_B'][:,:,i+1:]], dim=2)
            #         hp_prior['gamma'] = torch.cat([hp_prior['gamma'][:,:i], hp_prior['gamma'][:,i+1:]], dim=1)
            #         hp_prior['Nc'] = torch.sum(hp_prior['q_of_z'], dim=0, keepdim=True)
            #         hp_prior['true_Nc'] = torch.sum(hp_prior['q_of_z'], dim=0, keepdim=True)

            #     i += 1

            hp_posterior_no_additional = hp_prior
            #print(f"sum_Nc:{torch.sum(hp_posterior_no_additional['Nc'])}")
            #disp_status(free_energy, hp_posterior, opts)
            return hp_posterior_no_additional, free_energy


#     q_of_z_set[:500,:2] = result['hp_posterior']['q_of_z']
#     q_of_z_set1, _ = sort_q_of_z(data, q_of_z_set[:, :K+1], opts)
#     q_of_z_set2 = normalize(q_of_z_set1[:,:K],1)
#     M, I = torch.max(q_of_z_set2, dim=1)
