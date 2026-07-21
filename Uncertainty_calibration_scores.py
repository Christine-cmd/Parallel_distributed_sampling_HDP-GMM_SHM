import numpy as np
from scipy.special import logsumexp
from scipy.special import gammaln, logsumexp

# -------------------------- Calculate responsibilities from assignment variables (only required in sampling methods)-------------------------- #
# collapsed Gibbs - CRP DPGMM
def log_mvt_pdf(x, mu, Sigma, nu):
    x = np.atleast_1d(x); d = x.size
    xc = x - mu
    sgn, logdet = np.linalg.slogdet(Sigma)
    if sgn <= 0: raise ValueError("Sigma must be SPD.")
    q = xc @ np.linalg.solve(Sigma, xc)
    return (gammaln(0.5*(nu + d)) - gammaln(0.5*nu)
            - 0.5*d*np.log(nu*np.pi) - 0.5*logdet
            - 0.5*(nu + d)*np.log1p(q/nu))

def niw_predictive_params(mu_n, kappa_n, Psi_n, nu_n):
    mu_n = np.asarray(mu_n).ravel()
    Psi_n = np.asarray(Psi_n)
    d = mu_n.size
    df = nu_n - d + 1
    if df <= 0: raise ValueError("NIW predictive df <= 0")
    scale = (kappa_n + 1.0) / (kappa_n * df) * Psi_n
    return mu_n, scale, df

# --- CRP mixture weights ---
def crp_weights_closed(n_k, eps=1e-300):
    """Ignore 'new' component: pi_k = n_k / N."""
    n_k = np.asarray(n_k, float).ravel()
    N = float(n_k.sum())
    return n_k / max(N, eps)

def crp_weights_open(n_k, alpha, eps=1e-300):
    """Include 'new': pi_k = n_k/(N+alpha), pi_new = alpha/(N+alpha)."""
    n_k = np.asarray(n_k, float).ravel()
    N = float(n_k.sum())
    denom = N + alpha
    return n_k / max(denom, eps), alpha / max(denom, eps)

# --- responsibilities (closed-menu) ---
def dpgmm_crp_responsibilities_closed(X, components, n_k, eps=1e-300):
    """
    X: (N,D)
    components: list of dicts per k: {'mu_n','kappa_n','Psi_n','nu_n'}  (NIW posterior)
    n_k: (K,) cluster sizes in this posterior state
    Returns R: (N,K), rows sum to 1.
    """
    X = np.asarray(X)
    K = len(components)
    # mixture weights
    pi = crp_weights_closed(n_k, eps=eps)
    log_pi = np.log(pi + eps)[:, None]  # (K,1)

    # predictive log-likelihoods, stacked (K,N)
    L = np.empty((K, X.shape[0]), float)
    preds = []
    for k in range(K):
        mu, S, df = niw_predictive_params(
            components[k]['mu_n'], components[k]['kappa_n'],
            components[k]['Psi_n'], components[k]['nu_n']
        )
        preds.append((mu, S, df))
        L[k] = [log_mvt_pdf(x, mu, S, df) for x in X]

    Z = log_pi + L                  # (K,N)
    R = np.exp(Z - logsumexp(Z, axis=0))  # (K,N)
    return R.T                      # (N,K)

# collapsed Gibbs - CRF HDP-GMM
def predict_cluster_posteriors_CRF(
    data_groups,                 # list of arrays; data_groups[d] has shape (n_d, D)
    global_components,           # list-like; each has .logpdf(x) (Student-t or Gaussian)
    n_kd_total,                  # array (K, D_docs): counts N_{kd} per component k, doc d
    mk,                          # array (K,): table counts m_k
    alpha, gamma,                # scalars
    ignore_new=True              # 'closed menu' (no new component)
):
    """
    Returns:
      R_list: list over docs; each is (n_d, K) of responsibilities r_{ik} = P(k|x_i)
      idx_map: list of (doc_index, within_doc_index) for concatenation bookkeeping
    """
    K, D_docs = n_kd_total.shape
    m = float(np.sum(mk))
    R_list, idx_map = [], []

    for d in range(D_docs):
        n_kd = n_kd_total[:, d].astype(float)
        n_d  = float(n_kd.sum())
        # CRF closed-menu weights:
        numer = n_kd + alpha * (mk / (m + gamma))
        denom = n_d + alpha * (m / (m + gamma))
        pi_k  = numer / denom  # (K,)

        Xd = np.asarray(data_groups[d])
        n_d_pts = Xd.shape[0]
        R_d = np.empty((n_d_pts, K), dtype=float)


        log_pi = np.log(pi_k + 1e-300)
        for i, x in enumerate(Xd):
            log_fk = np.array([global_components[k].logpdf(x).item() for k in range(K)], dtype=float)
            # r_{ik} ∝ pi_k * f_k(x)
            z = log_pi + log_fk
            z -= logsumexp(z)
            R_d[i, :] = np.exp(z)
            idx_map.append((d, i))
        R_list.append(R_d)

    return R_list, idx_map  # each R_d sums to 1 across K

# -------------------------- Brier Score-------------------------- #
def brier_score(p_hat, y_true, classes=None, sample_weight=None):
    """
    Multiclass Brier score:
      BS = (1/N) * sum_i sum_c (p_hat[i,c] - 1{y_i=c})^2
    """
    p_hat = np.asarray(p_hat, dtype=float)
    N, C = p_hat.shape
    if classes is None:
        classes = np.unique(y_true)
    class_to_idx = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_true], dtype=int)
    Y = np.eye(C)[y_idx]   # one-hot
    if sample_weight is None:
        return np.mean(np.sum((p_hat - Y)**2, axis=1))
    w = np.asarray(sample_weight, dtype=float)
    w /= w.sum()
    return np.sum(w * np.sum((p_hat - Y)**2, axis=1))

def brier_by_class(p_hat, y_true, classes=None):
    """Classwise Brier (averaged over samples of that class)."""
    p_hat = np.asarray(p_hat, dtype=float)
    if classes is None:
        classes = np.unique(y_true)
    C = len(classes)
    class_to_idx = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_true], dtype=int)
    Y = np.eye(C)[y_idx]
    out = {}
    for c, cl in enumerate(classes):
        mask = (y_idx == c)
        if mask.any():
            out[cl] = np.mean(np.sum((p_hat[mask] - Y[mask])**2, axis=1))
        else:
            out[cl] = np.nan
    return out

def brier_skill_score(p_hat, y_true, classes=None):
    """
    BSS = 1 - BS / BS_ref, where BS_ref uses climatology (class frequency).
    """
    if classes is None:
        classes = np.unique(y_true)
    C = len(classes)
    # climatology
    counts = np.array([(y_true == c).sum() for c in classes], dtype=float)
    p_ref = counts / counts.sum()
    P_ref = np.tile(p_ref, (len(y_true), 1))
    BS = brier_score(p_hat, y_true, classes)
    BS_ref = brier_score(P_ref, y_true, classes)
    return 1.0 - (BS / BS_ref if BS_ref > 0 else np.nan)

# --------- learn P(class|cluster) and produce class probs --------- #
def estimate_Q_c_given_k(R, y_true, classes=None, reg=1e-6):
    """
    Estimate Q[c,k] = P(class=c | cluster=k) from a labeled set.
    R: (N, K) responsibilities on those same N labeled points
    y_true: length-N integer/label array
    """
    if classes is None:
        classes = np.unique(y_true)
    C = len(classes)
    class_to_idx = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_true], dtype=int)
    N, K = R.shape
    counts = np.zeros((C, K), dtype=float)
    for i in range(N):
        counts[y_idx[i], :] += R[i, :]
    counts += reg
    Q = counts / counts.sum(axis=0, keepdims=True)  # column-normalize over classes
    return Q, classes

def class_probs_from_R(R, Q):
    """Given R (N,K) and Q (C,K), return P_hat (N,C) with P(y=c|x)=sum_k Q[c,k] R[i,k]."""
    return R @ Q.T  # (N,K) @ (K,C)^T -> (N,C)

# ------------------------------ usage ------------------------------ #
# 1) If you ALREADY have class probabilities P_hat:
# BS  = brier_score(P_hat, y_true)
# BSc = brier_by_class(P_hat, y_true)
# BSS = brier_skill_score(P_hat, y_true)

# 2) If you have soft cluster posteriors R and a learned Q:
# P_hat = class_probs_from_R(R, Q)
# BS = brier_score(P_hat, y_true)

# 3) Full HDP-GMM path (compute R from your CRF state, learn Q, score):
def brier_from_hdp_state(
    data_groups, global_components, n_kd_total, mk, alpha, gamma,
    y_true, labeled_mask=None, classes=None
):
    """
    - data_groups: list of groups/docs; concatenate for scoring
    - If you only want to use a subset to estimate Q (semi-supervised),
      pass labeled_mask (length N_all) marking which points are labeled.
    """
    # (a) responsibilities per point
    R_list, idx_map = predict_cluster_posteriors_CRF(
        data_groups, global_components, n_kd_total, mk, alpha, gamma, ignore_new=True
    )
    R = np.vstack(R_list)  # (N_all, K)

    # (b) choose labeled subset for Q (default: all points + their y_true)
    if labeled_mask is None:
        R_lab = R
        y_lab = y_true
    else:
        labeled_mask = np.asarray(labeled_mask, dtype=bool)
        R_lab = R[labeled_mask]
        y_lab = np.asarray(y_true)[labeled_mask]

    # (c) estimate Q and build P_hat on all points
    Q, classes = estimate_Q_c_given_k(R_lab, y_lab, classes=classes, reg=1e-6)

    P_hat = class_probs_from_R(R, Q)

    # (d) Brier scores
    BS  = brier_score(P_hat, y_true, classes=classes)
    BSc = brier_by_class(P_hat, y_true, classes=classes)
    BSS = brier_skill_score(P_hat, y_true, classes=classes)
    return BS, BSc, BSS, P_hat, R, Q

def dpgmm_brier_from_responsibilities(
    R, y_true, labeled_mask=None, classes=None, sample_weight=None
):
    """
    Use the exact same pipeline as HDP-GMM: R -> learn Q on labeled subset -> P_hat -> Brier.
    """
    R = np.asarray(R, float)
    if labeled_mask is None:
        R_lab, y_lab = R, y_true
    else:
        m = np.asarray(labeled_mask, bool)
        R_lab, y_lab = R[m], np.asarray(y_true)[m]

    Q, classes = learn_Q_soft(R_lab, y_lab, classes=classes, reg=1e-6)
    P_hat = class_probs_from_R(R, Q)

    BS  = brier_score(P_hat, y_true, classes=classes, sample_weight=sample_weight)
    BSc = brier_by_class(P_hat, y_true, classes=classes)
    BSS = brier_skill_score(P_hat, y_true, classes=classes)
    return BS, BSc, BSS, P_hat, R, Q, classes

# ------------------------------ ECE ------------------------------ #

# ---------- soft mapping and probs ----------
def learn_Q_soft(R, y_true, classes=None, reg=1e-6):
    if classes is None:
        classes = np.unique(y_true)
    C, K = len(classes), R.shape[1]
    cls2idx = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([cls2idx[y] for y in y_true], int)
    E = np.zeros((C, K), float)
    for i in range(R.shape[0]):        # expected counts per (class, cluster)
        E[y_idx[i]] += R[i]
    Q = (E + reg) / (E + reg).sum(axis=0, keepdims=True)
    return Q, classes

def class_probs_from_R(R, Q):
    return R @ Q.T  # (N,K) @ (K,C)^T -> (N,C)

# ---------- ECE (top-label & OvR) ----------
def ece_toplabel(P_hat, y_true, n_bins=15):
    N, C = P_hat.shape
    conf = P_hat.max(axis=1)
    pred = P_hat.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx  = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m): 
            continue
        acc_b  = correct[m].mean()
        conf_b = conf[m].mean()
        ece += m.mean() * abs(acc_b - conf_b)
    return ece

def ece_ovr(P_hat, y_true, n_bins=15, classes=None):
    P_hat = np.asarray(P_hat, float)
    N, C = P_hat.shape
    if classes is None:
        classes = np.arange(C)
    cls2idx = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([cls2idx[y] for y in y_true], int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for c in range(C):
        p_c = P_hat[:, c]
        idx = np.clip(np.digitize(p_c, bins) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            m = (idx == b)
            if not np.any(m):
                continue
            acc_b  = (y_idx[m] == c).mean()
            conf_b = p_c[m].mean()
            ece += (m.sum() / N) * abs(acc_b - conf_b)
    return ece

# ---------- K-fold CV on Q only (no extra data needed) ----------
def kfold_indices(N, k=5, shuffle=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    if shuffle:
        rng.shuffle(idx)
    return np.array_split(idx, k)

def probs_with_cv_Q(R, y_true, classes=None, kfold=5, reg=1e-6):
    """
    Learn Q on K-1 folds, predict P_hat on the held-out fold; repeat and stitch.
    R: (N,K) responsibilities from a fitted model (HDP or DPGMM)
    """
    N = R.shape[0]
    folds = kfold_indices(N, k=kfold, shuffle=True, seed=0)
    if classes is None:
        classes = np.unique(y_true)
    C = len(classes)
    P_hat = np.zeros((N, C), float)

    for val_idx in folds:
        train_idx = np.setdiff1d(np.arange(N), val_idx, assume_unique=True)
        Q, classes = learn_Q_soft(R[train_idx], y_true[train_idx], classes=classes, reg=reg)
        P_hat[val_idx] = class_probs_from_R(R[val_idx], Q)
    return P_hat, classes

def ece_ovr_classbalanced(P_hat, y_true, n_bins=15):
    N, C = P_hat.shape
    y_true = np.asarray(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    eces = []
    for c in range(C):
        p = P_hat[:, c]
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        ece_c = 0.0
        for b in range(n_bins):
            m = (idx == b)
            if not m.any(): 
                continue
            acc_b  = (y_true[m] == c).mean()
            conf_b = p[m].mean()
            ece_c += (m.sum() / N) * abs(acc_b - conf_b)
        eces.append(ece_c)
    return float(np.mean(eces))

# ---------- example usage ----------
# def compare_calibration_from_R(R_hdp, R_dpgmm, y_true, n_bins=15, kfold=5):
#     # HDP
#     P_hdp, classes = probs_with_cv_Q(R_hdp, y_true, kfold=kfold)
#     ece_hdp_top = ece_toplabel(P_hdp, y_true, n_bins=n_bins)
#     ece_hdp_ovr = ece_ovr(P_hdp, y_true, n_bins=n_bins, classes=classes)

#     # DPGMM
#     P_dpg, _ = probs_with_cv_Q(R_dpgmm, y_true, classes=classes, kfold=kfold)
#     ece_dpg_top = ece_toplabel(P_dpg, y_true, n_bins=n_bins)
#     ece_dpg_ovr = ece_ovr(P_dpg, y_true, n_bins=n_bins, classes=classes)

#     return {
#         "HDP": {"ECE_toplabel": ece_hdp_top, "ECE_OvR": ece_hdp_ovr, "P_hat": P_hdp, "classes": classes},
#         "DPGMM": {"ECE_toplabel": ece_dpg_top, "ECE_OvR": ece_dpg_ovr, "P_hat": P_dpg, "classes": classes}
#     }
