import numpy as np
import ot
import torch
from sklearn.model_selection import train_test_split
########################################################################################################################################
########################################################################################################################################
## CODES TO SOLVE OPTIMAL TRANSPORT / LEARN MK QUANTILES AND RANKS :
########################################################################################################################################
########################################################################################################################################


def sample_grid(data, positive=False):
    """Sample the reference distribution."""
    n = data.shape[0]
    d = data.shape[1]
    R = np.linspace(0, 1, n)
    if positive == False:
        sampler = qmc.Halton(d=d)
        sample_gaussian = sampler.random(n=n + 1)[1:]
        sample_gaussian = norm.ppf(sample_gaussian, loc=0, scale=1)
        mu = []
        for i in range(n):
            Z = sample_gaussian[i]
            Z = Z / np.linalg.norm(Z)
            mu.append(R[i] * Z)
    else:
        mu = []
        for i in range(n):
            Z = np.random.exponential(scale=1.0, size=d)
            Z = Z / np.sum(Z)
            mu.append(R[i] * Z)
    return np.array(mu)


def T0(x, DATA, psi):
    """Returns the image of `x` by the OT map parameterized by `psi` towards the empirical distribution of `sample_sort`."""
    if len(x.shape) == 1:
        to_max = (DATA @ x) - psi
        res = DATA[np.argmax(to_max)]
    else:
        to_max = (DATA @ x.T).T - psi
        res = DATA[np.argmax(to_max, axis=1)]
    return res


def learn_psi(mu, data):
    M = ot.dist(data, mu) / 2
    res = ot.solve(M)
    g, f = res.potentials
    psi = 0.5 * np.linalg.norm(mu, axis=1) ** 2 - f
    psi_star = 0.5 * np.linalg.norm(data, axis=1) ** 2 - g
    to_return = [psi, psi_star]
    return to_return


def RankFunc(x, mu, psi, ksi=0):
    # ksi >0 computes a smooth argmax (LogSumExp). ksi is a regularisation parameter, hence approximates the OT map.
    if len(x.shape) == 1:
        to_max = ((mu @ x) - psi) * ksi
        to_sum = np.exp(to_max - np.max(to_max))
        weights = to_sum / (np.sum(to_sum))
        res = np.sum(mu * weights.reshape(len(weights), 1), axis=0)
    else:
        res = []
        for xi in x:
            to_max = ((mu @ xi) - psi) * ksi
            to_sum = np.exp(to_max - np.max(to_max))
            weights = to_sum / (np.sum(to_sum))
            res.append(np.sum(mu * weights.reshape(len(weights), 1), axis=0))
        res = np.array(res)
    # For exact recovery of the argsup, one can use T0.
    if ksi == 0:
        res = T0(x, mu, psi)
    return res


def QuantFunc(u, data, psi_star):
    return T0(u, data, psi_star)


from scipy.stats import qmc
from scipy.stats import norm


def MultivQuantileTreshold(data, alpha=0.9, positive=False):
    """To change the reference distribution towards a positive one, set positive = True."""
    data_calib, data_valid = train_test_split(data, test_size=0.25)
    # Solve OT
    mu = sample_grid(data_calib, positive=positive)
    psi, psi_star = learn_psi(mu, data_calib)
    # QUANTILE TRESHOLDS
    n = len(data_valid)
    Ranks_data_valid = RankFunc(data_valid, mu, psi)
    Norm_ranks_valid = np.linalg.norm(Ranks_data_valid, axis=1, ord=2)
    Quantile_Treshold = np.quantile(
        Norm_ranks_valid, np.min([np.ceil((n + 1) * alpha) / n, 1])
    )
    return (Quantile_Treshold, mu, psi, psi_star, data_calib)


def MultivVectorCalibration(data_calib, positive=False):
    """To change the reference distribution towards a positive one, set positive = True."""
    # Solve OT
    mu = sample_grid(data_calib, positive=positive)
    psi, psi_star = learn_psi(mu, data_calib)

    return (mu, psi, psi_star)


def ScoreClassif(pi, BarY):
    S = np.abs(BarY - pi)
    return S


####################################################
# OTCP
####################################################


def func_prediction_set(pi_test, range_BarY, Quantile_Treshold, mu, psi):
    """Returns prediction set for our method, for classification."""
    Prediction_Set = []
    for BarY in range_BarY:
        S_testy = ScoreClassif(pi_test, BarY)
        # Test if it is conform
        RankMK = RankFunc(S_testy, mu, psi)
        norm_RankMK = np.linalg.norm(RankMK, axis=1, ord=2)
        test = 1 * (norm_RankMK <= Quantile_Treshold)
        # Gather results
        Prediction_Set.append(test)
    Prediction_Set = np.array(Prediction_Set).T  # multi-hot encoding
    Prediction_Set = Prediction_Set * np.arange(
        1, pi_test.shape[1] + 1
    )  # replace ones by corresponding value of label
    Prediction_Set = [[i - 1 for i in l if i != 0] for l in Prediction_Set.tolist()]
    return Prediction_Set


from sklearn.preprocessing import LabelBinarizer  # One hot encoding


def calib_OTCP_classif(X_cal, y_cal, clf, alpha, K):
    enc = LabelBinarizer()
    range_BarY = enc.fit_transform(np.arange(K).reshape(K, 1))
    BarY_cal = enc.transform(y_cal)
    try:
        pi_cal = clf.predict_proba(X_cal)
    except:
        pi_cal = clf.predict(X_cal)
    S_cal = ScoreClassif(pi_cal, BarY_cal)
    Quantile_Treshold, mu, psi, psi_star, data_calib = MultivQuantileTreshold(
        S_cal, alpha=alpha, positive=True
    )
    L = [Quantile_Treshold, mu, psi, psi_star, clf, range_BarY]
    return L


def evaluate_OTCP_classif(Xtest, L):
    Quantile_Treshold, mu, psi, clf, range_BarY = L[0], L[1], L[2], L[4], L[5]
    try:
        pi_test = clf.predict_proba(Xtest)
    except:
        pi_test = clf.predict(Xtest)

    Prediction_Set = func_prediction_set(
        pi_test, range_BarY, Quantile_Treshold, mu, psi
    )
    return Prediction_Set


####################################################
# IP AND MS SCORES
####################################################


def InverseProba(probas, y):
    """
    Computes the Hinge Loss, with 'probas' of size (n,K) for n probabilities over K classes.
    y is the index of a class.
    """
    return (1 - probas)[:, y]


def MarginScore(probas, y):
    """
    Computes the Margin Score, with 'probas' of size (n,K) for n probabilities over K classes.
    y is the index of a class.
    """
    indexes = list(range(np.shape(probas)[1]))
    indexes.pop(y)
    MS = np.max(probas[:, indexes], axis=1) - probas[:, y]
    return MS


def calib_IP_MS_scores(pi_cal, y_cal, alpha):
    K = len(np.unique(y_cal))
    y = 0  # one iteration to initialize
    IP_score = InverseProba(pi_cal[y_cal == y], y)
    MS_score = MarginScore(pi_cal[y_cal == y], y)
    for y in range(1, K):
        s1 = InverseProba(pi_cal[y_cal == y], y)
        s2 = MarginScore(pi_cal[y_cal == y], y)
        IP_score = np.concatenate([IP_score, s1])
        MS_score = np.concatenate([MS_score, s2])
    IP_score = np.array(IP_score).T
    MS_score = np.array(MS_score).T

    n = len(y_cal)
    q = alpha * (1 + 1 / n)
    Q1 = np.quantile(IP_score, q)
    Q2 = np.quantile(MS_score, q)
    return (Q1, Q2)


def evaluate_IP_MS_scores(pi_test, Q1, Q2, K):
    Prediction_Set_IP = []
    Prediction_Set_MS = []
    for y in np.arange(K):
        test = InverseProba(pi_test, y) <= Q1
        Prediction_Set_IP.append(test)
        test = MarginScore(pi_test, y) <= Q2
        Prediction_Set_MS.append(test)
    Prediction_Set_MS = np.array(Prediction_Set_MS).T  # multi-hot encoding
    Prediction_Set_IP = np.array(Prediction_Set_IP).T  # multi-hot encoding
    Prediction_Set_MS = Prediction_Set_MS * np.arange(
        1, pi_test.shape[1] + 1
    )  # replace ones by corresponding value of label
    Prediction_Set_MS = [
        [i - 1 for i in l if i != 0] for l in Prediction_Set_MS.tolist()
    ]

    Prediction_Set_IP = Prediction_Set_IP * np.arange(
        1, pi_test.shape[1] + 1
    )  # replace ones by corresponding value of label
    Prediction_Set_IP = [
        [i - 1 for i in l if i != 0] for l in Prediction_Set_IP.tolist()
    ]
    return (Prediction_Set_IP, Prediction_Set_MS)


#################################################
# TO GET METRICS
#################################################


def get_metrics(predictions, y, X):
    """WSC-coverage takes time, thus it is commented. To compute it, it suffices to remove the '#'."""
    MarginalCoverage = np.mean([y[i] in predictions[i] for i in range(len(y))])
    Efficiency = np.mean(
        [len(predictions[i]) for i in range(len(y))]
    )  # size of prediction set
    Informativeness = np.mean(
        [1 * (len(predictions[i]) == 1) for i in range(len(y))]
    )  # proportion of singletons
    wsc_coverage = -1  # coverage.wsc_unbiased(X, y, predictions,verbose=False,delta=0.15,M=300) # verbose = True to show bar indicating time left for computation
    return (MarginalCoverage, Efficiency, Informativeness, wsc_coverage)


def get_OTCP_ordering(scores_cal, scores_test, positive=True):
    """
    Takes calibration and test scores and returns the ordering induced by OTCP for test scores.

    Parameters:
    -----------
    scores_cal : array-like
        Calibration scores used to learn the optimal transport mapping
    scores_test : array-like
        Test scores for which we want to compute the ordering

    Returns:
    --------
    mk_ranks : array
        The MK ranks computed by the optimal transport map for test scores
    mk_norms : array
        The norms of the MK ranks (used for conformity testing)
    ordering_indices : array
        Indices that sort the test data by MK rank norms (ascending order)
    """
    # Learn OT parameters from the calibration scores
    mu, psi, _ = MultivVectorCalibration(scores_cal, positive=positive)

    # Compute MK ranks for test scores using learned mapping
    mk_ranks = RankFunc(scores_test, mu, psi)

    # Compute norms of MK ranks (this is what's used for conformity testing)
    mk_norms = np.linalg.norm(mk_ranks, axis=1, ord=2)

    # Get ordering indices (sorted by norm, ascending)
    ordering_indices = np.argsort(mk_norms)

    return mk_ranks, mk_norms, ordering_indices


class OTCPOrdering:
    def __init__(self, positive=True):
        self.positive = positive
        self.mu_ = None
        self.psi_ = None
        self.psi_star_ = None
        self.is_fitted_ = False

    def fit(self, train_loader: torch.utils.data.DataLoader, train_params: dict):
        scores_cal = train_loader.dataset.cpu().numpy()
        scores_cal = np.asarray(scores_cal)

        # Learn OT parameters from calibration scores
        self.mu_, self.psi_, self.psi_star_ = MultivVectorCalibration(
            scores_cal, positive=self.positive
        )

        self.is_fitted_ = True
        return self

    def predict(self, scores_test):
        if not self.is_fitted_:
            raise ValueError(
                "This OTCPOrdering instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        scores_test = np.asarray(scores_test)

        # Compute MK ranks for test scores using learned mapping
        mk_ranks = RankFunc(scores_test, self.mu_, self.psi_)

        # Compute norms of MK ranks (this is what's used for conformity testing)
        mk_norms = np.linalg.norm(mk_ranks, axis=1, ord=2)

        # Get ordering indices (sorted by norm, ascending)
        ordering_indices = np.argsort(mk_norms)

        return mk_norms, ordering_indices

    def predict_ranks(self, scores_test):
        if not self.is_fitted_:
            raise ValueError(
                "This OTCPOrdering instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        scores_test = np.asarray(scores_test)

        # Compute MK ranks for test scores using learned mapping
        mk_ranks = RankFunc(scores_test, self.mu_, self.psi_)

        return mk_ranks
