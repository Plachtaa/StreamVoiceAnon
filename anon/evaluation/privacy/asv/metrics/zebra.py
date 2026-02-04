import numpy as np
from scipy.special import expit


def ece(tar, non, plo):
    if np.isscalar(tar):
        tar = np.array([tar])
    if np.isscalar(non):
        non = np.array([non])
    if np.isscalar(plo):
        plo = np.array([plo])

    ece = np.zeros(plo.shape)
    for i, p in enumerate(plo):
        ece[i] = expit(p) * (-np.log(expit(tar + p))).mean()
        ece[i] += expit(-p) * (-np.log(expit(-non - p))).mean()

    ece /= np.log(2)

    return ece


def int_ece(x, epsilon=1e-6):
    """
    Z(X) = avg( [(x-3)*(x-1) + 2*log(x)] / [4 * (x-1)^2] )  # x as LR
         = avg( 0.25 - 1/[2*(x-1)] + log(x)/[2*(x-1)^2] )  # a = log(x)
         = 0.25 + 0.5 * avg( - 1/[(exp(a) - 1)] + a/[(exp(a)-1)^2] )  # b = exp(a)-1
         = 0.25 + 0.5 * avg( (a-b))/b^2 )
    """
    idx = (~np.isinf(x)) & (np.abs(x) > epsilon)
    contrib = np.zeros(len(x))  # for +inf, the contribution is 0.25; the later on constant term
    xx = x[idx]
    LRm1 = np.exp(xx) - 1
    contrib[idx] = (xx - LRm1) / LRm1 ** 2
    # if x == 0 or if x < epsilon
    # numerical issue of exp() function for small values around zero, thus also hardcoded value
    contrib[(np.abs(x) < epsilon)] = -0.5  # Z(0) = 0 = 0.25 + (-0.5)/2
    return 0.25 + contrib.mean() / 2


def dece(tar_llrs, nontar_llrs):
    int_diff_ece = int_ece(tar_llrs) + int_ece(-nontar_llrs)
    return int_diff_ece / np.log(2)


def max_abs_LLR(matedScores_opt, nonMatedScores_opt):
    max_abs_LLR = np.abs(np.hstack((matedScores_opt,nonMatedScores_opt))).max() / np.log(10)
    return max_abs_LLR


def category_tag_evidence(max_abs_LLR):
    # smallest float value we can numerically tract in this computational environment
    eps = np.finfo(float).eps

    # Here are our categorical tags, inspired by the ENFSI sacle on the stength of evidence
    # Please feel free to try out your own scale as well :)
    # dict: { TAG : [min max] value of base10 LLRs }
    categorical_tags = {
        '0': np.array([0, eps]),
        'A': np.array([eps, 1]),
        'B': np.array([1, 2]),
        'C': np.array([2, 4]),
        'D': np.array([4, 5]),
        'E': np.array([5, 6]),
        'F': np.array([6, np.inf])
    }

    # pre-computation for easier later use
    cat_ranges = np.vstack(list(categorical_tags.values()))
    cat_idx = np.argwhere((cat_ranges < max_abs_LLR).sum(1) == 1).squeeze()
    cat_tag = list(categorical_tags.keys())[cat_idx]
    return cat_tag


def fast_actDCF(tar, non, plo, normalize=False):
    D = 1
    if not np.isscalar(plo):
        D = len(plo)
    T = len(tar)
    N = len(non)

    ii = np.argsort(np.hstack([-plo,tar]))
    r = np.zeros(T+D)
    r[ii] = np.arange(T+D) + 1
    r = r[:D]
    Pmiss = r - np.arange(start=D, step=-1, stop=0)

    ii = np.argsort(np.hstack([-plo, non]))  # -plo are thresholds
    r = np.zeros(N+D)
    r[ii] = np.arange(N+D) + 1
    r = r[:D]  # rank of thresholds
    Pfa = N - r + np.arange(start=D, step=-1, stop=0)

    Pmiss = Pmiss / T
    Pfa = Pfa / N

    Ptar = expit(plo)
    Pnon = expit(-plo)
    dcf = Ptar * Pmiss + Pnon * Pfa

    if normalize:
        dcf /= np.minimum([Ptar, Pnon])

    return dcf
