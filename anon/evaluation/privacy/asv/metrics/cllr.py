import logging
import numpy as np
from scipy.special import expit

from .helpers import optimal_llr
from .utils.io import read_targets_and_nontargets

logger = logging.getLogger(__name__)

def compute_cllr(score_file, key_file, compute_eer=False):
    # Computing Cllr and min Cllr for binary decision classifiers
    tar, non = read_targets_and_nontargets(score_file=score_file, key_file=key_file)

    cllr_act = cllr(tar, non)
    if compute_eer:
        cllr_min, eer = min_cllr(tar, non, compute_eer=True)
    else:
        cllr_min = min_cllr(tar, non)

    logger.info("Cllr (min/act): %.3f/%.3f" % (cllr_min, cllr_act))
    if compute_eer:
        logger.info("ROCCH-EER: %2.3f%%" % (100*eer))


def cllr(tar_llrs, nontar_llrs):
    """Computes the application-independent cost function
    It is an expected error of a binary decision
    based on the target and non-target scores (mated/non-mated)
    The higher wrost is the average error
    Parameters
    ----------
    tar_llrs : ndarray
        list of scores associated to target pairs
        (uncalibrated LLRs)
    nontar_llrs : ndarray
        list of scores associated to non-target pairs
        (uncalibrated LLRs)

    Returns
    -------
    c : float
        Cllr of the scores.

    Notes
    -----
    Adaptation of the cllr measure of Brümmer et al. [2]
    Credits to Andreas Nautsch (EURECOM)
    https://gitlab.eurecom.fr/nautsch/cllr


    References
    ----------

    .. [2] Brümmer, N., & Du Preez, J. (2006).
    Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
    """
    tar_posterior = expit(tar_llrs)  # sigmoid
    non_posterior = expit(-nontar_llrs)
    if any(tar_posterior == 0) or any(non_posterior == 0):
        return np.inf

    c1 = (-np.log(tar_posterior)).mean() / np.log(2)
    c2 = (-np.log(non_posterior)).mean() / np.log(2)
    c = (c1 + c2) / 2
    return c


def min_cllr(tar_llrs, nontar_llrs, monotonicity_epsilon=1e-6, compute_eer=False, return_opt=False):
    """Computes the minimum application-independent cost function
    under calibrated scores (LLR)
    It is an expected error of a binary decision
    based on the target and non-target scores (mated/non-mated)
    The higher wrost is the average error
    Parameters
    ----------
    tar_llrs : ndarray
        list of scores associated to target pairs
        (uncalibrated LLRs)
    nontar_llrs : ndarray
        list of scores associated to non-target pairs
        (uncalibrated LLRs)
    monotonicity_epsilon : float
        Unsures monoticity of the optimal LLRs
    compute_eer : bool
        Returns ROCCH-EER
    return_opt : bool
        Returns optimal scores

    Returns
    -------
    cmin : float
        minCllr of the scores.
    eer : float
        ROCCH-EER
    tar : ndarray
        Target optimally calibrated scores (PAV)
    non : ndarray
        Non-target optimally calibrated scores (PAV)

    Notes
    -----
    Adaptation of the cllr measure of Brümmer et al. [2]
    Credits to Andreas Nautsch (EURECOM)
    https://gitlab.eurecom.fr/nautsch/cllr


    References
    ----------

    .. [2] Brümmer, N., & Du Preez, J. (2006).
    Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
    """
    if compute_eer:
        [tar, non, eer] = optimal_llr(tar_llrs, nontar_llrs, laplace=False, monotonicity_epsilon=monotonicity_epsilon,
                                     compute_eer=compute_eer)
        cmin = cllr(tar, non)
        if not return_opt:
            return cmin, eer
        else:
            return cmin, eer, tar, non
    else:
        [tar, non] = optimal_llr(tar_llrs, nontar_llrs, laplace=False, monotonicity_epsilon=monotonicity_epsilon)
        cmin = cllr(tar, non)
        if not return_opt:
            return cmin
        else:
            return cmin, tar, non
