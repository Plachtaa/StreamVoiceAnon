import logging
import numpy as np

from .utils.visualization import draw_linkability_scores
from .utils.io import read_targets_and_nontargets

logger = logging.getLogger(__name__)

def compute_linkability(score_file, key_file, omega=1.0, use_draw_scores=False, output_file=None):
    # Computing the global linkability measure for a list of linkage function score
    # score_file: path to score file
    # key_file: path to key file
    # omega: prior ratio (default is 1)
    # draw_scores: flag: draw the score distribution in a figure
    # output_file: output path of the png and pdf file (default is linkability_<score_file>)
    mated_scores, non_mated_scores = read_targets_and_nontargets(score_file=score_file, key_file=key_file)

    Dsys, D, bin_centers, bin_edges = linkability(mated_scores, non_mated_scores, omega)

    if use_draw_scores:
        if not output_file:
            output_file = "linkability_" + score_file
        draw_linkability_scores(mated_scores, non_mated_scores, Dsys, D, bin_centers, bin_edges, str(output_file))

    logger.info("linkability: %f" % (Dsys))


def linkability(mated_scores, non_mated_scores, omega=1):
    """Compute Linkability measure between mated
    and non-mated scores.

    Parameters
    ----------
    mated_scores : Array_like
        List of scores associated to mated pairs
    non_mated_scores : Array_like
        List of scores associated to non-mated pairs
    omega : float
        Prior ration P[mated]/P[non-mated]

    Returns
    -------
    Dsys : float
        Global linkability measure.
    D : ndarray
        Local linkability measure for each bin.
    bin_centers : ndarray
        Center of the bins (from historgrams).
    bin_edges : ndarray
        Edges of the bis (from histograms)

    Notes
    -----
    Adaptation of the linkability measure of Gomez-Barrero et al. [1]

    References
    ----------

    .. [1] Gomez-Barrero, M., Galbally, J., Rathgeb, C. and Busch,
    C., 2017. General framework to evaluate unlinkability in biometric
    template protection systems. IEEE Transactions on Information
    Forensics and Security, 13(6), pp.1406-1420.
    """
    # Limiting the number of bins (100 maximum or lower if few scores available)
    nBins = min(int(len(mated_scores) / 10), 100)

    # define range of scores to compute D
    bin_edges = np.linspace(min([min(mated_scores), min(non_mated_scores)]),
                            max([max(mated_scores), max(non_mated_scores)]),
                            num=nBins + 1, endpoint=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # compute score distributions using normalized histograms
    y1 = np.histogram(mated_scores, bins=bin_edges, density=True)[0]
    y2 = np.histogram(non_mated_scores, bins=bin_edges, density=True)[0]
    # LR = P[s|mated ]/P[s|non-mated]
    LR = np.divide(y1, y2, out=np.ones_like(y1), where=y2 != 0)
    # compute D
    D = 2*(omega*LR/(1 + omega*LR)) - 1
    # Def of D
    D[omega*LR <= 1] = 0
    # Taking care of inf/NaN
    mask = [True if (y2[i] == 0 and y1[i] != 0) else False for i in range(len(y1))]
    D[mask] = 1
    # Global measure using trapz numerical integration
    Dsys = np.trapz(x=bin_centers, y=D * y1)
    return Dsys, D, bin_centers, bin_edges

