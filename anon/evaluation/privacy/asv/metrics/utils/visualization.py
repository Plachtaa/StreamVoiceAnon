import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ..zebra import ece
from ..helpers import bayes_error_rate


def draw_linkability_scores(mated_scores, non_mated_scores, Dsys, D, bin_centers, bin_edges, output_file):
    """Draw both mated and non-mated score distributions
    and associated their associated local linkability

    Parameters
    ----------
    mated_scores : Array_like
        list of scores associated to mated pairs
    non_mated_scores : Array_like
        list of scores associated to non-mated pairs
    Dsys : float
        Global linkability measure.
    D : ndarray
        Local linkability measure for each bin.
    bin_centers : ndarray
        Center of the bins (from historgrams).
    bin_edges : ndarray
        Edges of the bis (from histograms)
    output_file : String
        Path to png and pdf output file.

    References
    ----------

    .. [1] Gomez-Barrero, M., Galbally, J., Rathgeb, C., & Busch,
     C. (2017). General framework to evaluate unlinkability in
     biometric template protection systems. IEEE Transactions
     on Information Forensics and Security, 13(6), 1406-1420.
    """
    # Colorblind and photocopy friendly colors
    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']
    legendLocation='upper left'
    plt.clf()
    # Kernel density estimate of the score
    ax = sns.kdeplot(mated_scores, fill=False, label='Same Speaker', color=colors[2], linewidth=2, linestyle='--')
    x1, y1 = ax.get_lines()[0].get_data()
    ax = sns.kdeplot(non_mated_scores, fill=False, label='Not Same Speaker', color=colors[0], linewidth=2, linestyle=':')
    x2, y2 = ax.get_lines()[1].get_data()
    # Associated local linkability
    ax2 = ax.twinx()
    lns3, = ax2.plot(bin_centers, D, label='$\mathrm{D}_{\leftrightarrow}(s)$', color=colors[3],linewidth=2)

    # #print omega * LR = 1 lines
    index = np.where(D <= 0)
    ax.axvline(bin_centers[index[0][0]], color='k', linestyle='--')

    # Figure formatting
    ax.spines['top'].set_visible(False)
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("Score")
    # Global Linkability
    ax.set_title("$\mathrm{D}_{\leftrightarrow}^{\mathit{sys}}$ = %.2f" % (Dsys),  y = 1.02)
    # Legends
    labs = [ax.get_lines()[0].get_label(), ax.get_lines()[1].get_label(), ax2.get_lines()[0].get_label()]
    lns = [ax.get_lines()[0], ax.get_lines()[1], lns3]
    ax.legend(lns, labs, loc = legendLocation)
    # Frame of Values
    ax.set_ylim([0, max(max(y1), max(y2)) * 1.05])
    ax.set_xlim([bin_edges[0]*0.98, bin_edges[-1]*1.02])
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel("$\mathrm{D}_{\leftrightarrow}(s)$")

    # Optional: getting rid of possible extensions
    outname= output_file.replace('.pdf', '').replace('.png', '').replace('.csv', '').replace('.txt', '')
    plt.savefig(outname + ".pdf", format="pdf")
    plt.savefig(outname + ".png", format="png")


def ape_plot(mated_scores, non_mated_scores, mated_scores_opt, non_mated_scores_opt, cllr, cmin, eer, output_file):
    """Draw both APE-plot for calibrated and uncalibrated input scores

    Parameters
    ----------
    mated_scores : Array_like
        list of scores associated to mated pairs
    non_mated_scores : Array_like
        list of scores associated to non-mated pairs
    mated_scores_opt : Array_like
        Calibrated mated scores
    non_mated_scores_opt : Array_like
        Calibrated non-mated scores
    cllr : float
        application independent cost-function on uncalibrated scores.
    cmin : float
        application independent cost-function on calibrated scores.
    eer : float
        ROCCH Equal Error Rate.
    output_file : String
        Path to png and pdf output file.

    References
    ----------

    .. [2] Brümmer, N., & Du Preez, J. (2006).
    Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
    """
    # Colorblind and photocopy friendly colors
    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']  # ['#edf8b1','#7fcdbb','#2c7fb8']
    legendLocation = 'upper right'
    plt.clf()
    ax = plt.gca()
    # Priors to consider
    # plo = np.concatenate((np.arange(-7, 7, 0.5),np.arange(7,50,2)))
    plo = np.arange(-7, 7, 0.25)
    pe = bayes_error_rate(mated_scores, non_mated_scores, plo)
    minPe = bayes_error_rate(mated_scores_opt, non_mated_scores_opt, plo)
    refPe = bayes_error_rate([0], [0], plo)
    l3, = plt.plot(plo, refPe, label='$\mathrm{P}^{ref}_{e}$', color='black', linewidth=2, linestyle=':')
    l2, = plt.plot(plo, minPe, label='$\mathrm{P}^{min}_{e}$', color=colors[0], linewidth=2)
    l1, = plt.plot(plo, pe, label='$\mathrm{P}_{e}$', color=colors[3], linewidth=2, linestyle='--')
    leer = plt.plot([min(plo), max(plo)], [eer, eer], label='EER', color='black', linewidth=1, linestyle='-.')
    # Information of the figure
    ax.set_ylabel("P(error)")
    ax.set_xlabel("logit prior")
    ax.set_title("$\mathrm{C}_{LLR}$ = %.2f, $\mathrm{C}_{LLR}^{min}$ = %.2f, EER = %.2f" % (cllr, cmin, eer), y=1.02)
    ax.legend(loc=legendLocation)
    # Saving Figure (the replacements of extentions are optional)
    outname = output_file.replace('.pdf', '').replace('.png', '').replace('.csv', '').replace('.txt', '')
    plt.savefig(outname + ".pdf", format="pdf")
    plt.savefig(outname + ".png", format="png")


def ece_plot(matedScores_opt, nonMatedScores_opt, dece, max_abs_LLR, cat_tag, output_file):
    colors = ['#e66101','#fdb863','#b2abd2','#5e3c99']# ['#edf8b1','#7fcdbb','#2c7fb8']
    figureTitle = ''
    if figureTitle == '':
     figureTitle = 'Clean'
    legendLocation = 'upper right'
    plt.clf()
    ax = plt.gca()
    # Prior to consider
    #plo = np.concatenate((np.arange(-7, 7, 0.5),np.arange(7,50,2)))
    plo = np.arange(-7, 7, 0.25)
    minPe = ece(matedScores_opt, nonMatedScores_opt, plo)
    refPe = ece(np.array([0]),np.array([0]),plo)
    # or for ref self.defECE = (sigmoid(self.plo) * -log(sigmoid(self.plo)) + sigmoid(-self.plo) * -log(sigmoid(-self.plo))) / log(2)
    l3, = plt.plot(plo, refPe, label='$\mathrm{ECE}^{ref}$', color='black',linewidth=2, linestyle=':')
    l2, = plt.plot(plo, minPe, label='$\mathrm{ECE}$', color=colors[0],linewidth=2)
    #l1, = plt.plot(plo, pe, label='$\mathrm{P}_{e}$', color=colors[3],linewidth=2,linestyle='--')
    #leer = plt.plot([min(plo), max(plo)], [eer, eer], label='EER', color='black',linewidth=1,linestyle='-.')
    # Information of the figure
    ax.set_ylabel("ECE (bits)")
    ax.set_xlabel("logit prior")
    ax.set_title("$\mathrm{D}_{\mathrm{ECE}}$ = %.2f, $max_{|llr|}$ = %.2f, %s" % (dece,max_abs_LLR,cat_tag),  y = 1.02)
    ax.legend(loc = legendLocation)
    # Saving Figure (the replacements of extentions are optional)
    outname= output_file.replace('.pdf', '').replace('.png', '').replace('.csv', '').replace('.txt', '')
    plt.savefig(outname + ".pdf", format="pdf")
    plt.savefig(outname + ".png", format="png")