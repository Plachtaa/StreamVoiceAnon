import logging
import numpy as np
from matplotlib._cm import datad
import matplotlib.pyplot as mpl
from copy import deepcopy
from os import sep

from .plo_plots import PriorLogOddsPlots
from .io import read_targets_and_nontargets

logger = logging.getLogger(__name__)

__author__ = "Andreas Nautsch"
__email__ = "nautsch@eurecom.fr"
__coauthor__ = ["Jose Patino", "Natalia Tomashenko", "Junichi Yamagishi", "Paul-Gauthier Noé", "Jean-François Bonastre", "Massimiliano Todisco", "Nicholas Evans"]
__credits__ = ["Niko Brummer", "Daniel Ramos", "Edward de Villiers", "Anthony Larcher"]
__license__ = "LGPLv3"


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

# color map
cmap_tab20 = datad['tab20']['listed']


def zebra_framework(plo_plot, scr_path, key_path, label='ZEBRA profile',
                    color_min='blue', style_min='-',
                    color_act=None, style_act='-',
                    dcf_pot=False, ):
    """
    Run the ZEBRA framework for a pair of score and ground-truth files.

    :param plo_plot: the handler object, class: prior log-odds plots
    :param scr_path: path to score file
    :param key_path: path to ground-truth (key) file
    :param label: name of the current experiment under assessment
    :param color_min: line specification (color); default: None - no profile for discrimination performance
    :param style_min: line specification (color); default: solid
    :param color_act: line specification (color); default: None - no profile for discrimination and calibration performance
    :param style_act: line specification (color); default: solid
    :param dcf_pot: flag also creates DCF plots; default: False
    """
    assert isinstance(plo_plot, PriorLogOddsPlots)
    plo_plot.ece_label = 'ZEBRA'
    if len(plo_plot.legend_ECE) == 1:
        plo_plot.legend_ECE = ['perfect privacy (0, 0, 0)']

    # read the class A (classA_scores) and class B (classB_scores) scores
    classA_scores, classB_scores = read_targets_and_nontargets(score_file=scr_path, key_file=key_path)

    # set these scores for assessment
    # note: an isotonic regression is performed (can take some time)
    plo_plot.set_system(classA_scores, classB_scores)

    # compute ZEBRA metrics
    dece = plo_plot.get_delta_ECE()
    max_abs_LLR = abs(np.hstack((plo_plot.classA_llr_laplace, plo_plot.classB_llr_laplace))).max() / np.log(10)
    cat_idx = np.argwhere((cat_ranges < max_abs_LLR).sum(1) == 1).squeeze()
    cat_tag = list(categorical_tags.keys())[cat_idx]

    # string representations
    str_dece = ('%.3f' if dece >= 5e-4 else '%.e') % dece
    str_max_abs_llr = ('%.3f' if max_abs_LLR >= 5e-4 else '%.e') % max_abs_LLR

    if dece == 0:
        str_dece = '0'

    if max_abs_LLR == 0:
        str_max_abs_llr = '0'

    # print outs
    logger.info("%s" % label)
    logger.info("Population: %s bit" % str_dece)
    logger.info("Individual: %s (%s)" % (str_max_abs_llr, cat_tag))

    # Creating log-odds plots
    if color_min is not None:
        legend_entry = "%s (%s, %s, %s)" % (label, str_dece, str_max_abs_llr, cat_tag)

        # ECE
        plo_plot.plot_ece(color_min=color_min, style_min=style_min, color_act=color_act, style_act=style_act)

        # DCF
        if dcf_pot:
            plo_plot.plot_dcf(color_min=color_min, style_min=style_min, color_act=color_act, style_act=style_act)
            logger.info("1 - min Cllr: %.3f (0 is good)" % plo_plot.get_delta_DCF())

        plo_plot.add_legend_entry(legend_entry)


def export_zebra_framework_plots(plo_plot, filename, save_plot_ext=None, save_dcf=False, legend_loc='best'):
    """
    Saves created figures that are handled by the prior log-odds plot object.

    :param plo_plot: the handler object, class: prior log-odds plots
    :param filename: name of exported file
    :param save_plot_ext: format/extension of saved picture (valid: tex, pdf, png); default: None - no save
    :param save_dcf: flag to save also DCF plots; default: False
    """
    def zebra_legend(align):
        # set alignment of legend text: right for LaTeX export
        for texts in mpl.gca().get_legend().texts:
            assert isinstance(texts, mpl.Text)
            texts.set_horizontalalignment(align)

    assert isinstance(plo_plot, PriorLogOddsPlots)
    assert (save_plot_ext is None) or (save_plot_ext in ['tex', 'pdf', 'png'])
    assert (type(legend_loc) is str) or callable(legend_loc)

    align = 'left'
    if save_plot_ext == 'tex':
        align = 'right'

    if save_plot_ext is not None:
        plo_plot.show_legend(plot_type='ECE', legend_loc=legend_loc)
        zebra_legend(align=align)
        plo_plot.save(filename=filename, plot_type='ECE', ext=save_plot_ext)

        if save_dcf:
            plo_plot.show_legend(plot_type='DCF', legend_loc=legend_loc)
            zebra_legend(align=align)
            plo_plot.save(filename=filename, plot_type='DCF', ext=save_plot_ext)

def zebra_plots_sorted_legend(dece_values, zebra_objects, title_strings, filename_strings, ext='png', legend_loc='best'):
    """
    Re-creates ZEBRA plots from pre-computations but with sorted legend by DECE values.

    :param dece_values: list of lists with DECE values; main list represents different ZEBRA plots, sub-lists the profiles of in each
    :param zebra_objects: list of lists with PriorLogOddsPlots; main list represents different ZEBRA plots, sub-lists the profiles of in each
    :param title_strings: list of ZEBRA plot titles
    :param filename_strings: list of file names with paths where to store the plots
    :param ext: file extension of plot; default: png
    :param legend_loc: legend positioning
    """
    for dece_handle, zebra_handle, title_str, fname in zip(dece_values, zebra_objects, title_strings, filename_strings):
        zebra_plot = PriorLogOddsPlots()
        zebra_plot.ece_label = 'ZEBRA'
        zebra_plot.legend_ECE = ['perfect privacy (0, 0, 0)']

        sorted_idx = np.argsort(dece_handle, kind='quicksort')

        for idx in sorted_idx:
            # copy from stored zebra obj
            zebra_plot.defECE = deepcopy(zebra_handle[idx].defECE)
            zebra_plot.minECE = deepcopy(zebra_handle[idx].minECE)
            zebra_plot.actECE = deepcopy(zebra_handle[idx].actECE)
            zebra_plot.plot_ece(color_min=cmap_tab20[idx % len(cmap_tab20)], style_min='-', color_act=None)
            zebra_plot.add_legend_entry(zebra_handle[idx].legend_ECE[-1])

        mpl.title(title_str)
        fname_parts = fname.split(sep)
        if len(fname_parts) > 1:
            fname_parts[-1] = 'sorted-' + fname_parts[-1]
            fname_sorted = sep.join(fname_parts)
        else:
            fname_sorted = 'sorted-' + fname

        export_zebra_framework_plots(plo_plot=zebra_plot, filename=fname_sorted, save_plot_ext=ext, legend_loc=legend_loc)
        mpl.close(zebra_plot.ece_fig)
