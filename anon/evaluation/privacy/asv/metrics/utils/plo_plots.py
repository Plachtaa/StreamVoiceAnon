import numpy as np
from tikzplotlib import save as tikz_save
from os.path import split as path_split
from copy import deepcopy as copy
import matplotlib.pyplot as mpl
from uuid import uuid4
from os import sep
from scipy.special import expit

from ..zebra import ece, fast_actDCF
from ..cllr import cllr
from ..helpers import rocch_pava, optimal_llr_from_Popt, optimal_llr



__author__ = "Andreas Nautsch"
__email__ = "nautsch@eurecom.fr"
__coauthor__ = ["Jose Patino", "Natalia Tomashenko", "Junichi Yamagishi", "Paul-Gauthier Noé", "Jean-François Bonastre", "Massimiliano Todisco", "Nicholas Evans"]
__credits__ = ["Niko Brummer", "Daniel Ramos", "Edward de Villiers", "Anthony Larcher"]
__license__ = "LGPLv3"


class PriorLogOddsPlots(object):
    """
    Class handling ZEBRA, ECE & DCF visualizations as well as metrics: ECE, DCF, Cllr (actual/minimum) as well as the ROCCH-EER.

    Plot exports are possible to: LaTeX, PNG & PDF
    """

    def __init__(self, classA_scores=None, classB_scores=None, normalize=False, plo=np.linspace(-10, 10, 201)):
        """
        :param classA_scores: scores of class A; default: None (scores must be set later)
        :param classB_scores: scores of class B; default: None (scores must be set later)
        :param normalize: flag to normalize y-values by the default performance of each depending x-axis prior - a horizontal line at y=1 resembles the default performance; default: False
        :param plo: array of prior log-odds; the x-axis values for which ECE & DCF values are computed; default: numpy.linspace(-10, 10, 201)
        """
        self.normalize = normalize
        self.plo = plo

        self.Ptar = expit(self.plo)
        self.Pnon = expit(-self.plo)
        self.Pdcf = np.vstack([[self.Ptar, self.Pnon]]).T

        self.defDCF = np.minimum(self.Ptar, self.Pnon)
        self.defECE = (expit(self.plo) * -np.log(expit(self.plo)) + expit(-self.plo) * -np.log(expit(
            -self.plo))) \
                      / np.log(2)

        uuid = str(uuid4())
        self.dcf_fig = 'DCF' + '-' + uuid
        self.ece_fig = 'ECE' + '-' + uuid

        # for file names only
        self.dcf_label = 'DCF'
        self.ece_label = 'ECE'

        self.legend_DCF = ['default']
        self.legend_ECE = ['default']

        if classA_scores is not None and classB_scores is not None:
            self.set_system(classA_scores=classA_scores, classB_scores=classB_scores)

    def add_legend_entry(self, entry):
        """
        adds entries to internal plots, both ECE and DCF (expected to be called alongside the curve plotting); or in depending sequence afterwards

        :param entry: profile label
        """
        self.legend_DCF.append(entry)
        self.legend_ECE.append(entry)

    def set_system(self, classA_scores, classB_scores):
        """
        Sets a new system (of class A & B scores) to the framework's assessment, which is carried out here as well.

        :param classA_scores: scores of class A
        :param classB_scores: scores of class B
        """
        self.classA_scores = classA_scores
        self.classB_scores = classB_scores

        pmiss, pfa, Popt, perturb = rocch_pava(self.classA_scores, self.classB_scores, laplace=False)
        self.Pmiss = pmiss
        self.Pfa = pfa

        classA_llr, classB_llr = optimal_llr_from_Popt(Popt, perturb, Ntar=len(self.classA_scores), Nnon=len(self.classB_scores))
        self.classA_llr = classA_llr
        self.classB_llr = classB_llr

        classA_llr_laplace, classB_llr_laplace, rocch_eer = optimal_llr(tar=classA_scores, non=classB_scores, laplace=True, compute_eer=True)
        self.classA_llr_laplace = classA_llr_laplace
        self.classB_llr_laplace = classB_llr_laplace
        self.rocch_eer = rocch_eer

        cdet = self.Pdcf @ np.vstack((self.Pmiss, self.Pfa))
        self.minDCF = cdet.min(axis=1)
        self.actDCF = fast_actDCF(self.classA_scores, self.classB_scores, self.plo)
        self.eer = self.minDCF.max()

        self.minECE = ece(tar=self.classA_llr, non=self.classB_llr, plo=self.plo)
        self.actECE = ece(tar=self.classA_scores, non=self.classB_scores, plo=self.plo)

    def save(self, filename, plot_type, ext='png', dpi=None, width='120pt', height='80pt'):
        """
        Export of ECE or DCF plots to a file location

        :param filename: path and name of file to export a plot to
        :param plot_type: which plot to save (DCF or ECE)
        :param ext: file extension [tex, pdf, png]; default: png
        :param dpi: for LaTeX export only; default: None
        :param width: for LaTeX export only; default: 120pt
        :param height: for LaTeX export only; default: 80pt
        """
        # dpi, witdh, height only for LaTeX

        assert plot_type in ['ECE', 'DCF']
        assert ext in ['png', 'pdf', 'tex']

        if plot_type == 'DCF':
            figure = self.dcf_fig
        elif plot_type == 'ECE':
            figure = self.ece_fig
        else:
            figure = None

        # in case of manual plot labeling
        if (plot_type == 'DCF') and (self.dcf_label != plot_type):
            plot_type = self.dcf_label
        elif (plot_type == 'ECE') and (self.ece_label != plot_type):
            plot_type = self.ece_label

        # set the proper plot type for use in file name
        if self.normalize:
            if plot_type == 'DCF':
                plot_type = 'NBER'
            elif plot_type == 'ECE':
                plot_type = 'NECE'
        else:
            if plot_type == 'DCF':
                plot_type = 'APE'

        file_path, file_name = path_split(filename)
        if len(file_path) > 0:
            file_path += sep
        fname = file_path + plot_type + '-' + file_name + '.' + ext

        if ext in ['png', 'pdf']:
            mpl.figure(figure)
            mpl.savefig(fname)
        elif ext == 'tex':
            self.__save_as_tikzpgf__(outfilename=fname, fig=figure, dpi=dpi, width=width, height=height)
        else:
            raise ValueError('unknown save format')

    def __save_as_tikzpgf__(self, outfilename,
                            fig=None,
                            dpi=None,
                            width='120pt',
                            height='80pt',
                            standalone=False,
                            extra_tikzpicture_parameters=['font=\\scriptsize'],
                            extra_axis_parameters=['legend pos=outer north east','legend columns=1']):
        # see: https://codeocean.com/algorithm/154591c8-9d3f-47eb-b656-3aff245fd5c1/code
        # minor edits for legend positioning

        def replace_tick_label_notation(tick_textpos):
            tick_label = tick_textpos.get_text()
            if 'e' in tick_label:
                tick_label = int(tick_label.replace('1e', '')) - 2
                tick_textpos.set_text('%f' % (10**(int(tick_label)-2)))

        if dpi is not None:
            mpl.figure(fig).set_dpi(dpi)

        if fig is None:
            fig = self.ece_fig

        ax = mpl.figure(fig).get_axes()[0]
        ax.set_title('')

        if '\\' in ax.get_xlabel():
            ax.set_xlabel('$' + ax.get_xlabel() + '$')

        if '\\' in ax.get_ylabel():
            ax.set_ylabel('$' + ax.get_ylabel() + '$')

        for tick_textpos in ax.get_xmajorticklabels():
            replace_tick_label_notation(tick_textpos)
        for tick_textpos in ax.get_ymajorticklabels():
            replace_tick_label_notation(tick_textpos)

        tikz_save(outfilename,
                  figurewidth=width,
                  figureheight=height,
                  standalone=standalone,
                  extra_tikzpicture_parameters=extra_tikzpicture_parameters,
                  extra_axis_parameters=extra_axis_parameters)

    def show_legend(self, plot_type, legend_loc='best'):
        """
        Shows legend for ECE or DCF plot

        :param plot_type: ECE or DCF
        :param legend_loc: legend position, see: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.legend.html or partial function taking a figure handle and a legend
        """
        assert plot_type in ['ECE', 'DCF']
        assert (type(legend_loc) is str) or callable(legend_loc)

        if plot_type == 'DCF':
            figure = self.dcf_fig
            legend = self.legend_DCF
        elif plot_type == 'ECE':
            figure = self.ece_fig
            legend = self.legend_ECE
        else:
            figure = None
            legend = None

        mpl.figure(figure)
        if type(legend_loc) is str:
            mpl.legend(legend, loc=legend_loc)
        else:
            legend_loc(figure=figure, legend=legend)

    def plot_dcf(self, plot_err=False, flip_xaxis=False, color_min='k', style_min='--', color_act='g', style_act='-'):
        """
        Plotting a DCF profile; min/act profiles can be made optional by setting their color: None

        :param plot_err: flag to plot the min max DCF derived EER; default: False
        :param flip_xaxis: flag to flip the x-axis to be aligned with threshold/score histograms; default: False
        :param color_min: color of minDCF profile; default: black
        :param style_min: line style of minDCF profile; default: dashed
        :param color_act: color of actDCF profile; default: green
        :param style_act: line style of actDCF profile; default: solid
        """
        defDCF = copy(self.defDCF)
        minDCF = copy(self.minDCF)
        actDCF = copy(self.actDCF)
        eer = np.repeat(self.eer, len(self.plo))

        if self.normalize:
            minDCF /= defDCF
            actDCF /= defDCF
            eer /= defDCF
            defDCF /= defDCF

        mpl.figure(self.dcf_fig)

        if flip_xaxis:
            # plo remains as is, we just flip the other arrays
            mpl.xlabel('LLR threshold')
            defDCF = np.flipud(defDCF)
            minDCF = np.flipud(minDCF)
            actDCF = np.flipud(actDCF)
            eer = np.flipud(eer)
        else:
            mpl.xlabel('logit(\\tilde\\pi)')

        if not self.normalize:
            mpl.ylabel('DCF')
        else:
            mpl.ylabel('NBER')

        if len(self.legend_DCF) == 1:
            mpl.plot(self.plo, defDCF, color='k', linewidth=2)

        if color_min is not None:
            mpl.plot(self.plo, minDCF, color=color_min, linestyle=style_min, linewidth=2)

        if color_act is not None:
            mpl.plot(self.plo, actDCF, color=color_act, linestyle=style_act, linewidth=1)
            mpl.ylim([0, 1.4 * max(1, actDCF.min())])
        else:
            mpl.ylim([0, 1.4])

        if plot_err:
            mpl.plot(self.plo, eer, 'k:')

    def plot_ece(self, color_min='b', style_min='--', color_act='r', style_act='-'):
        """
        Plotting a ECE profile; min/act profiles can be made optional by setting their color: None

        :param color_min: color of minECE profile; default: blue
        :param style_min: line style of minECE profile; default: dashed
        :param color_act: color of actECE profile; default: red
        :param style_act: line style of actECE profile; default: solid
        """
        defECE = copy(self.defECE)
        minECE = copy(self.minECE)
        actECE = copy(self.actECE)

        if self.normalize:
            minECE /= defECE
            actECE /= defECE
            defECE /= defECE

        mpl.figure(self.ece_fig)

        mpl.xlabel('logit(\\pi)')

        if not self.normalize:
            mpl.ylabel('ECE')
        else:
            mpl.ylabel('NECE')

        if len(self.legend_ECE) == 1:
            mpl.plot(self.plo, defECE, color='k', linewidth=2)

        if color_min is not None:
            mpl.plot(self.plo, minECE, color=color_min, linestyle=style_min, linewidth=2)

        if color_act is not None:
            mpl.plot(self.plo, actECE, color=color_act, linestyle=style_act, linewidth=1)
            mpl.ylim([0, 1.4 * max(1, actECE.min())])
        else:
            mpl.ylim([0, 1.4])

    def get_delta_DCF(self):
        """
        Privacy attributed summary of DCF characteristic: one minus the analytical area under DCF profile; deprecated for the purpose of privacy assessment

        :return: 1-Cllr
        """
        return 1 - cllr(self.classA_llr, self.classB_llr)

    def get_delta_ECE(self):
        """
        Privacy attributed summary of ECE characteristic (population metric in our ZEBRA paper)

        :return: integral between default and minimum ECE profiles
        """

        def int_ece(x, epsilon=1e-6):
            """
            Z(X) = avg( [(x-3)*(x-1) + 2*log(x)] / [4 * (x-1)^2] )  # x as LR
                 = avg( 0.25 - 1/[2*(x-1)] + log(x)/[2*(x-1)^2] )  # a = log(x)
                 = 0.25 + 0.5 * avg( - 1/[(exp(a) - 1)] + a/[(exp(a)-1)^2] )  # b = exp(a)-1
                 = 0.25 + 0.5 * avg( (a-b))/b^2 )
            """
            idx = (~np.isinf(x)) & (abs(x) > epsilon)
            contrib = np.zeros(len(x))  # for +inf, the contribution is 0.25; the later on constant term
            xx = x[idx]
            LRm1 = np.exp(xx) - 1
            contrib[idx] = (xx - LRm1) / LRm1 ** 2
            # if x == 0 or if x < epsilon
            # numerical issue of exp() function for small values around zero, thus also hardcoded value
            contrib[(abs(x) <= epsilon)] = -0.5  # Z(0) = 0 = 0.25 + (-0.5)/2
            return 0.25 + contrib.mean() / 2

        int_diff_ece = int_ece(self.classA_llr) + int_ece(-self.classB_llr)
        return int_diff_ece / np.log(2)

