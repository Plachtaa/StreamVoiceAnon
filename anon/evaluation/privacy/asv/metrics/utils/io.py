import numpy as np
import pandas as pd


def read_targets_and_nontargets(score_file, key_file):
    scores = pd.read_csv(score_file, sep=' ', header=None)
    scores = scores.pivot_table(index=0, columns=1, values=2)

    keys = pd.read_csv(key_file, sep=' ', header=None)
    keys = keys.replace('nontarget', False).replace('target', True)
    keys = keys.pivot_table(index=0, columns=1, values=2)

    targets = scores.values[keys.values == 1.0]  # mated scores
    nontargets = scores.values[keys.values == 0.0]  # non-mated scores
    return targets, nontargets


def writeScores(mated_scores, non_mated_scores, output_file):
    """Writes scores in a single file
    One line per score in the form of: "<score_value (float)> <key (1 or 0)>"
    (1 is for mated and 0 is for non-mated)

    Parameters
    ----------
    mated_scores : Array_like
        list of scores associated to mated pairs
    non_mated_scores : Array_like
        list of scores associated to non-mated pairs
    output_file : String
        Path to output file.
    """
    keys = np.append(np.zeros(len(non_mated_scores)), np.ones(len(mated_scores)))
    scores = np.append(non_mated_scores, mated_scores)
    sortedScores = sorted(zip(scores,keys), key=lambda pair: pair[0])
    with open(output_file, 'w') as out_f:
        for i in range(len(sortedScores)):
            score = sortedScores[i][0]
            key = sortedScores[i][1]
            out_f.write("{0} {1}\n".format(score,int(key)))


def readScoresSingleFile(input_file):
    """Read scores from a single file
    One line per score in the form of: "<score_value (float)> <key (1 or 0)>"
    (1 is for mated and 0 is for non-mated)

    Parameters
    ----------
    input_file : String
        Path to the socre file.

    Returns
    -------
    matedScores : Array_like
        list of scores associated to mated pairs
    nonMatedScores : Array_like
        list of scores associated to non-mated pairs
    """
    df = pd.read_csv(input_file, sep=' ', header=None)
    matedScores = df[df[1]==1][0].values
    nonMatedScores = df[df[1]==0][0].values
    return matedScores, nonMatedScores


def my_split(s, seps):
    """Splits a string using multiple separators

    Parameters
    ----------
    s : String
        String to split.
    seps : Array_like
        List of separators
    Returns
    -------
    res : list
        list of tokens from splitting the input string
    """
    res = [s]
    for sep in seps:
        s, res = res, []
        for seq in s:
            res += seq.split(sep)
    return res


def readScoresKaldSpkv(input_file):
    """Read score-file from the kaldi speaker verification protocol

    Parameters
    ----------
    input_file : String
        Path to the socre file.

    Returns
    -------
    matedScores : Array_like
        list of scores associated to mated pairs
    nonMatedScores : Array_like
        list of scores associated to non-mated pairs
    """
    # workarround the kaldi codification of utterance informaiton
    def extract_info_from_scp_key(key):
        tokens = my_split(str(key), '-_')
        if len(tokens) == 7:
            targetId = tokens[3]
            userId = tokens[4]
            chapterId = tokens[5]
            uttId = tokens[6].replace(' ', '')
        elif len(tokens) == 4:
            userId = tokens[0]
            targetId = tokens[0]
            chapterId = tokens[1]
            uttId = tokens[2] + '-' + tokens[3].replace(' ', '')
        elif len(tokens) == 3:
            userId = tokens[0]
            targetId = tokens[0]
            chapterId = tokens[1]
            uttId = tokens[2].replace(' ', '')
        elif len(tokens) == 1:
            userId = tokens[0]
            targetId = tokens[0]
            chapterId = "00000"
            uttId = "0000"
        return userId, "{0}-{1}-{2}".format(targetId,chapterId,uttId)
        #return userId, "{targetId}-{chapterId}-{uttId}".format(targetId,chapterId,uttId)
    df = pd.read_csv(input_file, header=None,dtype={'0':'str','1':'str'}, delimiter=r"\s+", engine='python')
    keys = df.apply(lambda row: extract_info_from_scp_key(row[0])[0] == extract_info_from_scp_key(row[1])[0], axis=1)
    matedScores = df[2].values[keys == True]
    nonMatedScores = df[2].values[keys == False]
    return matedScores, nonMatedScores