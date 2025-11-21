import pandas as pd
from collections import defaultdict


def convert_results(in_file, out_file):
    asv_df = _convert_orig_results_asv(in_file=in_file)
    asr_df = _convert_orig_results_asr(in_file=in_file)
    deid_pitch_df = _convert_orig_results_deid_pitch(in_file=in_file)
    with pd.ExcelWriter(out_file) as writer:
        asv_df.to_excel(writer, 'ASV', index=False)
        asr_df.to_excel(writer, 'ASR', index=False)
        deid_pitch_df.to_excel(writer, 'DeID-Pitch', index=False)


def _convert_orig_results_asv(in_file):
    results = {'dataset': [], 'split': [], 'gender': [], 'enrollment': [], 'trial': [], 'EER': [], 'cllr min': [],
               'cllr act': [], 'rocch-EER': [], 'linkability': [], 'population': [], 'individual': []}

    with open(in_file, 'r') as f:
        dataset = None
        split = None
        gender = None
        enrollment = None
        trial = None
        eer = None
        cllr_min = None
        cllr_act = None
        rocch_eer = None
        linkability = None
        population = None

        for line in f:
            line = line.strip()
            if line.startswith('ASV'):
                dataset = 'libri' if 'libri' in line else 'vctk'
                split = 'dev' if 'dev' in line else 'test'
                enrollment = 'anon' if 'enrolls_anon' in line else 'original'
                trial = 'anon' if line.strip().endswith('anon') else 'original'
                if 'f_common' in line:
                    gender = 'female common'
                elif 'm_common' in line:
                    gender = 'male common'
                elif 'trials_f' in line:
                    gender = 'female'
                else:
                    gender = 'male'
            elif line.startswith('EER'):
                eer = float(line.strip('%').replace('EER: ', ''))
            elif line.startswith('Cllr'):
                line = line.replace('Cllr (min/act): ', '')
                split_line = line.split('/')
                cllr_min = float(split_line[0])
                cllr_act = float(split_line[1])
            elif line.startswith('ROCCH-EER'):
                rocch_eer = float(line.strip('%').replace('ROCCH-EER: ', ''))
            elif line.startswith('linkability'):
                linkability = float(line.replace('linkability: ', ''))
            elif line.startswith('Population'):
                population = line.replace('Population: ', '')
            elif line.startswith('Individual'):
                individual = line.replace('Individual: ', '')
                results['dataset'].append(dataset)
                results['split'].append(split)
                results['gender'].append(gender)
                results['enrollment'].append(enrollment)
                results['trial'].append(trial)
                results['EER'].append(eer)
                results['cllr min'].append(cllr_min)
                results['cllr act'].append(cllr_act)
                results['rocch-EER'].append(rocch_eer)
                results['linkability'].append(linkability)
                results['population'].append(population)
                results['individual'].append(individual)

                dataset = None
                split = None
                gender = None
                enrollment = None
                trial = None
                eer = None
                cllr_min = None
                cllr_act = None
                rocch_eer = None
                linkability = None
                population = None

    return pd.DataFrame(results)


def _convert_orig_results_asr(in_file):
    results = {'dataset': [], 'split': [], 'anon': [], 'WER small': [], 'WER large': []}

    with open(in_file, 'r') as f:
        dataset = None
        split = None
        anon = None
        wer_small = None

        for line in f:
            line = line.strip()
            if line.startswith('ASR'):
                dataset = 'libri' if 'libri' in line else 'vctk'
                split = 'dev' if 'dev' in line else 'test'
                anon = 'anon' if '_anon' in line else 'original'
            elif line.startswith('%WER'):
                splitted_line = line.split()
                if '_tgsmall' in line:
                    wer_small = float(splitted_line[1])
                elif '_tglarge' in line:
                    wer_large = float(splitted_line[1])

                    results['dataset'].append(dataset)
                    results['split'].append(split)
                    results['anon'].append(anon)
                    results['WER small'].append(wer_small)
                    results['WER large'].append(wer_large)

                    dataset = None
                    split = None
                    anon = None
                    wer_small = None

    return pd.DataFrame(results)


def _convert_orig_results_deid_pitch(in_file):
    def get_split_and_gender(line):
        split = 'dev' if 'dev' in line else 'test'
        if 'f_common' in line:
            gender = 'female common'
        elif 'm_common' in line:
            gender = 'male common'
        elif 'trials_f' in line:
            gender = 'female'
        else:
            gender = 'male'
        return split, gender

    def get_pitch_correlation(line):
        splitted_line = line.split()
        tag = splitted_line[0]
        mean = float(splitted_line[-2].replace('mean=', ''))
        std = float(splitted_line[-1].replace('std=', ''))
        return tag, mean, std

    results = defaultdict(lambda: {'dataset': None, 'split': None, 'gender': None, 'DeID': None,
                                   'Pitch corr mean': None, 'Pitch corr std': None})

    with open(in_file, 'r') as f:
        tag = None
        dataset = None
        split = None
        gender = None
        deid = None

        for line in f:
            line = line.strip()
            if line.startswith('libri'):
                if 'Pitch_correlation' in line:
                    tag, mean_pitch, std_pitch = get_pitch_correlation(line)
                    results[tag]['Pitch corr mean'] = mean_pitch
                    results[tag]['Pitch corr std'] = std_pitch
                else:
                    tag = line
                    dataset = 'libri'
                    split, gender = get_split_and_gender(line)
            elif line.startswith('vctk'):
                if 'Pitch_correlation' in line:
                    tag, mean_pitch, std_pitch = get_pitch_correlation(line)
                    results[tag]['Pitch corr mean'] = mean_pitch
                    results[tag]['Pitch corr std'] = std_pitch
                else:
                    tag = line
                    dataset = 'vctk'
                    split, gender = get_split_and_gender(line)
            elif line.startswith('De-Identification'):
                deid = float(line.replace('De-Identification : ', ''))

                results[tag]['dataset'] = dataset
                results[tag]['split'] = split
                results[tag]['gender'] = gender
                results[tag]['DeID'] = deid

                tag = None
                dataset = None
                split = None
                gender = None
                deid = None

    df = pd.DataFrame(results).T
    return df


if __name__ == '__main__':
    filepath = 'results.txt'
    outfile = 'results.xlsx'
    convert_results(filepath, outfile)
