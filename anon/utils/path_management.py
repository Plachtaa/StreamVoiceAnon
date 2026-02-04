from pathlib import Path
import shutil
import os
import glob
from multiprocessing import Manager

def create_clean_dir(dir_name:Path, force:bool = True):
    if dir_name.exists() and force:
        remove_contents_in_dir(dir_name)
    else:
        dir_name.mkdir(exist_ok=True, parents=True)


def copy_data_dir(dataset_path, output_path):
    # Copy utt2spk wav.scp and so on, but not the directories inside (may contains clear or anonymzied *.wav)
    os.makedirs(output_path, exist_ok=True)
    for p in glob.glob(str(dataset_path / '*'), recursive=False):
        if os.path.isfile(p):
            shutil.copy(p, output_path)


def remove_contents_in_dir(dir_name:Path):
    # solution from https://stackoverflow.com/a/56151260
    for path in dir_name.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)



def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*****')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None

    try:
        cp_list_by_name = sorted([(int(ckpt.split(prefix)[-1]), ckpt) for ckpt in cp_list])
        return Path(cp_list_by_name[-1][1])
    except ValueError:
        # Handle the case where conversion to int fails
        return None


def get_datasets(config):
    datasets = {}
    data_dir = config.get('data_dir', None).expanduser() # if '~' is given in path then manually expand
    for dataset in config['datasets']:
        no_sub = True
        for subset in ['trials', 'enrolls']:
            if subset in dataset:
                for subset in dataset[subset]:
                    dataset_name = f'{dataset["data"]}{subset}'
                    datasets[dataset_name] = Path(data_dir, dataset_name)
                    no_sub = False
        if no_sub:
            dataset_name = f'{dataset["data"]}'
            datasets[dataset_name] = Path(data_dir, dataset_name)
    return datasets
