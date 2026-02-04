#!/bin/bash

set -e

source env.sh

### Variables

# Select the anonymization pipeline
if [ -n "$1" ]; then
  anon_config=$1
else
  anon_config=configs/anon_stream_voice_anon.yaml
fi
echo "Using config: $anon_config"

force_compute=
force_compute='--force_compute True'

# JSON to modify run_evaluation(s) configs, see below
eval_overwrite="{"

### Anonymization + Evaluation:

# find the $anon_suffix (data/dataset_$anon_suffix) = to where the anonymization produces the data files
anon_suffix=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('${anon_config}'); print(load_hyperpyyaml(f, None).get('anon_suffix', ''))")
if [[ $anon_suffix ]]; then
  eval_overwrite="$eval_overwrite \"anon_data_suffix\": \"$anon_suffix\"}"
fi

# Generate anonymized audio (libri dev+test set & IEMOCAP dev+test set & libri-360h)
python run_anonymization.py --config ${anon_config} ${force_compute}

# Perform libri dev+test & IEMOCAP dev+test pre evaluation using pretrained ASR/ASV/SER models
python run_evaluation.py --config $(dirname ${anon_config})/eval_pre.yaml --overwrite "${eval_overwrite}" ${force_compute}

# Train post ASV using anonymized libri-360 and perform libri dev+test post evaluation
# ASV training takes ~2hours
python run_evaluation.py --config $(dirname ${anon_config})/eval_post.yaml --overwrite "${eval_overwrite}" ${force_compute}

# Merge results
results_summary_path_orig=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('$(dirname ${anon_config})/eval_pre.yaml'); print(load_hyperpyyaml(f, ${eval_overwrite}).get('results_summary_path', ''))")
results_summary_path_anon=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('$(dirname ${anon_config})/eval_post.yaml'); print(load_hyperpyyaml(f, ${eval_overwrite}).get('results_summary_path', ''))")
[[ "$results_summary_path_anon" == *"_test_tool"* ]] && exit 0

results_exp=exp/results_summary
mkdir -p ${results_exp}
{ cat "${results_summary_path_orig}"; echo; cat "${results_summary_path_anon}"; } > "${results_exp}/result_for_rank${anon_suffix}"
zip ${results_exp}/result_for_submission${anon_suffix}.zip -r exp/asr/*${anon_suffix} exp/asr/*${anon_suffix}.csv exp/ser/*${anon_suffix}.csv exp/results_summary/*${anon_suffix}* exp/asv_orig/*${anon_suffix} exp/asv_orig/*${anon_suffix}.csv exp/asv_anon${anon_suffix} > /dev/null
