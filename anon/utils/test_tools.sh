#!/bin/bash

set -e

source env.sh

# Add your system here for fast testing
configs_to_test=("anon_yours.yaml")
configs_to_test=("anon_template.yaml" "anon_mcadams.yaml" "anon_asrbn.yaml" "anon_nac.yaml" "anon_sttts.yaml")

\rm data/*test_tool_* -rf || true

[ ! -d ./data/train-clean-360_test_tool ] && wget https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/data.zip/data_test_tools.zip && unzip data_test_tools.zip && rm data_test_tools.zip

[ -d ./test_tools_configs ] && yes | rm -rf ./test_tools_configs
mkdir -p ./test_tools_configs

for eval in configs/eval*; do
  cp $eval ./test_tools_configs
  suffix=$(python3 -c "
from hyperpyyaml import load_hyperpyyaml, dump_hyperpyyaml
f = open('./test_tools_configs/$(basename $eval)')
config = load_hyperpyyaml(f, None)
test_tools_dataset = [{'name': 'IEMOCAP_test', 'data': 'IEMOCAP_test_test_tool'}, {'name': 'libri_test', 'data': 'libri_test', 'enrolls': ['_enrolls'], 'trials': ['_trials_f', '_trials_m']}]
config['datasets'] = test_tools_dataset
with open('./test_tools_configs/$(basename $eval)', 'w') as f:
    dump_hyperpyyaml(config, f)
")
sed -i 's/exp\/results_summary/exp\/test_tool_results/g' ./test_tools_configs/$(basename $eval)
done
sed -i 's/train_data_name.*/train_data_name: !ref train-clean-360_test_tool<anon_data_suffix>/g' ./test_tools_configs/eval_post.yaml
sed -i -E 's/(.*:)(.*)(mcadams)(.*)/\1 !ref\2test_tool<anon_data_suffix>\4/g' ./test_tools_configs/eval_post.yaml
sed -i 's/train_data_dir.*/train_data_dir: !ref <data_dir>\/<train_data_name>/g' ./test_tools_configs/eval_post.yaml
sed -i 's/epochs:.*/epochs: 2/g' ./test_tools_configs/eval_post.yaml

for config in ${configs_to_test[@]} ; do
  python3 -c "
from hyperpyyaml import load_hyperpyyaml, dump_hyperpyyaml
f = open('./configs/${config}')
config = load_hyperpyyaml(f, None)
test_tools_dataset = [{'name': 'IEMOCAP_test', 'data': 'IEMOCAP_test_test_tool'}, {'name': 'libri_test', 'data': 'libri_test', 'enrolls': ['_enrolls'], 'trials': ['_trials_f', '_trials_m']}, {'name': 'train-clean-360_test_tool', 'data': 'train-clean-360_test_tool'}]
config['datasets'] = test_tools_dataset
with open('./test_tools_configs/${config}', 'w') as f:
    dump_hyperpyyaml(config, f)
"
  sed -i 's/IEMOCAP_test$/IEMOCAP_test_test_tool/g' ./test_tools_configs/${config}
  sed -i 's/train-clean-360$/train-clean-360_test_tool/g' ./test_tools_configs/${config}

  export VPC_TEST_TOOLS=True
  echo "================="
  echo " $config "
  echo "================="
  ./02_run.sh ./test_tools_configs/${config}
done

  echo "============================="
  echo " Test completed with success "
  echo "============================="

\rm data/*test_tool_* -rf || true
