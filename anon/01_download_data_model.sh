#!/bin/bash

set -e

source env.sh

# librispeech_corpus=PATH_TO_Librispeech
# iemocap_corpus=PATH_TO_IEMOCAP

for data_set in libri_dev libri_test; do
    dir=data/$data_set
    if [ ! -f $dir/wav.scp ] ; then
        if [ -z $password ]; then
          echo "Enter password provided by the organisers (check README.md registration):"
          read -s password
          echo
        fi
        [ -d $dir ] && rm -r $dir
        if [ ! -f corpora/$data_set.tar.gz ]; then
            mkdir -p corpora
            cd corpora
            sshpass -p "$password"  sftp -oStrictHostKeyChecking=no getdata@voiceprivacychallenge.univ-avignon.fr <<EOF
    cd /challengedata/corpora
    get $data_set.tar.gz
    bye
EOF
  cd -
  fi
  echo "  Unpacking $data_set data set..."
  tar -xf corpora/$data_set.tar.gz || exit 1
  [ ! -f $dir/text ] && echo "File $dir/text does not exist" && exit 1
  cut -d' ' -f1 $dir/text > $dir/text1
  cut -d' ' -f2- $dir/text | sed -r 's/,|!|\?|\./ /g' | sed -r 's/ +/ /g' | awk '{print toupper($0)}' > $dir/text2
  paste -d' ' $dir/text1 $dir/text2 > $dir/text
  rm $dir/text1 $dir/text2
fi

done


check=corpora/LibriSpeech/train-clean-360
if [ ! -d $check ]; then
    if [ ! -z $librispeech_corpus ]; then
        if [ -d $librispeech_corpus/train-clean-360 ]; then
            [ -d corpora/LibriSpeech ] && rm corpora/LibriSpeech
            echo "Linking '$librispeech_corpus' to 'corpora'"
            mkdir -p corpora
            ln -s $librispeech_corpus corpora
        else
          echo "librispeech_corpus is defined to '$librispeech_corpus', but '$librispeech_corpus/train-clean-360' does not exists."
          echo "Either remove the librispeech_corpus variable from the $0 script to download the dataset or modify it to the correct target."
          exit 1
        fi
    fi
fi
#Download LibriSpeech-360
if [ ! -d $check ]; then
    echo "Download train-clean-360..."
    mkdir -p corpora
    cd corpora
    if [ ! -f train-clean-360.tar.gz ] ; then
        echo "Download train-clean-360..."
        wget --no-check-certificate https://www.openslr.org/resources/12/train-clean-360.tar.gz
    fi
    echo "Unpacking train-clean-360"
    tar -xzf train-clean-360.tar.gz
    cd ../
fi

check_data=data/libri_dev_enrolls
if [ ! -d $check_data ]; then
    if  [ ! -f .data.zip ]; then
        echo "Download VPC kaldi format datadir..."
        wget https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/data.zip/data.zip
        mv data.zip .data.zip
    fi
    echo "Unpacking data"
    unzip .data.zip
fi

for model in asv_orig ser asr; do
    if [ ! -d "exp/$model" ]; then
        if [ ! -f .${model}.zip ]; then
            echo "Download pretrained $model models pre-trained..."
            wget https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/pre_model.zip/${model}.zip
            mv ${model}.zip .${model}.zip
        fi
        echo "Unpacking pretrained evaluation models"
        unzip .${model}.zip
    fi
done

if [ ! -d "data/IEMOCAP/wav/Session1" ]; then
    if [ ! -z $iemocap_corpus ]; then
        if [ -d $iemocap_corpus/Session1 ]; then
            echo "Linking '$iemocap_corpus' to 'data/IEMOCAP/wav'"
            ln -s $iemocap_corpus data/IEMOCAP/wav
        else
          echo "iemocap_corpus is defined to '$iemocap_corpus', but '$iemocap_corpus/Session1' does not exists."
          echo "Please fix your path to iemocap_corpus in the $0 script."
          exit 1
        fi
    fi
fi

# IEMOCAP_full_release
if [ ! -d "data/IEMOCAP/wav/Session1" ]; then
    mkdir -p ./data/IEMOCAP/
    cat << EOF
==============================================================================
    Plase download or link the IEMOCAP corpus to './data/IEMOCAP/wav'
      - Download IEMOCAP from its web-page (license agreement is required)
          - https://sail.usc.edu/iemocap/
      - Link
          - ln -s YOUR_PATH data/IEMOCAP/wav/
==============================================================================
EOF
exit 1
fi
