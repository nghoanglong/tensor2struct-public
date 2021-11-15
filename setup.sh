#!/bin/bash

# install tensor2struct
pip install -e .

# spacy and nltk
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# data dir
export PT_DATA_DIR="${PT_DATA_DIR:-$PWD}"
export CACHE_DIR=${PT_DATA_DIR}

# cache dir
mkdir -p "$CACHE_DIR/.vector_cache"
