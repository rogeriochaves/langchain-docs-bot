#!/bin/bash

pip install -r requirements.txt
mkdir -p models
cd models
ls ggml-gpt4all-j-v1.3-groovy.bin || wget https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin
