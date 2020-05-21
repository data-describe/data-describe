#!/bin/bash

source activate datadescribe
python -m ipykernel install --prefix=/usr/local/envs/py3env --name="DataDescribe"

source /datalab/base-run.sh
