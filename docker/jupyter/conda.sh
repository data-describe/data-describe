#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh
conda activate data-describe
jupyter kernelspec remove python3 -f
python -m ipykernel install --user --name data-describe
jupyter lab --port 8888 --ip 0.0.0.0 --allow-root --no-browser
