FROM gcr.io/deeplearning-platform-release/base-cpu

# Set up conda environment
WORKDIR app

COPY docker/aiplatform/environment.yml .

RUN conda env create --file=environment.yml

# Install Plotly extensions
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1 --no-build && \
    jupyter labextension install jupyterlab-plotly@1.5.4 --no-build && \
    jupyter labextension install plotlywidget@1.5.4 --no-build && \
    jupyter lab build --minimize=False

# Install data_describe
COPY data_describe ./data_describe

COPY setup.py .

RUN /bin/bash -c ". activate datadescribe && \
    pip install . && \
    python -m ipykernel install --name=datadescribe --display-name='data-describe'"

# Set up notebook workspace
WORKDIR home
RUN mkdir ./docs
COPY notebooks ./docs/datadescribe/