FROM gcr.io/deeplearning-platform-release/base-cpu

# Set up conda environment
WORKDIR app

COPY docker/aiplatform/environment.yml .

RUN conda env create --file=environment.yml

# Install Plotly extensions
RUN jupyter labextension install jupyterlab-plotly@4.10.0 --no-build && \
    jupyter lab build --minimize=False

# Install data_describe
COPY data_describe ./data_describe

COPY setup.py .

RUN /bin/bash -c ". activate data-describe && \
    pip install . && \
    python -m ipykernel install --name=data-describe --display-name='data-describe'"

# Set up notebook workspace
WORKDIR /home/jupyter
COPY examples ./examples