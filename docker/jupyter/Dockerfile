FROM continuumio/miniconda3

# Set up conda environment
WORKDIR app

COPY docker/jupyter/environment.yml .

RUN conda env create --file=environment.yml

# Install data_describe
COPY data_describe ./data_describe

COPY setup.py .

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate data-describe && \
    pip install .

# Set up notebook workspace
COPY examples ./examples

EXPOSE 8888

COPY docker/app/conda.sh .
RUN chmod +x ./conda.sh
ENTRYPOINT ["./conda.sh"]