FROM continuumio/miniconda3

# Set up conda environment
WORKDIR app

COPY docker/app/environment.yml .

RUN conda env create --file=environment.yml

# Install mwdata
COPY mwdata ./mwdata

COPY setup.py .

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate mw-data_describe && \
    pip install .

# Set up notebook workspace
COPY notebooks ./notebooks

EXPOSE 8888

COPY docker/app/conda.sh .
RUN chmod +x ./conda.sh
ENTRYPOINT ["./conda.sh"]