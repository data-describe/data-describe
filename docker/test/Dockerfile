FROM continuumio/miniconda3

COPY docker/test/environment.yml /

RUN conda env create --file=environment.yml

COPY . /

CMD . /opt/conda/etc/profile.d/conda.sh && \
    conda activate mw-data_describe && \
    pytest --durations=0 --cov=data_describe --cov-report term-missing --junitxml=junit.xml --cov-report xml tests