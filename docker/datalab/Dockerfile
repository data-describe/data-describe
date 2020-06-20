FROM gcr.io/cloud-datalab/datalab:latest

# Set up conda environment
WORKDIR app

COPY docker/datalab/environment.yml .

RUN conda env create --file=environment.yml

# Install data_describe
COPY data_describe ./data_describe

COPY setup.py .

RUN source activate datadescribe && \
    pip install .

# Set up notebook workspace
COPY notebooks /content/datalab/docs/datadescribe/

# Extend functionality to add Data Describe kernel
RUN cp /datalab/run.sh /datalab/base-run.sh
# Custom run script
ADD docker/datalab/run.sh /datalab/run.sh
RUN chmod +x /datalab/run.sh