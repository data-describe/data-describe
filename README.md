# MW Data Describe
Expediting data discovery and understanding

## AI Platform Notebook
Using the GUI or CLI, you can stand up an instance of AI Platform Notebooks with Data Describe by using the custom container at `gcr.io/mwpmltr/mw-data_describe`

### Building the AI Platform Image
From the project root directory, run (replace `<version>` with the current version):
```console
docker-compose build aiplatform
docker-compose tag mw-data_describe_aiplatform gcr.io/mwpmltr/mw-data_describe:<version>
docker push gcr.io/mwpmltr/mw-data_describe:<version>
```

### Quick Start
Some example notebooks are provided for reference. The data are not included.

To run a new notebook, use the DataDescribe Python kernel. You can import the Data Describe module using:
```python
import mwdata as mw
```

## Local Installation
This section covers manual, local installation of the `mwdata` Python package.

### Environment Prerequisites
#### Using Conda
It is generally recommended to use Conda to install the dependencies due to build complications. Run the following from Anaconda Prompt to create a new conda environment with the package dependencies.
```conda
conda env create -f docker/app/environment.yml -n mw-data_describe
conda activate mw-data_describe
```
#### Using pip in venv / virtualenv
If not using conda, you may have to address specific build dependencies as listed in `requirements.txt`:
```console
# Troubleshoot specific build dependencies from requirements list
# You may have to download a .whl file or install build tools for your OS architecture
pip install -f requirements.txt
```

### Installing the Package
To install the Data Describe python package to your local environment, use pip in the environment:
```console
# Use this to install the latest version from the master branch. You will have to enter your Bitbucket credentials.
pip install git+https://bitbucket.org/mavenwave/mw-data_describe.git

# Or, if you have cloned the repository, the following can be run from the currently checked out branch
pip install .
```

### Plotly Requirement
Plotly is used for many visualizations. The JupyterLab extension must be installed to view these charts: https://plot.ly/python/getting-started/
