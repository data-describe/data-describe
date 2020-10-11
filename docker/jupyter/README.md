# Jupyter Lab Docker Image

This docker image runs Jupyter Lab in a (local) Docker container, with data-describe pre-installed.

## Building the image

To build the image, run from the root directory of the repository:

```
docker-compose build jupyter
```

### Starting Jupyter Lab

Run the following command:

```
docker run -it --rm -p 8888:8888 data-describe_jupyter
```

You will see output in the terminal; copy and use the provided link in your browser.

```
# Example output
>> docker run -it --rm -p 8888:8888 data-describe_jupyter
[RemoveKernelSpec] Removed /opt/conda/envs/data-describe/share/jupyter/kernels/python3
Installed kernelspec data-describe in /root/.local/share/jupyter/kernels/data-describe
[I 01:13:56.178 LabApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 01:13:56.554 LabApp] JupyterLab extension loaded from /opt/conda/envs/data-describe/lib/python3.7/site-packages/jupyterlab
[I 01:13:56.554 LabApp] JupyterLab application directory is /opt/conda/envs/data-describe/share/jupyter/lab
[I 01:13:56.559 LabApp] Serving notebooks from local directory: /app
[I 01:13:56.559 LabApp] Jupyter Notebook 6.1.4 is running at:
[I 01:13:56.559 LabApp] http://616754ddb1be:8888/?token=558351d285edc281fbc947a12d21b1653283cc474617ecb9
[I 01:13:56.559 LabApp]  or http://127.0.0.1:8888/?token=558351d285edc281fbc947a12d21b1653283cc474617ecb9
[I 01:13:56.559 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 01:13:56.568 LabApp]

    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-17-open.html
    Or copy and paste one of these URLs:
        http://616754ddb1be:8888/?token=558351d285edc281fbc947a12d21b1653283cc474617ecb9
     or http://127.0.0.1:8888/?token=558351d285edc281fbc947a12d21b1653283cc474617ecb9
```