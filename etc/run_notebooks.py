import glob
import json

import papermill as pm


def run_all_notebooks():
    """Run all notebooks in the example directory."""
    for notebook in glob.glob("../examples/*.ipynb")[10:]:
        if "Distributions" in notebook or "Data_Loader" in notebook:
            continue
        nb = pm.execute_notebook(notebook, notebook, request_save_on_cell_execute=True)

        if nb["metadata"]["kernelspec"]["display_name"] != "Python 3":
            with open(notebook, "w") as fp:
                json.dump(nb, fp)


run_all_notebooks()
