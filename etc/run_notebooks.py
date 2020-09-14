import glob
import json

import papermill as pm


def run_all_notebooks():
    """Run all notebooks in the example directory."""
    for notebook in glob.glob("../examples/*.ipynb"):
        nb = pm.execute_notebook(notebook, notebook, request_save_on_cell_execute=True)

        if nb["metadata"]["kernelspec"]["display_name"] != "Python 3":
            nb["metadata"]["kernelspec"]["display_name"] = "Python 3"
            with open(notebook, "w") as fp:
                json.dump(nb, fp)


if __name__ == "__main__":
    run_all_notebooks()
