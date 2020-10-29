from pathlib import Path
import logging
import json

import papermill as pm


def run_all_notebooks():
    """Run all notebooks in the example directory."""
    for notebook in Path(__file__).parent.parent.glob("examples/*.ipynb"):
        notebook_path = str(notebook.resolve())
        nb = pm.execute_notebook(
            notebook_path,
            notebook_path,
            request_save_on_cell_execute=True,
            kernel_name="python3",
        )

        try:
            nb["metadata"]["kernelspec"]["display_name"] = "Python 3"
            nb["metadata"]["kernelspec"]["name"] = "python3"
        except KeyError:
            pass
        with open(notebook, "w") as fp:
            json.dump(nb, fp)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_notebooks()
