import re
import os
import shutil
import glob
import pathlib


widget_template = """.. _x-tutorial:

.. note::

    This tutorial is intended to be run in an IPython notebook.


.. include:: ../_notebooks/x.rst"""


def load_notebooks():
    """Load notebooks from the /example directory"""

    notebooks = glob.glob("../../examples/*.ipynb")

    outputs = []
    for notebook in notebooks:
        notebook_name = os.path.splitext(os.path.split(notebook)[1])[0]
        output_name = notebook_name.lower().replace(" ", "_")
        if output_name != "tutorial":
            outputs.append(output_name)
        print("Updating {}...".format(notebook_name))

        pathlib.Path("_notebooks").mkdir(exist_ok=True)

        shutil.copyfile(notebook, "_notebooks/" + output_name + ".ipynb")

    # Insert links to the widget ToC
    print("Finalizing Widget Page")
    text = open("widgets/index.rst", "r").read()
    text = re.sub(
        r"(:maxdepth: 1).*(Placeholder)",
        r"\1\n\n"
        + "\n".join(["   _notebooks/" + name for name in outputs])
        + r"\n\2",
        text,
        flags=re.S,
    )
    open("widgets/index.rst", "w").write(text)
