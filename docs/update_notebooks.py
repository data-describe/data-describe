import re
import os
import shutil
import glob
import pathlib

widget_template = """.. _x-tutorial:

.. note::

    This tutorial is intended to be run in an IPython notebook.


.. include:: ../_notebooks/x.rst"""


def run(argv=None):
    """Update notebooks."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    notebooks = glob.glob("../notebooks/*.ipynb")
    outputs = []
    for notebook in notebooks:
        notebook_name = os.path.splitext(os.path.split(notebook)[1])[0]
        output_name = notebook_name.lower().replace(" ", "_")
        if output_name != "tutorial":
            outputs.append(output_name)
        print("Updating {}...".format(notebook_name))

        pathlib.Path("source/_notebooks").mkdir(exist_ok=True)
        shutil.copyfile(notebook, "source/_notebooks/" + output_name + ".ipynb")

    # Insert links to the widget ToC
    print("Finalizing Widget Page")
    text = open("source/widgets/index.rst", "r").read()
    text = re.sub(
        r"(:maxdepth: 1).*(Placeholder)",
        r"\1\n\n"
        + "\n".join(["   ../_notebooks/" + name for name in outputs])
        + r"\n\2",
        text,
        flags=re.S,
    )
    open("source/widgets/index.rst", "w").write(text)


if __name__ == "__main__":
    run()
