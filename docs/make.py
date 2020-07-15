from argparse import ArgumentParser
import subprocess


def run(argv=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--version", help="Current version of documentation", default="dev"
    )
    parser.add_argument(
        "--no-build", action="store_true", help="Do not build the HTML documentation"
    )
    build_args = parser.parse_args(argv)

    subprocess.call(
        ["sphinx-apidoc", "-f", "-M", "-o", "docs/source/_api", "data_describe"]
    )
    subprocess.call(["python", "docs/update_notebooks.py"])

    if not build_args.no_build:
        subprocess.call(
            [
                "sphinx-build",
                "-b",
                "html",
                "docs/source",
                f"/build/docs/{build_args.version}",
            ]
        )


if __name__ == "__main__":
    run()
