import subprocess
import os


def run():
    subprocess.run(["sphinx-multiversion", "source", "build"])


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    run()
