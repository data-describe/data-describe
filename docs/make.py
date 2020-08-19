import subprocess
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    subprocess.run(["sphinx-multiversion", "source", "build"])
