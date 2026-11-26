"""python -m warpline.recorder <verb>: the recorder's verbs, with the common ones (`compare`, `run-manifest`)."""
from warpline.recorder import cli
from warpline.command import run

if __name__ == "__main__":
    run([cli], prog="python -m warpline.recorder")
