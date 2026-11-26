"""python -m warpline.checker <verb>: the checker's verbs, with the common ones (`compare`, `run-manifest`)."""
from warpline.checker import cli
from warpline.command import run

if __name__ == "__main__":
    run([cli], prog="python -m warpline.checker")
