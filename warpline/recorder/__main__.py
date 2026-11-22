"""python -m warpline.recorder <verb>: the recorder's verbs alone."""
from warpline.recorder import cli
from warpline.command import run

if __name__ == "__main__":
    run([cli], prog="python -m warpline.recorder")
