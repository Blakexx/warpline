"""The checker: a compiled CUDA kernel in, what is wrong with it out. Four layers, each depending only on the one before:
`readers` (a vendor tool's text -> plain data), `model` (one data structure a kernel), `rules` (small units that read the
model), `manifest` (the static manifest's collection, and a run manifest's profile rules: the forms, the checkers and the
declarations are the common part's, `warpline/manifest.py`, `checkers.py`, `declare.py`) with `runner` (the vendor tools'
invocation) and `cli` (the command)."""
