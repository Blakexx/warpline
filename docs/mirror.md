# The public mirror, and what is checked before a push

A push to the default branch publishes: `.github/workflows/mirror.yml` exports the paths `.github/mirror/declarations.json`
declares `ships` and refuses a tracked path declared neither `ships` nor `private`, an agent file, or a private-shaped
string. Declare a new top-level file or directory there when it is added.

`bazel test //...` runs the first of those refusals locally, as the test `//:mirror_declarations`: rola-devtools' mirror
tool (`python3 -m rola_devtools.mirror --check`), at the commit the workflow installs, over the checkout's git index. A
file that is `git add`ed (or committed) and undeclared fails it, naming the path. Run `bazel test //...` before a push.

What only the workflow covers: the tripwire over the exported files (agent files, home directories, personal addresses,
credentials) and the push itself. There is no commit hook here: rola's is its own `bazel/bazel` stub and
`tools/git-hooks/pre-commit` over a dev config, which this library does not carry; one would need the shared module's
front door (`rola_devtools.graph.front_door --commit-gate`) and a stub that finds it.

The tool's commit is pinned in `MODULE.bazel` and in `mirror.yml`: change both together.
