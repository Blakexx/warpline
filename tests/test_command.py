"""The one command: each module's verbs listed under its heading, the common verbs after them, no verb in two modules, and
the command of one module alone (`python -m warpline.checker`, `python -m warpline.recorder`) holds that module's verbs and
the common ones, and no other module's. Neither module imports the other; the common files import neither."""
import contextlib
import io
import re
import unittest
from pathlib import Path

from warpline import cli as common_cli
from warpline import command
from warpline.checker import cli as checker_cli
from warpline.recorder import cli as recorder_cli


def verbs(module):
    return [name for name, _ in _sections([module])[0][1]]


def _sections(modules):
    out = []
    for module in modules:
        sub = _Sub()
        module.register(sub)
        out.append((module.HEADING, sub.verbs))
    return out


class _Sub:
    def __init__(self):
        self.verbs = []

    def add_parser(self, name, help):
        self.verbs.append((name, help))
        return command.argparse.ArgumentParser()


class TestCommand(unittest.TestCase):
    def test_no_verb_is_in_two_modules(self):
        shared = set(verbs(recorder_cli)) & set(verbs(checker_cli))
        self.assertEqual(shared, set())
        command.build([recorder_cli, checker_cli])      # refuses a shared verb

    def test_a_shared_verb_is_refused(self):
        with self.assertRaises(ValueError):
            command.build([recorder_cli, recorder_cli])

    def test_help_lists_each_modules_verbs_under_its_heading(self):
        text = command.build([recorder_cli, checker_cli]).format_help()
        at = {m.HEADING: text.index(m.HEADING) for m in (recorder_cli, checker_cli)}
        self.assertLess(at[recorder_cli.HEADING], at[checker_cli.HEADING])
        for verb in verbs(recorder_cli):
            self.assertGreater(text.index(f"  {verb} "), at[recorder_cli.HEADING])
            self.assertLess(text.index(f"  {verb} "), at[checker_cli.HEADING])
        for verb in verbs(checker_cli):
            self.assertGreater(text.index(f"  {verb} "), at[checker_cli.HEADING])

    def test_a_module_alone_has_only_its_verbs(self):
        for module, other in ((recorder_cli, checker_cli), (checker_cli, recorder_cli)):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                module.main([verbs(other)[0]])
            self.assertIn("invalid choice", err.getvalue())

    def test_the_records_checks_and_the_checkers_check_are_two_verbs(self):
        self.assertIn("verify", verbs(recorder_cli))
        self.assertNotIn("check", verbs(recorder_cli))
        self.assertIn("check", verbs(checker_cli))
        self.assertNotIn("verify", verbs(checker_cli))

    def test_the_common_verbs_are_in_every_command_after_the_modules(self):
        self.assertEqual(verbs(common_cli), ["compare", "run-manifest", "run", "floor"])
        text = command.build([recorder_cli, checker_cli]).format_help()
        self.assertLess(text.index(checker_cli.HEADING), text.index(common_cli.HEADING))
        for module in (recorder_cli, checker_cli):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as e:
                module.main(["run-manifest", "--help"])
            self.assertIn("--place", out.getvalue())
            self.assertEqual(e.exception.code, 0)
        self.assertFalse(set(verbs(common_cli)) & (set(verbs(recorder_cli)) | set(verbs(checker_cli))))

    def test_neither_module_imports_the_other(self):
        for module, other in (("recorder", "checker"), ("checker", "recorder")):
            root = Path(command.__file__).parent / module
            for source in root.rglob("*.py"):
                found = re.findall(rf"^\s*(?:from|import)\s+(?:warpline\.{other}\b|\.\.{other}\b)", source.read_text(), re.M)
                self.assertEqual(found, [], f"{source} imports the {other}")

    def test_the_common_files_import_neither_module(self):
        common = sorted(Path(command.__file__).parent.glob("*.py"))
        self.assertTrue({"manifest.py", "checkers.py", "declare.py", "cli.py", "command.py", "run.py", "floor.py"} <=
                        {p.name for p in common})
        for source in common:
            found = re.findall(r"^\s*(?:from|import)\s+(?:warpline\.(?:recorder|checker)\b|\.(?:recorder|checker)\b|"
                               r"\.\s+import\s+(?:recorder|checker)\b)", source.read_text(), re.M)
            found += re.findall(r"^\s*from\s+(?:warpline|\.)\s+import\s+[^\n]*\b(?:recorder|checker)\b", source.read_text(), re.M)
            self.assertEqual(found, [], f"{source} imports a module")

    def test_both_modules_import_the_common_files(self):
        for module in ("recorder", "checker"):
            sources = (Path(command.__file__).parent / module).rglob("*.py")
            self.assertTrue(any(re.search(r"^from \.\.(?:manifest|cli|declare|checkers)? ?import", s.read_text(), re.M)
                                for s in sources), module)


if __name__ == "__main__":
    unittest.main()
