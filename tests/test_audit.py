"""Static checks on the repository itself (no data, no GPU)."""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
THIS = pathlib.Path(__file__).resolve()


def test_no_sys_path_hacks():
    hits = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py") if p.resolve() != THIS
            and ("sys.path.insert" in p.read_text() or "sys.path.append" in p.read_text())]
    assert not hits, hits


def test_every_module_is_reachable():
    """Every dig2 module is imported (transitively) from an entry point or a test."""
    pkg = ROOT / "dig2"
    modules = {".".join(p.relative_to(ROOT).with_suffix("").parts): p
               for p in pkg.rglob("*.py") if p.name != "__init__.py"}
    imported = set()
    for p in list(pkg.rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
    entry_points = {m for m in modules if m.startswith("dig2.tools.") or m == "dig2.train"}
    orphans = sorted(m for m in modules if m not in imported and m not in entry_points)
    assert not orphans, orphans
