#!/usr/bin/env bash
# Run inside the activated `dig2` conda env (see environment.yml).
set -euo pipefail
cd "$(dirname "$0")/.."
# MOSES metrics package. Imported as `moses`; installed from GitHub with --no-deps because PyPI's
# molsets stops at 0.3.1 and its pinned dependencies (pomegranate, old pandas) are not needed.
pip install --no-deps "molsets @ git+https://github.com/molecularsets/moses.git@dd7ed6ab38e23afd3ef5371d67939a1760bd8599"
pip install -e . --no-deps
bash scripts/build_orca.sh
# molsets needs dig2's compatibility shims for current rdkit/pandas before it imports
# (a bare `import moses` fails on rdkit.six); check it the way the metrics code loads it.
python -c "import dig2, graph_tool, rdkit
from dig2.metrics.moses_style import _ensure_moses_compat_shims
_ensure_moses_compat_shims()
from moses.metrics.metrics import get_all_metrics
print('dig2 install OK')"
