"""Runs every test_* function in tests/test_hydro.py (pytest is not installed here) and writes reports/v3_tables/unit_tests.csv.
Run: cd tests && PYTHONPATH=../src:. python run_all.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import test_hydro as T

rows = []
for name in sorted(n for n in dir(T) if n.startswith("test_")):
    try:
        getattr(T, name)()
        rows.append(dict(test=name, result="pass", detail=""))
    except Exception as e:  # noqa: BLE001
        rows.append(dict(test=name, result="FAIL", detail=repr(e)[:200]))
out = Path(__file__).resolve().parents[1] / "reports" / "v3_tables"
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / "unit_tests.csv", index=False)
print(df.result.value_counts().to_dict())
print(df[df.result != "pass"].to_string(index=False))
