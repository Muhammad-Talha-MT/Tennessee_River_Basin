"""Writes reports/RESULTS.md: the single results document. Part 1 is the working report (results against RQ1-RQ3, pending results as "-"),
part 2 the full appendix of tables. Both are generated from reports/v3_tables/*.csv by v3_working_report.py and v3_report.py.
Run after the other v3 scripts: python scripts/v3_results.py"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_common import REPO

rep = REPO / "reports"
working = (rep / "test_comparison_v3.md").read_text()
appendix = (rep / "test_comparison_v3_full.md").read_text()
head = (f"# Results: physics-guided routed model vs LSTM, Tennessee River Basin\n\n"
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/v3_results.py` from `reports/v3_tables/*.csv`. "
        "Part 1 answers the research questions; part 2 is the full set of tables. A dash (-) means the result is pending. "
        "Old reports in this folder (`test_comparison_M0_A0_LSTM-AS_NWM.md`, `test_comparison_v3.md`, `test_comparison_v3_full.md`) are superseded by this file.\n\n---\n\n")
body = working.replace("# Working report: results against the research questions", "## Part 1. Results against the research questions", 1)
app = appendix.replace("# Test-set comparison v3: M0, M1, M2, LSTM and NWM", "## Part 2. Appendix: full tables", 1)
app = "\n".join(("#" + l if l.startswith("#") else l) for l in app.splitlines())  # one level deeper, so the appendix sits under Part 2
(rep / "RESULTS.md").write_text(head + body + "\n\n---\n\n" + app)
print("wrote reports/RESULTS.md,", len((head + body + app).splitlines()), "lines")
