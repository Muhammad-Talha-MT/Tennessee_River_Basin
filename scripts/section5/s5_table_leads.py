"""Section 5 table of skill over the full horizon (table S? / Table 3 replacement): median NSE and KGE over the gauges of each protocol, for
(a) the mean over all 40 leads (6-240 h, i.e. 1-10 days) and (b) single leads 6, 12, 18, 24, 48, 72, 120 and 240 h. Per lead the median is taken over the gauges, as in the
other Section 5 tables. Models: GNN (M0), GNN-noIns (M1), GNN-noMP (M2), GNN-Instat (M4, held-out and blocked only), LSTM, NWM. Writes
reports/section5/s5_12_table_leads.csv (long format: protocol, model, lead_h or 'mean_1_10d', nse, kge, n). Run: python scripts/section5/s5_table_leads.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import pandas as pd

import v3_common as V

OUT = V.REPO / "reports" / "section5"
MODELS = ("M0", "M1", "M2", "M4", "LSTM")
LEADS = (6, 12, 18, 24, 48, 72, 120, 240)
rows = []
for p in V.PROTOCOLS:
    data = V.load_protocol(p, MODELS)
    data["NWM"] = V.load_nwm()
    base = np.isfinite(V.scores(*data["M0"])["nse"][3])  # the protocol's scored gauges (set by the GNN, as in the other figures); every model uses this set
    for m, (t, v) in data.items():
        sc = V.scores(t, v)
        ok = base & np.isfinite(sc["nse"][3])
        n = int(ok.sum())
        nse_l = np.array([np.nanmedian(sc["nse"][li, ok]) for li in range(40)])
        kge_l = np.array([np.nanmedian(sc["kge"][li, ok]) for li in range(40)])
        rows.append(dict(protocol=p, model=m, lead_h="mean_1_10d", nse=float(nse_l.mean()), kge=float(kge_l.mean()), n=n))
        for h in LEADS:
            li = h // 6 - 1
            rows.append(dict(protocol=p, model=m, lead_h=h, nse=float(nse_l[li]), kge=float(kge_l[li]), n=n))
        print(p, m, n, round(nse_l.mean(), 3), round(kge_l.mean(), 3), flush=True)
pd.DataFrame(rows).to_csv(OUT / "s5_12_table_leads.csv", index=False)
