"""Map each target to its TESS-SPOC FFI sectors (sector_map.json)."""
import warnings, re, time, json; warnings.filterwarnings("ignore")
import pandas as pd
from astroquery.mast import Observations
df = pd.read_csv("allsky_targets.csv"); ids = df.ID.astype(str).tolist(); m = {}
for i in range(0, len(ids), 400):
    for k in range(4):
        try:
            o = Observations.query_criteria(obs_collection="HLSP", provenance_name="TESS-SPOC", target_name=ids[i:i+400]).to_pandas(); break
        except Exception: time.sleep(15)
    else: raise RuntimeError(f"MAST query failed for batch {i}")
    for tn, oid in zip(o.target_name, o.obs_id):
        s = re.search(r"-s(\d{4})", oid)
        if s: m.setdefault(str(tn), set()).add(int(s.group(1)))
json.dump({k: sorted(v) for k, v in m.items()}, open("sector_map.json", "w"))
print(len(m), "TICs mapped,", sum(len(v) for v in m.values()), "sector files")
