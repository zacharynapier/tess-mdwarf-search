"""Run the search on known planet hosts (positive control).
usage: positive_control.py TIC [TIC ...]"""
import sys, json, re, os, warnings; warnings.filterwarnings("ignore")
from astroquery.mast import Observations
import search
for tic in map(int, sys.argv[1:]):
    o = Observations.query_criteria(obs_collection="HLSP", provenance_name="TESS-SPOC", target_name=str(tic)).to_pandas()
    secs = sorted({int(re.search(r"-s(\d{4})", x).group(1)) for x in o.obs_id if re.search(r"-s(\d{4})", x)})
    search.SECTORS[str(tic)] = secs; print(tic, "TESS-SPOC sectors:", secs, flush=True)
    d = search.prepare((tic, float("nan"), float("nan")))
    if "error" in d: print("prep error", d["error"]); continue
    os.makedirs("cand", exist_ok=True)
    out = search.analyze(d, "cand")
    json.dump(out, open(f"cand/{tic}.json", "w"), default=float)
    s = out["signals"][0]; print(f"  top signal P={s['period']:.5f} depth={s['depth_ppm']:.0f} SDE={s['sde']:.1f}", flush=True)
