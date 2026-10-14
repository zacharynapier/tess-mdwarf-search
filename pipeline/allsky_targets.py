"""Select nearby faint M dwarfs with >= 4 TESS-SPOC FFI sectors, no 2-min data and no known object nearby.
writes allsky_targets.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, time
from astroquery.mast import Catalogs, Observations
from astropy.coordinates import SkyCoord; import astropy.units as u
from known import check
t0 = time.time()
tab = Catalogs.query_criteria(catalog="Tic", Teff=[2700, 3800], Tmag=[13.0, 15.5], d=[0, 100], logg=[4.6, 5.6],
                              objType="STAR").to_pandas()[["ID", "ra", "dec", "Teff", "Tmag", "rad", "mass", "d", "contratio", "gallat"]]
print(f"TIC M dwarfs T=13-15.5, d<100pc: {len(tab)}  ({time.time()-t0:.0f}s)")
c = SkyCoord(tab.ra.values*u.deg, tab.dec.values*u.deg)
lmc = c.separation(SkyCoord(80.89*u.deg, -69.76*u.deg)).deg < 12; smc = c.separation(SkyCoord(13.19*u.deg, -72.83*u.deg)).deg < 7
tab = tab[(np.abs(tab.gallat) > 10) & ~lmc & ~smc & (tab.contratio.fillna(0) < 0.5)]
print("after plane/MCs/contamination cuts:", len(tab))
ids = tab.ID.astype(str).tolist(); cnt, cnt2 = {}, {}
for i in range(0, len(ids), 400):
    for tries in range(3):
        try:
            o = Observations.query_criteria(obs_collection="HLSP", provenance_name="TESS-SPOC", target_name=ids[i:i+400]).to_pandas()
            o2 = Observations.query_criteria(obs_collection="TESS", provenance_name="SPOC", t_exptime=[100, 140], target_name=ids[i:i+400]).to_pandas()
            break
        except Exception: time.sleep(5)
    else: raise RuntimeError(f"MAST query failed for batch {i}")
    for k, v in o.groupby("target_name").size().items(): cnt[k] = v
    for k, v in o2.groupby("target_name").size().items(): cnt2[k] = v
tab["nsec_ffi"] = tab.ID.astype(str).map(cnt).fillna(0).astype(int); tab["nsec_2min"] = tab.ID.astype(str).map(cnt2).fillna(0).astype(int)
print(f"with >=4 TESS-SPOC FFI sectors: {(tab.nsec_ffi>=4).sum()}   of those never 2-min: {((tab.nsec_ffi>=4)&(tab.nsec_2min==0)).sum()}")
tab = tab[(tab.nsec_ffi >= 4) & (tab.nsec_2min == 0)]
tab["known"] = [len(check(r.ra, r.dec, tic=int(r.ID), radius_arcmin=1.5)) for r in tab.itertuples()]
tab = tab[tab.known == 0].sort_values("nsec_ffi", ascending=False)
print(f"FINAL clean targets: {len(tab)}   total sector-files: {tab.nsec_ffi.sum()}   ({time.time()-t0:.0f}s)")
print(tab.nsec_ffi.describe().round(1).to_string())
tab.to_csv("allsky_targets.csv", index=False)
