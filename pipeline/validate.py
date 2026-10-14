"""TRICERATOPS validation of a planet candidate.
usage: validate.py TIC [N] [RUNS]"""
import os, sys, glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from astropy.io import fits
import triceratops.triceratops as tr

TIC = int(sys.argv[1]); N = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_000_000; RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
C = f"{os.environ.get('EXO_ROOT', '.')}/planet-candidates/TIC-{TIC}"
par = json.load(open(f"{C}/data/candidate_parameters.json"))
P, t0, depth, dur_h = par["P"], par["t0_bjd"] - 2457000, par["depth_ppm"] / 1e6, par["dur_h"]

# sectors and SPOC apertures as (column, row) pixels
sectors, apertures = [], []
for fn in sorted(glob.glob(f"{C}/data/tess_spoc_lightcurves/*.fits")):
    with fits.open(fn) as h:
        sec = int(h[0].header["SECTOR"]); ap = h[2].data; hd = h[2].header
        c0, r0 = hd["CRVAL1P"], hd["CRVAL2P"]
        yy, xx = np.where((ap & 2) > 0)                     # bit 2 = pipeline photometric aperture
        sectors.append(sec); apertures.append(np.column_stack([c0 + xx, r0 + yy]))
print(f"TIC {TIC}: P={P:.5f} d, depth={depth*1e6:.0f} ppm, dur={dur_h:.2f} h, sectors={sectors}, aperture sizes={[len(a) for a in apertures]}", flush=True)

lc = pd.read_csv(f"{C}/data/tess_lightcurve_detrended.csv")
ph = ((lc.btjd - t0 + 0.5 * P) % P) - 0.5 * P
m = np.abs(ph) < 3 * dur_h / 24
tt, ff = ph[m].values, lc.flux_detrended_1d[m].values
oot = np.abs(tt) > dur_h / 24
ff = ff / np.median(ff[oot])
bins = np.arange(tt.min(), tt.max() + 1e-9, 10 / 1440)
idx = np.digitize(tt, bins); tb = np.array([tt[idx == k].mean() for k in np.unique(idx) if (idx == k).sum() > 3])
fb = np.array([ff[idx == k].mean() for k in np.unique(idx) if (idx == k).sum() > 3])
err = np.std(fb[np.abs(tb) > dur_h / 24])
print(f"folded LC: {len(tb)} bins, per-bin scatter {err*1e6:.0f} ppm", flush=True)

target = tr.target(ID=TIC, sectors=np.array(sectors), search_radius=10)
print(target.stars[["ID", "Tmag", "ra", "dec", "sep (arcsec)"]].head(8).to_string(), flush=True)
target.calc_depths(tdepth=depth, all_ap_pixels=apertures)
print(target.stars[["ID", "Tmag", "sep (arcsec)", "fluxratio", "tdepth"]].head(8).to_string(), flush=True)
FPPs, NFPPs, tabs = [], [], []
for k in range(RUNS):
    target.calc_probs(time=tb, flux_0=fb, flux_err_0=err, P_orb=P, N=N, parallel=False, verbose=0)
    FPPs.append(target.FPP); NFPPs.append(target.NFPP); tabs.append(target.probs.copy())
    print(f"  run {k+1}/{RUNS}: FPP={target.FPP:.4f} NFPP={target.NFPP:.5f}", flush=True)
FPPs, NFPPs = np.array(FPPs), np.array(NFPPs)
print(f"\nRESULT TIC {TIC}: FPP = {FPPs.mean():.4f} ± {FPPs.std(ddof=1):.4f}   NFPP = {NFPPs.mean():.5f} ± {NFPPs.std(ddof=1):.5f}   ({RUNS} runs x N={N:.0e})", flush=True)
avg = pd.concat(tabs).groupby(["ID", "scenario"], as_index=False)["prob"].mean().sort_values("prob", ascending=False)
print(avg.head(8).to_string(), flush=True)
json.dump(dict(tic=TIC, FPP_mean=float(FPPs.mean()), FPP_std=float(FPPs.std(ddof=1)), NFPP_mean=float(NFPPs.mean()), NFPP_std=float(NFPPs.std(ddof=1)),
               runs=RUNS, N=N, sectors=sectors, depth_ppm=depth * 1e6, scenarios=avg.head(12).to_dict(orient="records"),
               stars=target.stars[["ID", "Tmag", "sep (arcsec)", "fluxratio", "tdepth"]].head(10).to_dict(orient="records")),
          open(f"triceratops_{TIC}.json", "w"), indent=1, default=str)
