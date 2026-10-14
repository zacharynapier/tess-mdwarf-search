"""Apply the automatic cuts to search results and list the signals that pass.
usage: RESDIR=results python triage.py"""
import glob, json, os, numpy as np, pandas as pd

R = os.environ.get("RESDIR", "results")


def flags_for(s):
    f = []
    if s["sde"] < 9: f.append("lowSDE")
    if s["snr"] < 10: f.append("lowSNR")
    if s["ntransits_obs"] < 3: f.append("<3tr")
    if s["odd_even_sig"] > 3: f.append("odd/even")
    if s["secondary_sig"] > 3: f.append("secondary")
    if s["secondary_sig"] < -5: f.append("brightening")
    if s["dur_h"] > 2.5 * s["dur_expected_h"]: f.append("too-long")
    if s.get("rot_harmonic"): f.append("rotation")
    if s.get("known_match"): f.append("KNOWN")
    h1, h2 = s["depth_half1_ppm"], s["depth_half2_ppm"]
    if np.isfinite(h1) and np.isfinite(h2) and min(h1, h2) < 0.3 * s["depth_ppm"]: f.append("half-inconsistent")
    for sp in (13.7, 6.85, 1.0, 0.5, 2.0):  # TESS orbit, half orbit, 1 d and aliases
        if abs(s["period"] / sp - 1) < 0.01: f.append("systematic")
    if s["rp_earth"] > 20: f.append("too-big")
    return f


def phase_variable(tic, P, t0, dur, depth):
    try:
        t, f = np.load(f"{R}/{tic}_lc.npy")
    except Exception:
        return False
    ph = (((t - t0) / P + 0.5) % 1) - 0.5; oot = np.abs(ph) > dur / P
    e = np.linspace(-0.5, 0.5, 41); i = np.digitize(ph[oot], e); fo = f[oot]
    mu = np.array([fo[i == k].mean() for k in range(1, 41) if (i == k).sum() > 20])
    se = np.array([fo[i == k].std() / np.sqrt((i == k).sum()) for k in range(1, 41) if (i == k).sum() > 20])
    if len(mu) < 5: return False
    chi2 = np.sum(((mu - np.average(mu, weights=1 / se ** 2)) / se) ** 2) / (len(mu) - 1)
    return chi2 > 4 and (mu.max() - mu.min()) / max(depth, 1e-9) > 0.3


rows, errors = [], 0
for fn in sorted(glob.glob(f"{R}/*.json")):
    o = json.load(open(fn))
    if "error" in o: errors += 1; continue
    for s in o.get("signals", []):
        r = dict(tic=o["tic"], rank=s["rank"], P=round(s["period"], 5), dur=round(s["dur_h"], 2), depth=round(s["depth_ppm"]),
                 rp=round(s["rp_earth"], 2), snr=round(s["snr"], 1), sde=round(s["sde"], 1), ntr=s["ntransits_obs"], t0=s["t0"],
                 flags=flags_for(s))
        if not r["flags"] and phase_variable(r["tic"], r["P"], r["t0"], r["dur"] / 24, r["depth"] / 1e6): r["flags"] = ["phase-variable"]
        rows.append(r)
df = pd.DataFrame(rows); df["flags"] = df["flags"].str.join(" ")
passing = df[df["flags"] == ""].sort_values("sde", ascending=False)
print(f"stars: {df.tic.nunique()}  errors: {errors}  signals: {len(df)}  passing: {len(passing)}")
print(passing.drop(columns=["t0", "flags"]).to_string(index=False))
df.to_csv(f"triage_{os.path.basename(os.path.normpath(R))}.csv", index=False)
