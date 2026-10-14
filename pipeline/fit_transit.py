"""Transit fit (batman + emcee) with free stellar density.
usage: fit_transit.py TIC [EXTRA_LC.npy]"""
import os, sys, glob, json, warnings, numpy as np
warnings.filterwarnings("ignore")
import batman, emcee
import lightkurve as lk
from wotan import flatten, transit_mask

TIC = int(sys.argv[1]); EXTRA = sys.argv[2] if len(sys.argv) > 2 else None
C = f"{os.environ.get('EXO_ROOT', '.')}/planet-candidates/TIC-{TIC}"
par = json.load(open(f"{C}/data/candidate_parameters.json"))
P0, T00, dep0, dur0 = par["P"], par["t0_bjd"] - 2457000, par["depth_ppm"] / 1e6, par["dur_h"] / 24
R, M = par["rad"], par["mass"]; rho_tic = M * 1.989e33 / (4 / 3 * np.pi * (R * 6.957e10) ** 3)
G = 6.674e-8

# per sector, with transits masked in the detrend
groups = []
for fn in sorted(glob.glob(f"{C}/data/tess_spoc_lightcurves/*.fits")):
    lc = lk.read(fn, quality_bitmask="default").remove_nans().normalize()
    t = np.asarray(lc.time.value, float); f = np.ma.filled(np.asarray(lc.flux.value, float), np.nan)
    msk = transit_mask(time=t, period=P0, duration=3 * dur0, T0=T00)
    fl, _ = flatten(t, f, method="biweight", window_length=1.0, break_tolerance=0.3, return_trend=True, mask=msk)
    g = np.isfinite(fl); t, fl = t[g], fl[g]
    ph = ((t - T00 + 0.5 * P0) % P0) - 0.5 * P0; near = np.abs(ph) < 4 * max(dur0, 1 / 24)
    cad = np.median(np.diff(t)) * 86400
    oot = near & (np.abs(ph) > 1.5 * dur0)
    groups.append(dict(t=t[near], f=fl[near], e=np.full(near.sum(), np.std(fl[oot]) if oot.sum() > 10 else np.std(fl)), cad=cad))
if EXTRA:
    tn, fn_ = np.load(EXTRA); ph = ((tn - T00 + 0.5 * P0) % P0) - 0.5 * P0; near = np.abs(ph) < 4 * max(dur0, 1 / 24)
    oot = near & (np.abs(ph) > 1.5 * dur0)
    groups.append(dict(t=tn[near], f=fn_[near], e=np.full(near.sum(), np.std(fn_[oot])), cad=200.0))
print(f"TIC {TIC}: {len(groups)} sectors, cadences {[round(g['cad']) for g in groups]} s, rho_TIC = {rho_tic:.1f} g/cc")

u = [0.25, 0.35]  # fixed quadratic LD for M dwarfs in the TESS band
models = []
for g in groups:
    pm = batman.TransitParams(); pm.t0, pm.per, pm.rp, pm.a, pm.inc, pm.ecc, pm.w, pm.u, pm.limb_dark = T00, P0, 0.1, 20, 89, 0, 90, u, "quadratic"
    ss = max(1, int(round(g["cad"] / 120)))
    models.append((pm, batman.TransitModel(pm, g["t"], supersample_factor=ss, exp_time=g["cad"] / 86400)))

def unpack(th):
    t0, P, k, b, lrho = th; rho = 10 ** lrho
    a = (G * rho * (P * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)          # a/R* from stellar density
    return t0, P, k, b, rho, a

def lnprob(th):
    t0, P, k, b, rho, a = unpack(th)
    if not (0.005 < k < 0.6 and 0 <= b < 1 + k and -1 < np.log10(rho) < 2.5 and abs(t0 - T00) < 0.05 and abs(P - P0) < 1e-3) or a <= 1 + k: return -np.inf
    inc = np.degrees(np.arccos(b / a)); lp = 0
    for (pm, m), g in zip(models, groups):
        pm.t0, pm.per, pm.rp, pm.a, pm.inc = t0, P, k, a, inc
        lp += -0.5 * np.sum(((g["f"] - m.light_curve(pm)) / g["e"]) ** 2)
    return lp

nw, nd = 32, 5; rng = np.random.default_rng(1)
p0 = np.column_stack([T00 + rng.normal(0, 2e-3, nw), P0 + rng.normal(0, 2e-5, nw), np.sqrt(dep0) * (1 + rng.normal(0, 0.05, nw)),
                      rng.uniform(0.05, 0.6, nw), np.log10(rho_tic) + rng.normal(0, 0.1, nw)])
S = emcee.EnsembleSampler(nw, nd, lnprob); S.run_mcmc(p0, 4000, progress=False)
ch = S.get_chain(discard=1500, thin=10, flat=True)
t0, P, k, b, rho, a = unpack(ch.T)
q = lambda x: np.percentile(x, [16, 50, 84])
def fmt(x, d=3): lo, md, hi = q(x); return f"{md:.{d}f} +{hi-md:.{d}f}/-{md-lo:.{d}f}"
Rp = k * R * 109.1
print(f"Rp/R* = {fmt(k)}   b = {fmt(b,2)}   rho* = {fmt(rho,1)} g/cc (TIC: {rho_tic:.1f})   Rp = {fmt(Rp,2)} Re")
print(f"P = {fmt(P,6)} d   T0 = {fmt(t0 + 2457000,4)} BJD_TDB   acceptance {np.mean(S.acceptance_fraction):.2f}")
frac_consistent = np.mean(np.abs(np.log10(rho) - np.log10(rho_tic)) < 0.3)
lo95 = np.percentile(rho, 2.5)
print(f"fraction of posterior with rho* within a factor 2 of TIC: {frac_consistent:.2f};  rho* 95% lower limit {lo95:.1f} g/cc;  P(b>0.9) = {np.mean(b>0.9):.2f}")
json.dump(dict(tic=TIC, rp_rs=list(q(k)), b=list(q(b)), rho=list(q(rho)), rho_tic=rho_tic, rp_earth=list(q(Rp)), P=list(q(P)), t0_bjd=list(q(t0 + 2457000)),
               frac_rho_consistent=float(frac_consistent), rho_lo95=float(lo95), p_grazing=float(np.mean(b > 0.9)), ld=u),
          open(f"fit_{TIC}{'_all' if EXTRA else ''}.json", "w"), indent=1)
np.save(f"fit_{TIC}{'_all' if EXTRA else ''}_chain.npy", ch)
