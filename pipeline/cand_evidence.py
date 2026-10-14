"""Evidence folder for a planet candidate: detrended light curve, summary figure, parameters.
usage: cand_evidence.py TIC [TIC ...]  (reads evid/TIC/)"""
import os, sys, glob, json, shutil, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lightkurve as lk
from wotan import flatten
from astropy.coordinates import SkyCoord; import astropy.units as u

ROOT = f"{os.environ.get('EXO_ROOT', '.')}/planet-candidates"
TGT = pd.read_csv("allsky_targets.csv").set_index("ID")

for tic in map(int, sys.argv[1:]):
    src = f"evid/{tic}"; out = f"{ROOT}/TIC-{tic}"
    os.makedirs(f"{out}/figures", exist_ok=True); os.makedirs(f"{out}/data/tess_spoc_lightcurves", exist_ok=True); os.makedirs(f"{out}/data/vetting", exist_ok=True)
    o = json.load(open(f"{src}/{tic}.json")); s = o["signals"][0]; P, t0, dur = s["period"], s["t0"], s["dur_h"] / 24
    star = TGT.loc[tic]
    TT, FF = [], []
    for fn in sorted(glob.glob(f"{src}/tess_spoc_lc/*.fits")):
        lc = lk.read(fn, quality_bitmask="default").remove_nans().normalize()
        tt = np.asarray(lc.time.value, float); ff = np.ma.filled(np.asarray(lc.flux.value, float), np.nan)
        fw = flatten(tt, ff, method="biweight", window_length=1.0, break_tolerance=0.3); g = np.isfinite(fw); TT.append(tt[g]); FF.append(fw[g])
        shutil.copy(fn, f"{out}/data/tess_spoc_lightcurves/")
    t, f = np.concatenate(TT), np.concatenate(FF)
    pd.DataFrame(dict(btjd=t, flux_detrended_1d=f)).to_csv(f"{out}/data/tess_lightcurve_detrended.csv", index=False)
    ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P; n = np.round((t - t0) / P).astype(int)
    oot = (np.abs(ph) > 1.5 * dur) & (np.abs(ph) < 6 * dur); base = np.median(f[oot])
    intr = np.abs(ph) < dur / 3; depth = (base - f[intr].mean()); derr = np.std(f[oot]) / np.sqrt(intr.sum())
    sec = np.abs(np.abs(ph) - 0.5 * P) < dur / 3; secd = base - f[sec].mean() if sec.sum() else np.nan; secerr = np.std(f[oot]) / np.sqrt(max(sec.sum(), 1))
    def binned(x, y, w):
        e = np.arange(x.min(), x.max() + w, w); i = np.digitize(x, e)
        r = [(x[i == k].mean(), np.mean(y[i == k]), y[i == k].std() / np.sqrt((i == k).sum())) for k in np.unique(i) if (i == k).sum() > 3]
        return np.array(r).T if r else (np.array([]), np.array([]), np.array([]))
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    m = np.abs(ph) < 0.4; x, y, e = binned(ph[m] * 24, f[m], max(dur * 24 / 4, 0.15))
    ax[0, 0].errorbar(x, (y - base) * 1e6, e * 1e6, fmt="o", ms=3, c="k"); ax[0, 0].axhline(0, c="gray", lw=.5)
    ax[0, 0].axvspan(-dur * 12, dur * 12, color="C1", alpha=.12, label=f"BLS duration {dur*24:.2f} h")
    ax[0, 0].set_xlim(-8, 8); ax[0, 0].set_xlabel("hours from mid-transit"); ax[0, 0].set_ylabel("ppm"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title(f"Transit: P = {P:.5f} d, depth {depth*1e6:.0f} ± {derr*1e6:.0f} ppm (1-day detrend)")
    for par, c, lab in [(0, "C0", "even"), (1, "C3", "odd")]:
        mm = (n % 2 == par) & (np.abs(ph) < 0.3); x, y, e = binned(ph[mm] * 24, f[mm], max(dur * 24 / 3, 0.2))
        ax[0, 1].errorbar(x, (y - base) * 1e6, e * 1e6, fmt="o", ms=3, c=c, label=lab)
    ax[0, 1].set_xlim(-6, 6); ax[0, 1].axhline(0, c="gray", lw=.5); ax[0, 1].legend(); ax[0, 1].set_xlabel("hours from mid-transit")
    ax[0, 1].set_title(f"Odd vs even transits: {s['depth_odd_ppm']:.0f} vs {s['depth_even_ppm']:.0f} ppm ({s['odd_even_sig']:.1f}σ apart)")
    phs = ((ph + P) % P) - 0.5 * P   # centred on phase 0.5
    mm = np.abs(phs) < 0.3; x, y, e = binned(phs[mm] * 24, f[mm], max(dur * 24 / 3, 0.2))
    ax[1, 0].errorbar(x, (y - base) * 1e6, e * 1e6, fmt="o", ms=3, c="k"); ax[1, 0].axhline(0, c="gray", lw=.5)
    ax[1, 0].axhline(-depth * 1e6, c="C3", ls=":", label="primary depth (for scale)"); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_xlim(-6, 6); ax[1, 0].set_xlabel("hours from phase 0.5"); ax[1, 0].set_ylabel("ppm")
    ax[1, 0].set_title(f"Secondary eclipse search at phase 0.5: {secd*1e6:.0f} ± {secerr*1e6:.0f} ppm")
    pix = np.load(f"{src}/pix_{tic}.npz", allow_pickle=True); S = pix["S"]; g = pix["grid"]
    ax[1, 1].imshow(np.clip(S, 0, None), origin="lower", extent=[g[0] - 1, g[-1] + 1, g[0] - 1, g[-1] + 1], cmap="magma")
    gm = pix["gaia_G"]; dx, dy = pix["gaia_dx"], pix["gaia_dy"]; vis = (np.abs(dx) < 90) & (np.abs(dy) < 90)
    ax[1, 1].scatter(dx[vis], dy[vis], s=np.clip(4 * (21 - gm[vis]), 4, 80), facecolors="none", edgecolors="cyan", lw=1, label="Gaia DR3 stars")
    ax[1, 1].plot(0, 0, "*", ms=16, mfc="none", mec="lime", mew=2, label=f"TIC {tic}")
    bx, by = pix["best"]; ex, ey = pix["err"]
    ax[1, 1].errorbar(bx, by, xerr=ex, yerr=ey, fmt="x", c="white", ms=10, mew=2, capsize=3, label="dip source (best fit)")
    ax[1, 1].legend(loc="lower left", fontsize=8); ax[1, 1].set_xlabel("ΔRA [arcsec]"); ax[1, 1].set_ylabel("ΔDec [arcsec]")
    ax[1, 1].set_title("Difference-image localization")
    plt.suptitle(f"TIC {tic}: planet candidate (Rp ≈ {np.sqrt(max(depth,0))*star.rad*109.1:.1f} R⊕, P = {P:.4f} d)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{out}/summary.png", dpi=100); plt.close()
    shutil.copy(f"{src}/cand_{tic}.png", f"{out}/figures/search_diagnostics.png")
    shutil.copy(f"{src}/pix_{tic}.png", f"{out}/figures/pixel_localization.png")
    shutil.copy(f"{src}/pix_{tic}.npz", f"{out}/data/vetting/pixel_localization.npz")
    shutil.copy(f"{src}/evidence_{tic}.txt", f"{out}/data/vetting/vetting_log.txt")
    shutil.copy(f"{src}/{tic}.json", f"{out}/data/vetting/search_result.json")
    c = SkyCoord(o["ra"] * u.deg, o["dec"] * u.deg)
    rp = np.sqrt(max(depth, 0)) * star.rad * 109.1
    a_au = (star["mass"] * (P / 365.25) ** 2) ** (1 / 3)
    info = dict(tic=tic, P=P, t0_bjd=t0 + 2457000, depth_ppm=depth * 1e6, depth_err=derr * 1e6, dur_h=dur * 24, dur_max=s["dur_expected_h"],
                rp=rp, a_au=a_au, snr=s["snr"], sde=s["sde"], ntr=s["ntransits_obs"], nsec=o["nsectors"], sectors=o["sectors"],
                odd=s["depth_odd_ppm"], even=s["depth_even_ppm"], oe=s["odd_even_sig"], secd=secd * 1e6, secerr=secerr * 1e6,
                ra=o["ra"], dec=o["dec"], radec=c.to_string("hmsdms", precision=2), Tmag=star.Tmag, Teff=star.Teff, rad=star.rad, mass=star["mass"], dist=star.d)
    json.dump(info, open(f"{out}/data/candidate_parameters.json", "w"), indent=1, default=float)
