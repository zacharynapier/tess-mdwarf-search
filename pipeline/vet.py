"""Vet one candidate: catalog checks, re-detrending, pixel-level localization.
usage: vet.py TIC [RANK]  (reads cand/TIC.json, cand/TIC_lc.npy)"""
import sys, os, json, subprocess, urllib.request, numpy as np, warnings
warnings.filterwarnings("ignore")
from multiprocessing import get_context
from astropy.coordinates import SkyCoord
import astropy.units as u

TIC = int(sys.argv[1]); RANK = int(sys.argv[2]) if len(sys.argv) > 2 else 1
o = json.load(open(f"cand/{TIC}.json")); s = o["signals"][RANK - 1]
P, t0, durh = s["period"], s["t0"], s["dur_h"]
REP = []
def say(x): print(x, flush=True); REP.append(x)

HIN, OUTA, OUTB = min(1.0, durh / 3) / 24, max(3.5, 1.5 * durh) / 24, max(7.0, 3 * durh) / 24
import lightkurve as lk

def one(sec):
    from wotan import flatten
    d = f"cand/tp{TIC}"; os.makedirs(d, exist_ok=True)
    try:
        from search import static_download
        fn = static_download(TIC, sec, d, kind="tp")  # SPOC FFI target pixel file
        if not fn: return dict(sector=sec, err="no tp file")
        tpf = lk.read(fn, quality_bitmask="default")
    except Exception as ex: return dict(sector=sec, err=repr(ex)[:100])
    t = tpf.time.value; F = np.asarray(tpf.flux.value, float); g = np.isfinite(t) & np.all(np.isfinite(F.reshape(len(t), -1)), 1)
    t, F = t[g], F[g]
    if len(t) < 200: return dict(sector=sec, err="empty cutout")
    F = F - np.percentile(F.reshape(len(t), -1), 30, axis=1)[:, None, None]
    ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P; intr = np.abs(ph) < HIN; oot = (np.abs(ph) > OUTA) & (np.abs(ph) < OUTB)
    ny, nx = F.shape[1:]; D = np.zeros((ny, nx)); E = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            fl = F[:, y, x]; tr = flatten(t, fl, method="biweight", window_length=1.0, return_trend=True, break_tolerance=0.3)[1]
            r = fl - tr; ok = np.isfinite(r)
            D[y, x] = np.mean(r[ok & oot]) - np.mean(r[ok & intr])
            E[y, x] = np.std(r[ok & oot]) * np.sqrt(1 / max((ok & intr).sum(), 1) + 1 / max((ok & oot).sum(), 1))
    col, row = np.meshgrid(np.arange(nx), np.arange(ny)); sky = tpf.wcs.pixel_to_world(col, row)
    return dict(sector=sec, D=D, E=E, ra=sky.ra.deg, dec=sky.dec.deg)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(here, "inspect_candidate.py"), "cand", str(TIC), str(RANK)], capture_output=True, check=True)

    def vz(src, ra, dec, r, cols="-out.all", extra=""):
        url = f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source={src}&-c={ra}%20{dec}&-c.rs={r}&{cols}&-out.add=_r&-sort=_r&-out.max=60{extra}"
        L = [l for l in urllib.request.urlopen(url, timeout=60).read().decode().splitlines() if l and not l.startswith("#")]
        if len(L) < 3: return []
        h = L[0].split("\t"); return [dict(zip(h, [x.strip() for x in l.split("\t")])) for l in L[3:]]
    G = vz("I/355/gaiadr3", o["ra"], o["dec"], 90, "-out=Source,RA_ICRS,DE_ICRS,Gmag,BP-RP,Plx,RUWE")
    tgt = G[0]; T = SkyCoord(float(tgt["RA_ICRS"]) * u.deg, float(tgt["DE_ICRS"]) * u.deg)
    say(f"TIC {TIC}: P={P:.5f} d, depth {s['depth_ppm']:.0f} ppm, dur {durh:.2f} h (expected <= {s['dur_expected_h']:.2f} h), SNR {s['snr']:.1f}, SDE {s['sde']:.1f}")
    say(f"target Gaia DR3 {tgt['Source']}: G={tgt['Gmag']} BP-RP={tgt['BP-RP']} plx={tgt['Plx']} RUWE={tgt['RUWE']}" + ("  <-- RUWE>1.4: likely unresolved binary" if tgt["RUWE"] and float(tgt["RUWE"]) > 1.4 else ""))
    G15 = [g for g in G[1:] if g["Gmag"] and float(g["Gmag"]) < float(tgt["Gmag"]) + 4]
    say(f"neighbours within 90\" brighter than G+4 ({float(tgt['Gmag'])+4:.1f}): {len(G15)}")
    try:
        ex = json.load(urllib.request.urlopen(f"https://exofop.ipac.caltech.edu/tess/target.php?id={TIC}&json", timeout=60))
        say(f"ExoFOP: TOIs={len(ex.get('tois', []))} CTOIs={len(ex.get('ctois', []))} planets='{ex.get('basic_info', {}).get('confirmed_planets', '')}'")
    except Exception as e: say(f"ExoFOP check failed: {e}")
    for name, src in [("VSX", "B/vsx/vsx"), ("Gaia variability", "I/358/vclassre"), ("ZTF periodic", "J/ApJS/249/18/table2")]:
        try: say(f"{name} within 2.5': {len(vz(src, o['ra'], o['dec'], 150))}")
        except Exception as e: say(f"{name} check failed: {e}")

    from search import static_download
    from wotan import flatten
    dd = f"cand/lc{TIC}"; os.makedirs(dd, exist_ok=True); lcs = []
    for sec in o["sectors"]:
        fn = f"{dd}/hlsp_tess-spoc_tess_phot_{TIC:016d}-s{sec:04d}_tess_v1_lc.fits"
        if not os.path.exists(fn): fn = static_download(TIC, sec, dd)
        if fn: lcs.append(lk.read(fn, quality_bitmask="default").remove_nans().normalize())
    dur = durh / 24
    for win in (0.3, 1.0, None):
        Tt, Ff = [], []
        for lc in lcs:
            t = np.asarray(lc.time.value, float); f = np.ma.filled(np.asarray(lc.flux.value, float), np.nan)
            fl = f / np.nanmedian(f) if win is None else flatten(t, f, method="biweight", window_length=win, break_tolerance=0.3)
            g = np.isfinite(fl); Tt.append(t[g]); Ff.append(fl[g])
        t = np.concatenate(Tt); f = np.concatenate(Ff); ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
        intr = np.abs(ph) < dur / 3; far = (np.abs(ph) > 3 * dur) & (np.abs(ph) < 8 * dur)
        base = np.median(f[far]); sd = 1.4826 * np.median(np.abs(f[far] - base))
        say(f"detrend window {str(win) + ' d' if win else 'none (raw)':>12}: depth {(base - np.mean(f[intr])) * 1e6:6.0f} ± {sd / np.sqrt(intr.sum()) * 1e6:4.0f} ppm")

    secs = sorted(o["sectors"], reverse=True)[:12]
    with get_context("spawn").Pool(6) as p: res = [r for r in p.map(one, secs) if "err" not in r]
    sig = 15.0; grid = np.arange(-90, 91, 2.0)
    pix = []
    for r in res:
        c = SkyCoord(r["ra"].ravel() * u.deg, r["dec"].ravel() * u.deg); dx, dy = (a.arcsec for a in T.spherical_offsets_to(c))
        pix.append((dx, dy, r["D"].ravel(), r["E"].ravel()))
    def scan(pix):
        S = np.zeros((len(grid), len(grid)))
        for dx, dy, D, E in pix:
            ok = np.hypot(dx, dy) < 110; w = 1 / E[ok] ** 2
            for i, gy in enumerate(grid):
                m = np.exp(-((dx[ok][None] - grid[:, None]) ** 2 + (dy[ok][None] - gy) ** 2) / (2 * sig ** 2))
                num = (D[ok] * w * m).sum(1); den = (m * m * w).sum(1); S[i] += num ** 2 / den * np.sign(num)
        return S
    S = scan(pix); k = np.unravel_index(np.argmax(S), S.shape); bx, by, bS = grid[k[1]], grid[k[0]], S[k]
    def at(c):
        dx, dy = (a.arcsec for a in T.spherical_offsets_to(c)); return S[np.argmin(abs(grid - dy)), np.argmin(abs(grid - dx))], dx, dy
    rng = np.random.default_rng(0); bs = []
    for _ in range(25):
        Sb = scan([pix[i] for i in rng.integers(0, len(pix), len(pix))]); kb = np.unravel_index(np.argmax(Sb), Sb.shape); bs.append((grid[kb[1]], grid[kb[0]]))
    bs = np.array(bs); ex_, ey_ = max(bs[:, 0].std(), 3), max(bs[:, 1].std(), 3)
    st = at(T)[0]
    say(f"PIXEL TEST ({len(res)} sectors): dip source at ({bx:.0f}\", {by:.0f}\") ± ({ex_:.0f}, {ey_:.0f})\" from target; detection Δχ²={bS:.0f}; on-target Δχ²={st:.0f}")
    best_src = []
    for g in G:
        c = SkyCoord(float(g["RA_ICRS"]) * u.deg, float(g["DE_ICRS"]) * u.deg); v, dx, dy = at(c)
        best_src.append((bS - v, g["Source"], g["Gmag"], g["BP-RP"], dx, dy))
    best_src.sort()
    for d_, src, gm, br, dx, dy in best_src[:4]:
        say(f"   Gaia {src} G={gm} BP-RP={br} at ({dx:.0f}\", {dy:.0f}\"): Δχ² worse than best by {d_:.0f}" + ("   <-- TARGET" if src == tgt["Source"] else ""))
    # read by cand_evidence.py
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    gx = [(a.arcsec for a in T.spherical_offsets_to(SkyCoord(float(g["RA_ICRS"]) * u.deg, float(g["DE_ICRS"]) * u.deg))) for g in G]
    gx = np.array([tuple(v) for v in gx]); gm = np.array([float(g["Gmag"]) if g["Gmag"] else 21 for g in G])
    np.savez(f"cand/pix_{TIC}.npz", grid=grid, S=S, best=(bx, by), err=(ex_, ey_), gaia_dx=gx[:, 0], gaia_dy=gx[:, 1], gaia_G=gm,
             gaia_src=np.array([g["Source"] for g in G]), D=np.array([p_[2] for p_ in pix], dtype=object), Dx=np.array([p_[0] for p_ in pix], dtype=object), Dy=np.array([p_[1] for p_ in pix], dtype=object), allow_pickle=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(np.clip(S, 0, None), origin="lower", extent=[grid[0] - 1, grid[-1] + 1, grid[0] - 1, grid[-1] + 1], cmap="magma")
    plt.colorbar(im, label="Δχ²")
    sz = np.clip(4 * (21 - gm), 4, 80); vis = (np.abs(gx[:, 0]) < 90) & (np.abs(gx[:, 1]) < 90)
    ax.scatter(gx[vis, 0], gx[vis, 1], s=sz[vis], facecolors="none", edgecolors="cyan", lw=1, label="Gaia DR3 stars (size ∝ brightness)")
    ax.plot(0, 0, "*", ms=16, mfc="none", mec="lime", mew=2, label=f"target TIC {TIC}")
    ax.errorbar(bx, by, xerr=ex_, yerr=ey_, fmt="x", c="white", ms=10, mew=2, capsize=3, label="best-fit dip source")
    ax.set_xlabel("ΔRA [arcsec]"); ax.set_ylabel("ΔDec [arcsec]"); ax.set_title(f"TIC {TIC}: difference-image localization ({len(res)} sectors)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.8); plt.tight_layout(); plt.savefig(f"cand/pix_{TIC}.png", dpi=90); plt.close()
    on_target = (bS - st) < 9 or np.hypot(bx / ex_, by / ey_) < 3
    verdict = "ON TARGET (consistent with the target star)" if on_target else "OFF TARGET -> blended background source (false positive)"
    if bS < 25: verdict = "INCONCLUSIVE (dip not detected in pixels)"
    say(f"VERDICT: {verdict}")
    json.dump(REP, open(f"cand/vet_{TIC}.json", "w"), indent=1)

if __name__ == '__main__':
    main()
