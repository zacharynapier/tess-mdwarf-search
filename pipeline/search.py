"""Resumable transit search: CPU workers download and detrend, the GPU runs iterative BLS.
usage: search.py targets.csv OUTDIR [NWORKERS] [LIMIT]  (SHARD=i/n splits the list)"""
import os, sys, json, time, shutil, warnings, traceback
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def period_grid(T, pmin=0.5, pmax=15.0, rho_dur=0.03, os_=2.0):
    f, fs = 1 / pmax, []
    while f < 1 / pmin:
        fs.append(f); P = 1 / f; f += rho_dur * P ** (1 / 3) / (os_ * P * T)
    return 1 / np.array(fs)[::-1]


SECTORS = json.load(open(os.path.join(HERE, "sector_map.json")))


def static_download(tic, sec, dd, kind="lc"):
    """Download a TESS-SPOC HLSP file from the static archive server (no MAST API rate limit)."""
    import urllib.request, random
    t16 = f"{tic:016d}"
    name = f"hlsp_tess-spoc_tess_phot_{t16}-s{sec:04d}_tess_v1_{kind}.fits"
    url = f"https://archive.stsci.edu/hlsps/tess-spoc/s{sec:04d}/target/{t16[0:4]}/{t16[4:8]}/{t16[8:12]}/{t16[12:16]}/{name}"
    fn = f"{dd}/{name}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(fn, "wb") as fo: fo.write(r.read())
            return fn
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(min(120, 5 * 2 ** attempt) * (0.5 + random.random()))
        except Exception:
            time.sleep(min(120, 5 * 2 ** attempt) * (0.5 + random.random()))
    return None


def prepare(row):
    """CPU worker: download, detrend and bin one star."""
    import lightkurve as lk
    from wotan import flatten
    from astropy.timeseries import LombScargle
    tic, rad, mass = int(row[0]), float(row[1]), float(row[2])
    rad = rad if np.isfinite(rad) else 0.4; mass = mass if np.isfinite(mass) else 0.4
    dd = f"/tmp/exo_dl/{tic}"
    try:
        lcs = []
        os.makedirs(dd, exist_ok=True)
        for sec in SECTORS.get(str(tic), []):
            fn = static_download(tic, sec, dd)
            if fn: lcs.append(lk.read(fn, quality_bitmask="default"))
        if not lcs: raise RuntimeError("no light curves downloaded")
        RA, DEC = lcs[0].meta["RA_OBJ"], lcs[0].meta["DEC_OBJ"]
        raw = [lc.remove_nans().normalize() for lc in lcs]
        raw = [lc for lc in raw if len(lc) > 400]
        sectors = sorted({int(lc.meta["SECTOR"]) for lc in raw})
        best = max(raw, key=len); tt0 = np.asarray(best.time.value, float); ff0 = np.ma.filled(np.asarray(best.flux.value, float), np.nan)
        g = np.isfinite(ff0)
        fr, pw0 = LombScargle(tt0[g], ff0[g]).autopower(minimum_frequency=1 / 13, maximum_frequency=1 / 0.1, samples_per_peak=10)
        prot = float(1 / fr[np.argmax(pw0)])
        win = 0.3 if prot > 2 else max(0.15, prot / 6)
        T_, F_, E_ = [], [], []
        for lc in raw:
            tt = np.asarray(lc.time.value, float); ff = np.ma.filled(np.asarray(lc.flux.value, float), np.nan)
            fl = flatten(tt, ff, method="biweight", window_length=win, break_tolerance=0.3, edge_cutoff=0.0)
            g = np.isfinite(fl); tt, fl = tt[g], fl[g]
            if len(tt) < 100: continue
            med, sd = np.median(fl), 1.4826 * np.median(np.abs(fl - np.median(fl)))
            g = fl < med + 3 * sd; tt, fl = tt[g], fl[g]
            b = np.floor((tt - tt[0]) / (10 / 1440)).astype(np.int64); _, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
            tb = np.bincount(inv, tt) / cnt; fb = np.bincount(inv, fl) / cnt
            T_.append(tb); F_.append(fb); E_.append(np.full_like(fb, 1.4826 * np.median(np.abs(fb - np.median(fb)))))
        t = np.concatenate(T_); f = np.concatenate(F_); e = np.concatenate(E_); o = np.argsort(t)
        return dict(tic=tic, rad=rad, mass=mass, ra=RA, dec=DEC, prot=prot, detrend_win=win, sectors=sectors,
                    t=t[o], f=f[o], e=e[o])
    except Exception as ex:
        return dict(tic=tic, error=repr(ex)[:300])
    finally:
        shutil.rmtree(dd, ignore_errors=True)


def analyze(d, outdir, nsig=3):
    """GPU side: iterative BLS + vetting stats."""
    from astropy.timeseries import BoxLeastSquares
    from scipy.ndimage import median_filter
    from gpu_bls import bls_power
    from known import check
    t, f, e = d.pop("t"), d.pop("f"), d.pop("e")
    rad, mass, tic = d["rad"], d["mass"], d["tic"]
    T = t.max() - t.min(); periods = period_grid(T)
    a15 = (6.674e-11 * mass * 1.989e30 * (15 * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / (rad * 6.957e8)
    durs = np.geomspace(0.017, min(0.15, 1.3 * 15 / np.pi / a15, d["detrend_win"] / 2.5), 6)
    out = dict(d, nsectors=len(d["sectors"]), npts=len(t), baseline=T, nperiods=len(periods), signals=[])
    mask = np.zeros(len(t), bool)
    for k in range(nsig):
        tt, ff, ee = t[~mask], f[~mask], e[~mask]; ww = 1 / ee ** 2
        r = bls_power(tt, ff, ee, periods, durs)
        pw = r["power"]; base = median_filter(pw, 2001, mode="nearest"); resid = pw - base; sde = resid / np.std(resid)
        i = int(np.argmax(sde)); P, t0, dur, dep = periods[i], r["transit_time"][i], r["duration"][i], r["depth"][i]
        m = BoxLeastSquares(tt, ff, ee)
        try:
            st = m.compute_stats(P, dur, t0)
        except ValueError:                        # no in-transit points left
            break
        intr = m.transit_mask(tt, P, dur, t0)
        ntr = int(sum(np.any(np.abs(tt - x) < dur / 2) for x in st["transit_times"]))
        mid = np.median(tt); h1 = intr & (tt < mid); h2 = intr & (tt >= mid)
        d1 = 1 - np.mean(ff[h1]) if h1.sum() else np.nan; d2 = 1 - np.mean(ff[h2]) if h2.sum() else np.nan
        a_rs = (6.674e-11 * mass * 1.989e30 * (P * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / (rad * 6.957e8)
        prot = d["prot"]
        s = dict(rank=k + 1, period=float(P), t0=float(t0), dur_h=float(dur * 24), depth_ppm=float(dep * 1e6),
                 snr=float(dep * np.sqrt(np.sum(ww[intr]))), sde=float(sde[i]), bls_snr=float(pw[i]), ntransits_obs=ntr,
                 depth_odd_ppm=float(st["depth_odd"][0] * 1e6), depth_even_ppm=float(st["depth_even"][0] * 1e6),
                 odd_even_sig=float(abs(st["depth_odd"][0] - st["depth_even"][0]) / np.hypot(st["depth_odd"][1], st["depth_even"][1])),
                 secondary_ppm=float(st["depth_phased"][0] * 1e6), secondary_sig=float(st["depth_phased"][0] / st["depth_phased"][1]),
                 depth_half1_ppm=float(d1 * 1e6), depth_half2_ppm=float(d2 * 1e6),
                 dur_expected_h=float(P / np.pi / a_rs * 24), rp_earth=float(np.sqrt(max(dep, 0)) * rad * 109.1),
                 rot_harmonic=next((f"{a}/{c}" for a in range(1, 5) for c in range(1, 5) if abs(P * c / (prot * a) - 1) < 0.01), None))
        kn = check(d["ra"], d["dec"], tic=tic, period=P); s["known_match"] = kn[kn.harmonic.notna()].name.tolist() if len(kn) else []
        out["signals"].append(s)
        mask |= m.transit_mask(t, P, max(dur * 1.5, 0.04), t0)
        if sde[i] < 9: break                      # below the triage SDE cut
    np.save(f"{outdir}/{tic}_lc.npy", np.vstack([t, f]).astype(np.float32))
    return out


if __name__ == "__main__":
    targets, outdir = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 8; limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(targets)
    if limit: df = df.head(limit)
    shard, nshard = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
    df = df.iloc[shard::nshard]
    done = {int(x.split(".")[0]) for x in os.listdir(outdir) if x.endswith(".json")}
    rows = [r for r in zip(df.ID, df.rad, df["mass"]) if int(r[0]) not in done]
    print(f"{len(rows)} to do ({len(done)} already done)", flush=True)
    t_start = time.time(); n = 0
    with Pool(nw) as pool:
        for d in pool.imap_unordered(prepare, rows):
            tic = d["tic"]; t1 = time.time()
            try:
                out = d if "error" in d else analyze(d, outdir)
            except Exception as ex:
                out = dict(tic=tic, error="analyze: " + repr(ex)[:300]); traceback.print_exc()
            out["gpu_s"] = time.time() - t1
            json.dump(out, open(f"{outdir}/{tic}.json", "w"), default=float)
            n += 1
            if n % 25 == 0 or n == len(rows):
                el = time.time() - t_start
                print(f"[{n}/{len(rows)}] {el/60:.1f} min elapsed, {n/el*3600:.0f} stars/h, last gpu {out['gpu_s']:.1f}s", flush=True)
    print("ALL DONE", flush=True)
