"""Check the GPU BLS against astropy's BoxLeastSquares and time the full period grid.
usage: check_gpu_bls.py LC.npy  (rows: time, flux)"""
import sys, time, numpy as np, torch
from astropy.timeseries import BoxLeastSquares
from gpu_bls import bls_power, device
from search import period_grid


def sync():
    if device().type == "cuda": torch.cuda.synchronize()
    elif device().type == "mps": torch.mps.synchronize()


t, f = np.load(sys.argv[1]).astype(float); e = np.full_like(f, np.std(f))
durs = np.geomspace(0.017, 0.12, 6); P = period_grid(t.max() - t.min())
print(f"N={len(t)} points, {len(P)} periods, {len(durs)} durations")
sub = np.concatenate([P[(P > 0.60) & (P < 0.65)], P[(P > 3.0) & (P < 3.05)]])  # two slices of the grid
t0 = time.time(); a = BoxLeastSquares(t, f, e).power(sub, durs, objective="snr"); ta = time.time() - t0
t0 = time.time(); g = bls_power(t, f, e, sub, durs); sync(); tg = time.time() - t0
r = np.corrcoef(a.power, g["power"])[0, 1]
ia, ig = np.argmax(a.power), np.argmax(g["power"])
print(f"slice ({len(sub)} periods): astropy {ta:.1f}s  gpu {tg:.1f}s   corr={r:.4f}")
print(f"  astropy peak P={a.period[ia]:.6f} snr={a.power[ia]:.2f} dur={a.duration[ia]*24:.2f}h depth={a.depth[ia]*1e6:.0f}ppm t0={a.transit_time[ia]:.4f}")
print(f"  gpu     peak P={sub[ig]:.6f} snr={g['power'][ig]:.2f} dur={g['duration'][ig]*24:.2f}h depth={g['depth'][ig]*1e6:.0f}ppm t0={g['transit_time'][ig]:.4f}")
dt0 = ((g['transit_time'][ig] - a.transit_time[ia] + sub[ig]/2) % sub[ig]) - sub[ig]/2
print(f"  t0 difference: {dt0*24*60:.1f} min")
t0 = time.time(); G = bls_power(t, f, e, P, durs); sync(); print(f"full grid on {device().type}: {time.time()-t0:.1f}s   peak P={P[np.argmax(G['power'])]:.6f}")
