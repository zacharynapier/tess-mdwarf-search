"""CTOI parameters (16/50/84th percentiles) from a fit_transit.py chain.
usage: ctoi_params.py TIC [TIC ...]  (reads fit_TIC_chain.npy)"""
import os, sys, json, numpy as np

G = 6.674e-8
for tic in map(int, sys.argv[1:]):
    ch = np.load(f"fit_{tic}_chain.npy"); t0, P, k, b, lrho = ch.T; rho = 10 ** lrho
    a = (G * rho * (P * 86400) ** 2 / (3 * np.pi)) ** (1 / 3); inc = np.arccos(np.clip(b / a, -1, 1))
    arg = np.sqrt(np.clip((1 + k) ** 2 - b ** 2, 0, None)) / (a * np.sin(inc)); T14 = P / np.pi * np.arcsin(np.clip(arg, 0, 1)) * 24
    dep = k ** 2 * 1e6
    q = lambda x: np.percentile(x, [16, 50, 84])
    par = json.load(open(f"{os.environ.get('EXO_ROOT', '.')}/planet-candidates/TIC-{tic}/data/candidate_parameters.json"))
    R = par["rad"]; Rp = k * R * 109.1
    out = dict(tic=tic, period=q(P), epoch_bjd=q(t0 + 2457000), duration_h=q(T14), depth_ppm_k2=q(dep), depth_ppm_measured=[par["depth_ppm"], par["depth_err"]],
               rp_earth=q(Rp), rp_rs=q(k), b=q(b), rho=q(rho), a_rs=q(a), inc_deg=q(np.degrees(inc)))
    out = {kk: (list(map(float, v)) if isinstance(v, np.ndarray) else v) for kk, v in out.items()}
    json.dump(out, open(f"ctoi_params_{tic}.json", "w"), indent=1)
    fm = lambda v, d: f"{v[1]:.{d}f} +{v[2]-v[1]:.{d}f}/-{v[1]-v[0]:.{d}f}"
    print(f"TIC {tic}: P {fm(out['period'],6)} d | T0 {fm(out['epoch_bjd'],4)} | T14 {fm(out['duration_h'],2)} h | Rp {fm(out['rp_earth'],2)} Re | b {fm(out['b'],2)} | a/R* {fm(out['a_rs'],1)}")
