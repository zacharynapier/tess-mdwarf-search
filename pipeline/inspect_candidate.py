"""Diagnostic plots for one signal: fold, odd/even, depth per season and per transit.
usage: inspect_candidate.py DIR TIC RANK"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d, tic, rank = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
o = json.load(open(f"{d}/{tic}.json")); s = o["signals"][rank-1]; t, f = np.load(f"{d}/{tic}_lc.npy").astype(float)
P, t0, dur = s["period"], s["t0"], s["dur_h"]/24; ph = ((t - t0 + 0.5*P) % P) - 0.5*P
def binned(x, y, w):
    e = np.arange(x.min(), x.max()+w, w); i = np.digitize(x, e)
    return np.array([(x[i==k].mean(), np.mean(y[i==k]), y[i==k].std()/np.sqrt((i==k).sum())) for k in np.unique(i) if (i==k).sum() > 5]).T
fig, ax = plt.subplots(3, 2, figsize=(14, 12))
x, y, e = binned(ph*24, f, 0.25); m = np.abs(x) < 10
ax[0,0].errorbar(x[m], (y[m]-1)*1e6, e[m]*1e6, fmt='.k'); ax[0,0].set_title(f"TIC {tic}  P={P:.5f} d  depth={s['depth_ppm']:.0f} ppm  dur={dur*24:.2f} h (exp {s['dur_expected_h']:.2f})"); ax[0,0].set_xlabel("hours from mid-transit"); ax[0,0].axhline(0, c='gray', lw=.5)
x, y, e = binned(ph/P, f, 0.005); ax[0,1].errorbar(x, (y-1)*1e6, e*1e6, fmt='.k', ms=2); ax[0,1].set_title("full phase"); ax[0,1].set_xlabel("phase")
n = np.round((t - t0)/P).astype(int)
for par, c in [(0, 'C0'), (1, 'C3')]:
    mm = (n % 2 == par) & (np.abs(ph) < 0.4); x, y, e = binned(ph[mm]*24, f[mm], 0.3); ax[1,0].errorbar(x, (y-1)*1e6, e*1e6, fmt='.', c=c, label=['even','odd'][par])
ax[1,0].legend(); ax[1,0].set_xlim(-8, 8); ax[1,0].set_title("odd vs even transits")
yrs = [(t.min(), 1700), (1700, 2200), (2200, 2700), (2700, 3300), (3300, 5000)]
intr = np.abs(ph) < dur/2.5; oot = (np.abs(ph) > dur) & (np.abs(ph) < 3*dur)
for a, b in yrs:
    m1 = (t >= a) & (t < b)
    if (m1 & intr).sum() > 10: ax[1,1].errorbar(np.median(t[m1]), (np.mean(f[m1 & oot]) - np.mean(f[m1 & intr]))*1e6, np.std(f[m1 & oot])/np.sqrt((m1 & intr).sum())*1e6, fmt='o')
ax[1,1].axhline(s['depth_ppm'], c='r', ls='--'); ax[1,1].set_title("depth vs time (per observing season)")
ep = np.unique(n[intr]); dd = []
for k in ep:
    mi = (n == k) & intr; mo = (n == k) & oot
    if mi.sum() >= 3 and mo.sum() >= 5: dd.append((k, (np.mean(f[mo]) - np.mean(f[mi]))*1e6, np.std(f[mo])/np.sqrt(mi.sum())*1e6))
dd = np.array(dd); ax[2,0].errorbar(dd[:,0], dd[:,1], dd[:,2], fmt='.k'); ax[2,0].axhline(s['depth_ppm'], c='r', ls='--'); ax[2,0].axhline(0, c='gray'); ax[2,0].set_title(f"each transit's depth ({len(dd)} epochs)"); ax[2,0].set_xlabel("epoch")
ax[2,1].hist(dd[:,1]/dd[:,2], 30); ax[2,1].set_title("per-transit depth / error")
plt.tight_layout(); plt.savefig(f"cand_{tic}.png", dpi=75)
w = dd[:,1]/dd[:,2]**2; tot = np.sum(w)/np.sum(1/dd[:,2]**2)
top = np.sort(dd[:,1]/dd[:,2])[::-1]
print(f"epochs {len(dd)}; weighted depth {tot:.0f} ppm; chi2/dof of per-transit depths vs const: {np.sum(((dd[:,1]-tot)/dd[:,2])**2)/(len(dd)-1):.2f}; top 5 individual SNRs {np.round(top[:5],1)}")
lo = dd[:,1]/dd[:,2]; drop = np.argsort(lo)[::-1][:5]; keep = np.setdiff1d(np.arange(len(dd)), drop)
print(f"without 5 strongest epochs: depth {np.sum(dd[keep,1]/dd[keep,2]**2)/np.sum(1/dd[keep,2]**2):.0f} ppm, SNR {np.sum(dd[keep,1]/dd[keep,2]**2)/np.sqrt(np.sum(1/dd[keep,2]**2)):.1f}")
