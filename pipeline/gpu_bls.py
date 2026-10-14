"""Batched box least squares on GPU (CUDA or MPS); power matches astropy's objective="snr"."""
import numpy as np, torch


def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def bls_power(t, y, dy, periods, durations, nbins=2000, batch=4096, dev=None):
    """Returns dict of numpy arrays (power, depth, duration, transit_time) per period."""
    dev = dev or device()
    t = np.asarray(t, float); tref = 0.5 * (t.min() + t.max())
    w = 1.0 / np.asarray(dy, float) ** 2
    y = np.asarray(y, float); y = y - np.sum(w * y) / np.sum(w)      # weighted-mean-subtracted
    T = torch.tensor(t - tref, dtype=torch.float32, device=dev)     # centred: keeps fp32 phase error ~10 s
    WY = torch.tensor(w * y, dtype=torch.float32, device=dev)
    Wt = torch.tensor(w, dtype=torch.float32, device=dev)
    Wtot = float(w.sum()); minW = 3.0 * float(np.median(w))         # need >= ~3 points' weight inside a box
    durs = np.asarray(durations, float); periods = np.asarray(periods, float)
    kmax = int(np.ceil(durs.max() / periods.min() * nbins)) + 1
    out_p, out_d, out_q, out_t = (np.empty(len(periods)) for _ in range(4))
    N = len(t); ar = torch.arange(nbins, device=dev)
    for s in range(0, len(periods), batch):
        P = periods[s:s + batch]; B = len(P)
        Pt = torch.tensor(P, dtype=torch.float32, device=dev)
        x = T[None, :] * (1.0 / Pt)[:, None]                                         # cycles; fp32 err <~0.5 bin
        idx = ((x - torch.floor(x)) * nbins).to(torch.int32).clamp_(max=nbins - 1)
        idx += (torch.arange(B, device=dev, dtype=torch.int32) * nbins)[:, None]
        sy = torch.zeros(B * nbins, device=dev).index_add_(0, idx.reshape(-1), WY.expand(B, N).reshape(-1)).view(B, nbins)
        sw = torch.zeros(B * nbins, device=dev).index_add_(0, idx.reshape(-1), Wt.expand(B, N).reshape(-1)).view(B, nbins)
        k = min(kmax, nbins - 1)
        cy = torch.cat([torch.zeros(B, 1, device=dev), torch.cumsum(torch.cat([sy, sy[:, :k]], 1), 1)], 1)
        cw = torch.cat([torch.zeros(B, 1, device=dev), torch.cumsum(torch.cat([sw, sw[:, :k]], 1), 1)], 1)
        best = torch.full((B,), -np.inf, device=dev); bd = torch.zeros(B, device=dev)
        bq = torch.zeros(B, device=dev); bj = torch.zeros(B, dtype=torch.long, device=dev)
        for d in durs:
            kb = torch.clamp(torch.round(torch.tensor(d / P * nbins, dtype=torch.float32, device=dev)).long(), 1, k)  # (B,)
            valid = torch.tensor(d < 0.3 * P, device=dev)
            Yin = cy.gather(1, ar[None, :] + kb[:, None]) - cy[:, :nbins]
            Win = cw.gather(1, ar[None, :] + kb[:, None]) - cw[:, :nbins]
            Wout = Wtot - Win
            ok = (Win > minW) & (Wout > 0)
            depth = -Yin / torch.where(ok, Win, 1.0) * Wtot / torch.where(ok, Wout, 1.0)   # mean_out - mean_in
            snr = depth * torch.sqrt(torch.where(ok, Win * Wout / Wtot, 0.0))
            snr = torch.where(ok, snr, -np.inf)
            m, j = snr.max(1)
            better = (m > best) & valid
            best = torch.where(better, m, best); bj = torch.where(better, j, bj)
            bd = torch.where(better, depth.gather(1, j[:, None])[:, 0], bd)
            bq = torch.where(better, torch.full_like(bq, d), bq)
        sl = slice(s, s + B)
        out_p[sl] = best.cpu().numpy(); out_d[sl] = bd.cpu().numpy(); out_q[sl] = bq.cpu().numpy()
        out_t[sl] = tref + (bj.cpu().numpy() / nbins) * P + out_q[sl] / 2       # box start + half duration
    return dict(power=out_p, depth=out_d, duration=out_q, transit_time=out_t)
