# tess-mdwarf-search

A GPU transit search of 15,512 faint, nearby M dwarfs in TESS full-frame-image light curves, with automatic triage and
pixel-level vetting.

## Results

Pending. A Research Note is under review, and planet candidates are being submitted to ExoFOP. Results and the
follow-up analysis will be added here once they are public.

## How it works

The search ran on 2× H100 (Lambda Cloud) in about 2 hours, for under $20 of cloud time.

- [`gpu_bls.py`](pipeline/gpu_bls.py) runs box least squares over thousands of trial periods at once.
  - Each period's light curve is folded into 2,000 phase bins with `index_add_`.
  - Every box duration is scored with a circular cumulative-sum window.
  - On an H100 this takes a median of 0.7 s per star: 330,000–450,000 periods × 6 durations, up to three iterations.
  - A single pass of astropy's `BoxLeastSquares` over the same grid takes about 46 s on one CPU core. Both find the
    same peak ([`check_gpu_bls.py`](pipeline/check_gpu_bls.py)).
- [`search.py`](pipeline/search.py) is the search driver.
  - CPU workers download and detrend each light curve: a per-sector biweight filter with a rotation-aware window,
    then 10-minute bins.
  - Meanwhile the GPU searches 0.5–15 d and records odd/even, secondary-eclipse, duration, rotation and
    catalog-match statistics for each signal.
  - Runs can be resumed, and are split across GPUs by [`run_gpu_shards.sh`](pipeline/run_gpu_shards.sh).
- [`triage.py`](pipeline/triage.py) applies the automatic cuts.
- [`vet.py`](pipeline/vet.py) vets one signal at a time.
  - It checks the catalogs and re-measures the depth with different detrending.
  - It localizes the signal in the pixels by fitting difference images with a PSF at every Gaia position. This
    places LHS 3844 b and TOI-5205 b on their host stars ([`positive_control.py`](pipeline/positive_control.py)).
- For follow-up:
  - [`fit_transit.py`](pipeline/fit_transit.py) fits the transit (batman + emcee, free stellar density).
  - [`validate.py`](pipeline/validate.py) runs TRICERATOPS.
  - [`ctoi_params.py`](pipeline/ctoi_params.py) produces the parameters for an ExoFOP CTOI.

## Targets

The targets are TIC M dwarfs with:
- T = 13–15.5, d < 100 pc, T_eff 2700–3800 K, log g 4.6–5.6;
- |b| > 10°, outside the Magellanic Clouds, and contamination ratio < 0.5;
- at least 4 sectors of TESS-SPOC FFI light curves (Caldwell et al. 2020);
- no 2-minute data, and no known planet, TOI, CTOI or eclipsing binary within 1.5′.

The list is [`pipeline/allsky_targets.csv`](pipeline/allsky_targets.csv), built by
[`allsky_targets.py`](pipeline/allsky_targets.py) and [`sector_map.py`](pipeline/sector_map.py).

## Running

```
pip install -r requirements.txt
cd pipeline
./fetch_catalogs.sh                              # TOI, CTOI, planet and EB catalogs for the cross-match
python search.py allsky_targets.csv results 24   # or ./run_gpu_shards.sh on a multi-GPU machine
RESDIR=results python triage.py
python vet.py <TIC>                              # after copying results/<TIC>.json and _lc.npy to cand/
```

These are research scripts, not a package. Each script runs from a working directory that holds its inputs. The
follow-up scripts find per-candidate data folders through the `EXO_ROOT` environment variable.

## Known limitations

- The period search starts at 0.5 d, so ultra-short-period planets appear at twice their period.
- Periods ≳ 12 d with only 3–6 transits sit near TESS's 13.7-day orbital systematics and are low-confidence.
- The pixel test cannot separate stars closer than about 5–10″.
- Search-stage radii come from box depths and TIC stellar radii, so they are good to about 20–30%.

## Acknowledgments

This work uses TESS data from the Mikulski Archive for Space Telescopes (MAST), Gaia data from ESA, and the packages
`astropy`, `astroquery`, `lightkurve`, `wotan`, PyTorch, `batman`, `emcee` and `triceratops`.

## License

MIT; see [LICENSE](LICENSE).
