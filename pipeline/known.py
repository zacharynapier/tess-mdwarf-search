"""Cross-match a target (and optional period) against known planets, TOIs, CTOIs, and TESS EBs."""
import os, sys, numpy as np, pandas as pd
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cat') + '/'

def _load():
    rows = []
    t = pd.read_csv(D+'toi.csv')
    ra = Angle(t['RA'].astype(str), unit=u.hourangle).deg if not pd.api.types.is_numeric_dtype(t['RA']) else t['RA']
    dec = Angle(t['Dec'].astype(str), unit=u.deg).deg if not pd.api.types.is_numeric_dtype(t['Dec']) else t['Dec']
    rows.append(pd.DataFrame(dict(cat='TOI', name='TOI-'+t['TOI'].astype(str)+' '+t['TFOPWG Disposition'].fillna('').astype(str),
                                  tic=t['TIC ID'], ra=ra, dec=dec, per=t['Period (days)'])))
    c = pd.read_csv(D+'ctoi.csv')
    rows.append(pd.DataFrame(dict(cat='CTOI', name='CTOI-'+c['CTOI'].astype(str), tic=c['TIC ID'], ra=c['RA'], dec=c['Dec'], per=c['Period (days)'])))
    p = pd.read_csv(D+'ps.csv')
    rows.append(pd.DataFrame(dict(cat='PLANET', name=p['pl_name'], tic=p['tic_id'].str.replace('TIC ','').astype(float), ra=p['ra'], dec=p['dec'], per=p['pl_orbper'])))
    e = pd.read_csv(D+'tessebs.tsv', sep='\t', comment='#')
    e = e[pd.to_numeric(e['TIC'], errors='coerce').notna()]
    rows.append(pd.DataFrame(dict(cat='EB', name='TIC '+e['TIC'].astype(str).str.lstrip('0'), tic=e['TIC'].astype(float),
                                  ra=e['RAJ2000'].astype(float), dec=e['DEJ2000'].astype(float), per=pd.to_numeric(e['Per'], errors='coerce'))))
    return pd.concat(rows, ignore_index=True)
K = _load()
KC = SkyCoord(np.asarray(K.ra, float)*u.deg, np.asarray(K.dec, float)*u.deg)

def harmonic(p1, p2, tol=0.003):
    if not (np.isfinite(p1) and np.isfinite(p2)) or p2 <= 0: return None
    for a in range(1,6):
        for b in range(1,6):
            if abs(p1*b/(p2*a) - 1) < tol: return f"{a}/{b}"
    return None

def check(ra, dec, tic=None, period=None, radius_arcmin=2.5):
    sep = SkyCoord(ra*u.deg, dec*u.deg).separation(KC).arcmin
    m = K[(sep < radius_arcmin) | ((K.tic == tic) if tic else False)].copy()
    m['sep_arcmin'] = sep[m.index]
    if period: m['harmonic'] = [harmonic(period, p) for p in m.per]
    return m.sort_values('sep_arcmin')

if __name__ == '__main__':
    ra, dec = float(sys.argv[1]), float(sys.argv[2]); per = float(sys.argv[3]) if len(sys.argv) > 3 else None
    r = check(ra, dec, period=per)
    print(f"{len(K)} known objects loaded"); print(r.to_string() if len(r) else "NO KNOWN OBJECTS WITHIN 2.5'")
