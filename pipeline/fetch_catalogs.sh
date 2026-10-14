#!/bin/sh
# catalogs for known.py (snapshot used: 2026-10-11): ExoFOP TOIs/CTOIs, NASA Exoplanet Archive, TESS EBs (Prša+2022)
cd "$(dirname "$0")" && mkdir -p cat
curl -sL "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv" -o cat/toi.csv
curl -sL "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv" -o cat/ctoi.csv
curl -sL "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,hostname,tic_id,ra,dec,pl_orbper+from+pscomppars&format=csv" -o cat/ps.csv
curl -sL "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/258/16/tess-ebs&-out=TIC,RAJ2000,DEJ2000,Per&-out.max=unlimited" -o cat/tessebs.tsv
