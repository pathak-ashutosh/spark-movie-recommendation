#!/usr/bin/env python3
"""
Offline precompute for the CineMatch web UI.

This mirrors the item-item collaborative-filtering stage of
``movie_recommender_system.py`` (the PySpark project) but runs locally with
plain Python + numpy so it needs no Spark cluster or HDFS.

For every movie it computes the Pearson correlation to every other movie over
the users who rated *both* (mean-centering each movie's ratings first, exactly
like the Spark ``compute_similarity`` function), keeps the strongest neighbours,
and writes a single self-contained ``data.js`` that the browser app loads.

    Spark version                     ->  here
    ---------------------------------     ------------------------------------
    norm_rating = rating - mean_rating -> mean-centre each movie's ratings
    sum_xy / (sqrt(sum_xx)*sqrt(sum_yy)) -> Pearson over co-rating users
    (no min co-raters)                 -> MIN_COMMON quality gate (better recs)

Run:  python3 precompute.py
Output: ./data.js   (window.MOVIE_DATA = {...})
"""

import csv
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

import numpy as np

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "ml-latest-small")
OUT_FILE = os.path.join(HERE, "data.js")
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

MIN_RATINGS_CATALOG = 3   # a movie needs this many ratings to appear in search
MIN_RATINGS_SIM = 8       # ...and this many to get computed neighbours
MIN_COMMON = 8            # two movies need this many shared raters for a valid sim
SHRINK_LAMBDA = 25        # significance weighting: sim *= common/(common+lambda)
MIN_SIM = 0.05            # neighbours weaker than this (after shrinkage) are dropped
TOP_NEIGHBOURS = 30       # neighbours kept per movie


def ensure_dataset():
    """Use a local copy of ml-latest-small if present, else download it."""
    if os.path.exists(os.path.join(DATA_DIR, "ratings.csv")):
        return
    print(f"Dataset not found locally; downloading from {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL) as resp:  # noqa: S310 (trusted host)
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if name.startswith("ml-latest-small/") and name.endswith(".csv"):
                target = os.path.join(HERE, name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(zf.read(name))
    print("Download complete.")


# ---------------------------------------------------------------------------
# Load + clean movie metadata
# ---------------------------------------------------------------------------
_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")
# "Matrix, The" / "Godfather, The" -> "The Matrix" for nicer display
_ARTICLE_RE = re.compile(r"^(.*),\s+(The|A|An|La|Le|Les|Il|El|Das|Der|Die)$")


def clean_title(raw):
    year = None
    m = _YEAR_RE.search(raw)
    if m:
        year = int(m.group(1))
        raw = _YEAR_RE.sub("", raw)
    raw = raw.strip()
    m = _ARTICLE_RE.match(raw)
    if m:
        raw = f"{m.group(2)} {m.group(1)}"
    return raw, year


def load_movies():
    movies = {}  # movieId -> dict
    with open(os.path.join(DATA_DIR, "movies.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mid = int(row["movieId"])
            title, year = clean_title(row["title"])
            raw_genres = row["genres"].split("|")
            genres = [g for g in raw_genres if g and g != "(no genres listed)"]
            movies[mid] = {"id": mid, "title": title, "year": year, "genres": genres}
    return movies


def load_links():
    links = {}
    path = os.path.join(DATA_DIR, "links.csv")
    if not os.path.exists(path):
        return links
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mid = int(row["movieId"])
            links[mid] = {
                "imdb": row.get("imdbId") or "",
                "tmdb": int(row["tmdbId"]) if row.get("tmdbId") else None,
            }
    return links


def load_ratings():
    rows = []  # (userId, movieId, rating)
    with open(os.path.join(DATA_DIR, "ratings.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((int(row["userId"]), int(row["movieId"]), float(row["rating"])))
    return rows


# ---------------------------------------------------------------------------
# Item-item Pearson CF  (vectorised equivalent of the Spark compute_similarity)
# ---------------------------------------------------------------------------
def compute_neighbours(ratings, movie_stats):
    """Return {movieId: [[neighbourId, sim], ...]} for movies with enough data."""
    sim_ids = sorted(m for m, s in movie_stats.items() if s["n"] >= MIN_RATINGS_SIM)
    user_ids = sorted({u for (u, _m, _r) in ratings})
    row_of = {m: i for i, m in enumerate(sim_ids)}
    col_of = {u: j for j, u in enumerate(user_ids)}
    print(f"  matrix: {len(sim_ids)} movies x {len(user_ids)} users")

    R = np.zeros((len(sim_ids), len(user_ids)), dtype=np.float32)  # mean-centred
    M = np.zeros((len(sim_ids), len(user_ids)), dtype=np.float32)  # rated mask
    means = {m: movie_stats[m]["mean"] for m in sim_ids}
    for (u, m, r) in ratings:
        i = row_of.get(m)
        if i is None:
            continue
        j = col_of[u]
        R[i, j] = r - means[m]
        M[i, j] = 1.0

    # numerator[i,j] = sum over co-raters of centred_i * centred_j  (0s cancel)
    N = R @ R.T
    Rsq = R * R
    # sum of squares taken only over the *co-rated* users, per the Spark version
    SXX = Rsq @ M.T                 # SXX[i,j] = sum_u centred_i^2 * rated_j
    denom = np.sqrt(SXX) * np.sqrt(SXX.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom > 0, N / denom, 0.0)

    common = (M @ M.T)             # shared-rater counts
    sim = np.clip(sim, -1.0, 1.0)

    # Significance weighting: a raw Pearson of 0.97 from only ~5 shared raters is
    # noise, not signal. Shrink each similarity toward 0 in proportion to how few
    # users co-rated the pair, so genuinely-similar popular movies (hundreds of
    # co-raters, more honest correlation) rank above lucky low-overlap pairs.
    shrink = common / (common + SHRINK_LAMBDA)
    sim = sim * shrink
    sim[common < MIN_COMMON] = 0.0
    np.fill_diagonal(sim, 0.0)

    neighbours = {}
    for i, mid in enumerate(sim_ids):
        row = sim[i]
        # candidate indices with a meaningful positive similarity
        cand = np.where(row >= MIN_SIM)[0]
        if cand.size == 0:
            continue
        cand = cand[np.argsort(row[cand])[::-1][:TOP_NEIGHBOURS]]
        neighbours[mid] = [[int(sim_ids[j]), round(float(row[j]), 4)] for j in cand]
    return neighbours


# ---------------------------------------------------------------------------
def main():
    ensure_dataset()
    print("Loading data ...")
    movies = load_movies()
    links = load_links()
    ratings = load_ratings()
    print(f"  {len(movies)} movies, {len(ratings)} ratings")

    # per-movie popularity + mean rating
    stats = {}  # movieId -> {n, sum}
    for (_u, m, r) in ratings:
        s = stats.setdefault(m, {"n": 0, "sum": 0.0})
        s["n"] += 1
        s["sum"] += r
    for m, s in stats.items():
        s["mean"] = s["sum"] / s["n"]

    print("Computing item-item Pearson similarities ...")
    neighbours = compute_neighbours(ratings, stats)
    print(f"  neighbour lists for {len(neighbours)} movies")

    # Assemble catalog (movies with enough ratings to be worth showing)
    genre_set = sorted({g for m in movies.values() for g in m["genres"]})
    genre_idx = {g: i for i, g in enumerate(genre_set)}

    catalog = []
    for mid, mv in movies.items():
        s = stats.get(mid)
        if not s or s["n"] < MIN_RATINGS_CATALOG:
            continue
        lk = links.get(mid, {})
        entry = {
            "id": mid,
            "t": mv["title"],
            "y": mv["year"] or 0,
            "g": [genre_idx[g] for g in mv["genres"]],
            "n": s["n"],
            "r": round(s["mean"], 1),
        }
        if lk.get("tmdb"):
            entry["tmdb"] = lk["tmdb"]
        if lk.get("imdb"):
            entry["imdb"] = lk["imdb"]
        catalog.append(entry)

    catalog.sort(key=lambda e: e["n"], reverse=True)  # popular first

    # keep only neighbour entries whose movies are in the catalog
    catalog_ids = {e["id"] for e in catalog}
    nb = {
        str(mid): [pair for pair in lst if pair[0] in catalog_ids]
        for mid, lst in neighbours.items()
        if mid in catalog_ids
    }
    nb = {k: v for k, v in nb.items() if v}

    payload = {
        "genres": genre_set,
        "movies": catalog,
        "nb": nb,
        "meta": {
            "source": "MovieLens latest-small (GroupLens)",
            "movies": len(catalog),
            "withNeighbours": len(nb),
            "minCommon": MIN_COMMON,
        },
    }

    js = "// Auto-generated by precompute.py — do not edit by hand.\n"
    js += "window.MOVIE_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(js)

    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"Wrote {OUT_FILE}  ({size_mb:.2f} MB)")
    print(f"  catalog: {len(catalog)} movies, {len(genre_set)} genres, "
          f"{len(nb)} with neighbours")


if __name__ == "__main__":
    sys.exit(main())
