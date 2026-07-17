# 🎬 CineMatch — the movie recommender, made playable

A fun, zero-backend web UI on top of this project's recommender. Tap the films
you love; CineMatch finds what to watch next and tells you **why**.

**▶ Live demo: https://cinematch-peach.vercel.app**

It reuses the project's core idea — **item-item collaborative filtering with
Pearson similarity** — but precomputes it locally (no Spark/HDFS) and serves the
result entirely in the browser.

![flow](https://img.shields.io/badge/pick_films-→_get_matches-ffb43d?style=flat-square)

---

## Run it

**Option A — just open it**

Double-click `index.html`. That's it. All data is baked into `data.js`, so it
works straight from the filesystem (no server, no install).

**Option B — tiny static server** (nicer URLs, avoids any `file://` quirks)

```bash
cd webapp
python3 -m http.server 8899
# open http://localhost:8899
```

Nothing else to install — the page pulls its two fonts from Google Fonts and
degrades gracefully to system fonts offline.

---

## How to use

1. **Discover** — tap the films that stuck with you (search 5,000 titles or
   filter by genre). Each pick drops into the **filmstrip** at the bottom.
2. **For You** — see your ranked matches, each with a **match %** and a reason
   (*"Loved by fans of Forrest Gump & The Silence of the Lambs"*).
3. **Refine** — on any match hit **Love it** (adds it to your taste) or
   **Not for me** (steers away). Recommendations re-tune instantly.

Your picks are saved in `localStorage`, so a refresh keeps your reel.

---

## How it maps to the Spark model

`movie_recommender_system.py` builds a hybrid recommender (ALS + item-item CF +
a Random Forest on genres) on the MovieLens **25M** set via Spark. The item-item
CF stage computes, for every movie pair, the **Pearson correlation** of their
mean-centred ratings over the users who rated both.

`precompute.py` reproduces exactly that stage on the MovieLens **latest-small**
set with plain Python + numpy, and adds two production-quality touches so the
neighbours are actually good on sparse data:

| Spark pipeline (`compute_similarity`)        | `precompute.py`                                   |
| -------------------------------------------- | ------------------------------------------------- |
| `norm_rating = rating − mean_rating`         | mean-centre each movie's ratings                  |
| `sum_xy / (√sum_xx · √sum_yy)`               | Pearson over co-rating users                      |
| *(no minimum co-raters)*                     | **min 8 shared raters** to count a pair           |
| —                                            | **significance weighting** `sim × n/(n+25)`       |

> Why the extras? Raw Pearson on a small dataset happily returns 0.97 for two
> obscure films that share five raters. Requiring a handful of shared raters and
> shrinking similarities toward zero when overlap is thin makes genuinely
> similar, popular films rise to the top. (Before: *Toy Story → Percy Jackson.*
> After: *Toy Story → Toy Story 2, Finding Nemo, The Incredibles.*)

The browser then does the serving-time half of item-item CF: for your liked set,
it sums the precomputed neighbour similarities onto each candidate, subtracts the
influence of anything you rejected, and blends in a light genre-affinity signal
(a nod to the Spark model's genre-based Random Forest component).

---

## Regenerate the data

```bash
cd webapp
python3 precompute.py      # reads ml-latest-small/, writes data.js
```

The MovieLens `ml-latest-small` CSVs are bundled under `ml-latest-small/`. If you
delete them, `precompute.py` re-downloads the set (~1 MB) from GroupLens. Tune the
thresholds at the top of the script (`MIN_COMMON`, `SHRINK_LAMBDA`, …) to taste.

---

## Files

```
webapp/
├── index.html         # structure
├── css/style.css      # the cinematic look
├── js/app.js          # rendering + the in-browser recommender
├── data.js            # generated: catalog + per-movie neighbours (~1.4 MB)
├── precompute.py      # offline item-item Pearson CF → data.js
└── ml-latest-small/   # MovieLens data (GroupLens)
```

Data © GroupLens (MovieLens). See `ml-latest-small/README.txt` for their usage
terms.
