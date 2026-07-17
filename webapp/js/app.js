/* ══════════════════════════════════════════════════════════════════
   CineMatch — client-side app
   Recommendation = item-item collaborative filtering (precomputed Pearson
   neighbours) + genre-affinity blend + dislike refinement, all in-browser.
   ══════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  const DATA = window.MOVIE_DATA;
  if (!DATA) { document.body.innerHTML = "<p style='padding:40px'>data.js failed to load. Run <code>python3 precompute.py</code> first.</p>"; return; }

  const GENRES = DATA.genres;
  const MOVIES = DATA.movies;                 // sorted popular-first
  const NB = DATA.nb;                         // { "id": [[nid, sim], ...] }
  const byId = new Map(MOVIES.map((m) => [m.id, m]));

  // tuning for the live recommender
  const DISLIKE_W = 1.15;
  const GENRE_W = 0.35;
  const POP_W = 0.05;
  const MAX_RECS = 24;
  const PAGE = 54;

  // ── genre visual identity (emoji + duotone poster palette) ──────────
  const GMETA = {
    "Action":      { e: "💥", c1: "#ff6b3d", c2: "#711421" },
    "Adventure":   { e: "🧭", c1: "#2bb673", c2: "#0f3d3a" },
    "Animation":   { e: "🎨", c1: "#ff7ac6", c2: "#4c2380" },
    "Children":    { e: "🧸", c1: "#ffd166", c2: "#d8641f" },
    "Comedy":      { e: "😄", c1: "#ffcf4d", c2: "#b9540f" },
    "Crime":       { e: "🕵️", c1: "#6376a0", c2: "#141827" },
    "Documentary": { e: "🎥", c1: "#7fb2c9", c2: "#1c3746" },
    "Drama":       { e: "🎭", c1: "#e26a78", c2: "#451421" },
    "Fantasy":     { e: "🐉", c1: "#9b7bff", c2: "#1b2a5e" },
    "Film-Noir":   { e: "🎩", c1: "#9aa0a6", c2: "#111214" },
    "Horror":      { e: "🔪", c1: "#d0283a", c2: "#0a0a0c" },
    "IMAX":        { e: "🌌", c1: "#3ea0ff", c2: "#0a1e46" },
    "Musical":     { e: "🎵", c1: "#ff77a8", c2: "#6f2a63" },
    "Mystery":     { e: "🔍", c1: "#4d8f8b", c2: "#132433" },
    "Romance":     { e: "💗", c1: "#ff8fa3", c2: "#851b39" },
    "Sci-Fi":      { e: "🚀", c1: "#31d0e0", c2: "#0d2a52" },
    "Thriller":    { e: "🌀", c1: "#4aa6a0", c2: "#181130" },
    "War":         { e: "⚔️", c1: "#b5895a", c2: "#30200f" },
    "Western":     { e: "🤠", c1: "#e0a458", c2: "#5f2b18" },
  };
  const GDEFAULT = { e: "🎬", c1: "#8c7f74", c2: "#211a1f" };
  const gmeta = (m) => (m.g.length ? (GMETA[GENRES[m.g[0]]] || GDEFAULT) : GDEFAULT);

  // ── state (persisted) ──────────────────────────────────────────────
  const state = { likes: new Set(), dislikes: new Set() };
  const LSKEY = "cinematch.v1";
  function save() {
    try { localStorage.setItem(LSKEY, JSON.stringify({ l: [...state.likes], d: [...state.dislikes] })); } catch (e) {}
  }
  function load() {
    try {
      const s = JSON.parse(localStorage.getItem(LSKEY) || "{}");
      (s.l || []).forEach((id) => byId.has(id) && state.likes.add(id));
      (s.d || []).forEach((id) => byId.has(id) && state.dislikes.add(id));
    } catch (e) {}
  }

  // ── helpers ─────────────────────────────────────────────────────────
  function hash(s) { let h = 2166136261 >>> 0; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
  function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function posterCSS(m) {
    const g = gmeta(m); const h = hash(m.t);
    const ang = 118 + (h % 66);
    const px = 20 + (h % 60), py = 8 + ((h >> 3) % 30);
    return `background:radial-gradient(90% 70% at ${px}% ${py}%, ${g.c1}, transparent 60%), linear-gradient(${ang}deg, ${g.c1}, ${g.c2});`;
  }
  const els = {};
  const $ = (id) => document.getElementById(id);

  // ── movie card (Discover) ───────────────────────────────────────────
  function cardHTML(m) {
    const g = gmeta(m);
    const liked = state.likes.has(m.id);
    const disliked = state.dislikes.has(m.id);
    const gname = m.g.length ? GENRES[m.g[0]] : "Film";
    const heart = liked
      ? `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 21s-7.5-4.6-10-9C.4 8.9 1.6 5 5.2 5 7.4 5 9 6.4 12 9c3-2.6 4.6-4 6.8-4C22.4 5 23.6 8.9 22 12c-2.5 4.4-10 9-10 9z"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20s-6.8-4.2-9.2-8.2C1.2 9 2.4 5.6 5.6 5.6c2 0 3.4 1.3 6.4 3.8 3-2.5 4.4-3.8 6.4-3.8 3.2 0 4.4 3.4 2.8 6.2C18.8 15.8 12 20 12 20z"/></svg>`;
    return `<article class="card ${liked ? "is-liked" : ""} ${disliked ? "is-disliked" : ""}" data-id="${m.id}" tabindex="0" role="button" aria-pressed="${liked}" aria-label="${esc(m.t)}${m.y ? ", " + m.y : ""}. Tap to add to your picks">
      <div class="poster" style="${posterCSS(m)}">
        <span class="poster__mono">${esc((m.t[0] || "•").toUpperCase())}</span>
        <span class="poster__glyph">${g.e}</span>
        <button class="card__heart" aria-hidden="true" tabindex="-1">${heart}</button>
        <div class="poster__scrim"><div class="poster__title">${esc(m.t)}</div></div>
      </div>
      <div class="card__meta">
        ${m.y ? `<span class="card__year">${m.y}</span><span class="card__dot">·</span>` : ""}
        <span class="card__rating">★ ${m.r.toFixed(1)}</span>
        <span class="card__genre">${esc(gname)}</span>
      </div>
    </article>`;
  }

  // ── Discover grid: search + genre filter + infinite scroll ──────────
  let activeGenre = -1;      // index or -1
  let query = "";
  let filtered = MOVIES;
  let rendered = 0;

  function applyFilter() {
    const q = query.trim().toLowerCase();
    filtered = MOVIES.filter((m) => {
      if (activeGenre >= 0 && !m.g.includes(activeGenre)) return false;
      if (q && !m.t.toLowerCase().includes(q)) return false;
      return true;
    });
    rendered = 0;
    els.grid.innerHTML = "";
    els.gridEmpty.hidden = filtered.length > 0;
    renderPage();
    els.gridCount.textContent = filtered.length.toLocaleString() + " films";
    els.gridTitle.textContent = q
      ? `Results for “${query.trim()}”`
      : activeGenre >= 0 ? `${GENRES[activeGenre]} favorites` : "Most-loved on MovieLens";
  }
  function renderPage() {
    const slice = filtered.slice(rendered, rendered + PAGE);
    const frag = document.createElement("div");
    frag.innerHTML = slice.map(cardHTML).join("");
    // stagger the first screenful only
    [...frag.children].forEach((c, i) => { if (rendered + i < PAGE) c.style.animationDelay = (Math.min(i, 24) * 22) + "ms"; els.grid.appendChild(c); });
    rendered += slice.length;
  }

  // ── genre chips ─────────────────────────────────────────────────────
  function renderChips() {
    const all = `<button class="chip is-active" data-g="-1">All films</button>`;
    const chips = GENRES.map((g, i) => {
      const meta = GMETA[g] || GDEFAULT;
      return `<button class="chip" data-g="${i}"><span class="chip__emoji">${meta.e}</span>${esc(g)}</button>`;
    }).join("");
    els.chips.innerHTML = all + chips;
  }

  // ── filmstrip picks tray ────────────────────────────────────────────
  function renderTray() {
    const ids = [...state.likes];
    els.pickCount.textContent = ids.length;
    els.filmstrip.dataset.empty = ids.length ? "false" : "true";
    els.pickPlaceholder.hidden = ids.length > 0;
    els.clearBtn.hidden = ids.length === 0;
    els.matchesBtn.disabled = ids.length === 0;
    // reel frames (newest first)
    const frames = ids.slice().reverse().map((id) => {
      const m = byId.get(id); if (!m) return "";
      const g = gmeta(m);
      return `<div class="frame" data-id="${id}" title="Remove “${esc(m.t)}”" style="${posterCSS(m)}"><span class="frame__glyph">${g.e}</span><span class="frame__x">✕</span></div>`;
    }).join("");
    els.pickReel.querySelectorAll(".frame").forEach((n) => n.remove());
    els.pickReel.insertAdjacentHTML("beforeend", frames);

    const c = Math.min(state.dislikes.size + state.likes.size, 99);
    void c;
    updateBadge();
  }

  // ── recommendation engine ───────────────────────────────────────────
  function computeRecs() {
    const likes = [...state.likes];
    if (!likes.length) return [];

    const scores = new Map(); // id -> {cf, contribs:[{id,sim}]}
    for (const lid of likes) {
      const list = NB[lid]; if (!list) continue;
      for (const [nid, sim] of list) {
        if (state.likes.has(nid) || state.dislikes.has(nid)) continue;
        let e = scores.get(nid); if (!e) { e = { cf: 0, contribs: [] }; scores.set(nid, e); }
        e.cf += sim; e.contribs.push({ id: lid, sim });
      }
    }
    for (const did of state.dislikes) {
      const list = NB[did]; if (!list) continue;
      for (const [nid, sim] of list) { const e = scores.get(nid); if (e) e.cf -= DISLIKE_W * sim; }
    }

    // genre affinity from liked movies
    const aff = new Array(GENRES.length).fill(0);
    for (const lid of likes) { const m = byId.get(lid); if (m) m.g.forEach((g) => aff[g]++); }
    const affSum = aff.reduce((a, b) => a + b, 0) || 1;
    for (let i = 0; i < aff.length; i++) aff[i] /= affSum;
    const maxPopLog = Math.log10(MOVIES[0].n + 1);

    const recs = [];
    for (const [id, e] of scores) {
      const m = byId.get(id); if (!m) continue;
      let g = 0; for (const gi of m.g) g += aff[gi];
      const pop = Math.log10(m.n + 1) / maxPopLog;
      e.final = e.cf + GENRE_W * g + POP_W * pop;
      e.movie = m; e.genreScore = g;
      recs.push(e);
    }

    // genre-based backfill if collaborative signal is thin
    if (recs.length < 12) {
      const have = new Set(recs.map((e) => e.movie.id));
      const extra = MOVIES
        .filter((m) => !have.has(m.id) && !state.likes.has(m.id) && !state.dislikes.has(m.id) && m.g.some((g) => aff[g] > 0))
        .map((m) => {
          let g = 0; for (const gi of m.g) g += aff[gi];
          const pop = Math.log10(m.n + 1) / maxPopLog;
          return { cf: 0, contribs: [], movie: m, genreScore: g, final: GENRE_W * g + POP_W * pop, genreOnly: true };
        })
        .sort((a, b) => b.final - a.final)
        .slice(0, 12 - recs.length);
      recs.push(...extra);
    }

    recs.sort((a, b) => b.final - a.final);
    const top = recs.slice(0, MAX_RECS);
    const maxF = top.length ? top[0].final : 1;
    for (const e of top) {
      e.pct = Math.max(58, Math.min(99, Math.round(60 + 39 * (e.final / maxF))));
      e.contribs.sort((a, b) => b.sim - a.sim);
      e.reason = reasonHTML(e, aff);
    }
    return top;
  }

  function reasonHTML(e, aff) {
    if (e.contribs && e.contribs.length) {
      const names = e.contribs.slice(0, 2).map((c) => byId.get(c.id) && byId.get(c.id).t).filter(Boolean);
      if (names.length >= 2) return `Loved by fans of <b>${esc(names[0])}</b> & <b>${esc(names[1])}</b>`;
      if (names.length === 1) return `Because you liked <b>${esc(names[0])}</b>`;
    }
    const gi = e.movie.g.slice().sort((a, b) => (aff[b] || 0) - (aff[a] || 0))[0];
    const gname = gi != null ? GENRES[gi] : "great";
    return `A <b>${esc(gname)}</b> pick in your wheelhouse`;
  }

  // ── For You view ────────────────────────────────────────────────────
  let recsCache = [];
  function renderForYou() {
    recsCache = computeRecs();
    const has = recsCache.length > 0;
    els.foryouEmpty.hidden = has;
    els.foryouContent.hidden = !has;
    if (!has) return;

    // taste summary
    const aff = new Array(GENRES.length).fill(0);
    for (const lid of state.likes) { const m = byId.get(lid); if (m) m.g.forEach((g) => aff[g]++); }
    const topG = aff.map((v, i) => [i, v]).filter((x) => x[1] > 0).sort((a, b) => b[1] - a[1]).slice(0, 3);
    els.tasteGenres.innerHTML = topG.map(([i, v]) => {
      const meta = GMETA[GENRES[i]] || GDEFAULT;
      return `<span class="taste-pill"><span>${meta.e} ${esc(GENRES[i])}</span><span>×${v}</span></span>`;
    }).join("");
    const n = state.likes.size;
    els.tasteBased.innerHTML = `Screened against <b>${n}</b> film${n > 1 ? "s" : ""} you love and the viewers who loved them too.` +
      (state.dislikes.size ? ` Steering clear of <b>${state.dislikes.size}</b> you passed on.` : "");

    els.recGrid.innerHTML = recsCache.map(recHTML).join("");
    [...els.recGrid.children].forEach((c, i) => (c.style.animationDelay = Math.min(i, 20) * 40 + "ms"));
  }

  function recHTML(e) {
    const m = e.movie; const g = gmeta(m);
    const gname = m.g.length ? GENRES[m.g[0]] : "Film";
    return `<article class="card rec" data-id="${m.id}">
      <div class="poster" style="${posterCSS(m)}">
        <span class="poster__glyph">${g.e}</span>
        <div class="rec__badge"><span class="rec__pct">${e.pct}</span><span class="rec__pctlabel">match</span></div>
        <div class="poster__scrim"><div class="poster__title">${esc(m.t)}</div></div>
      </div>
      <div class="rec__body">
        <h3 class="rec__title">${esc(m.t)} ${m.y ? `<span>${m.y}</span>` : ""}</h3>
        <p class="rec__reason">${e.reason}</p>
        <div class="rec__actions">
          <button class="rec-btn rec-btn--like" data-act="like" data-id="${m.id}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20s-6.8-4.2-9.2-8.2C1.2 9 2.4 5.6 5.6 5.6c2 0 3.4 1.3 6.4 3.8 3-2.5 4.4-3.8 6.4-3.8 3.2 0 4.4 3.4 2.8 6.2C18.8 15.8 12 20 12 20z"/></svg>
            Love it
          </button>
          <button class="rec-btn rec-btn--no" data-act="dislike" data-id="${m.id}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18M6 6l12 12"/></svg>
            Not for me
          </button>
        </div>
      </div>
    </article>`;
  }

  // ── badge + view switching ──────────────────────────────────────────
  function updateBadge() {
    const n = state.likes.size ? computeRecs().length : 0;
    els.matchCount.textContent = n;
    els.matchCount.hidden = n === 0;
  }
  let currentView = "discover";
  function switchView(v) {
    currentView = v;
    document.querySelectorAll(".seg").forEach((b) => {
      const on = b.dataset.view === v;
      b.classList.toggle("is-active", on); b.setAttribute("aria-selected", on);
    });
    $("view-discover").classList.toggle("is-active", v === "discover");
    $("view-foryou").classList.toggle("is-active", v === "foryou");
    if (v === "foryou") renderForYou();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ── mutations ───────────────────────────────────────────────────────
  let toastT;
  function toast(msg) {
    els.toast.textContent = msg; els.toast.classList.add("is-show");
    clearTimeout(toastT); toastT = setTimeout(() => els.toast.classList.remove("is-show"), 2600);
  }
  function like(id) {
    state.dislikes.delete(id);
    if (state.likes.has(id)) { state.likes.delete(id); }
    else {
      state.likes.add(id);
      if (state.likes.size === 3) toast("Nice — that's enough to match. Peek at ✨ For You");
    }
    afterChange();
  }
  function dislike(id) {
    state.likes.delete(id); state.dislikes.add(id); afterChange();
  }
  function removePick(id) { state.likes.delete(id); afterChange(); }
  function clearAll() { state.likes.clear(); state.dislikes.clear(); afterChange(); toast("Cleared your reel"); }

  function afterChange() {
    save();
    syncCards();
    renderTray();
    if (currentView === "foryou") renderForYou();
  }
  function syncCards() {
    els.grid.querySelectorAll(".card").forEach((c) => {
      const id = +c.dataset.id;
      const liked = state.likes.has(id), dis = state.dislikes.has(id);
      c.classList.toggle("is-liked", liked); c.classList.toggle("is-disliked", dis);
      c.setAttribute("aria-pressed", liked);
      const heart = c.querySelector(".card__heart");
      if (heart) heart.innerHTML = liked
        ? `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 21s-7.5-4.6-10-9C.4 8.9 1.6 5 5.2 5 7.4 5 9 6.4 12 9c3-2.6 4.6-4 6.8-4C22.4 5 23.6 8.9 22 12c-2.5 4.4-10 9-10 9z"/></svg>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20s-6.8-4.2-9.2-8.2C1.2 9 2.4 5.6 5.6 5.6c2 0 3.4 1.3 6.4 3.8 3-2.5 4.4-3.8 6.4-3.8 3.2 0 4.4 3.4 2.8 6.2C18.8 15.8 12 20 12 20z"/></svg>`;
    });
  }

  // ── events ──────────────────────────────────────────────────────────
  function wire() {
    // Discover grid: click / keyboard toggles like
    els.grid.addEventListener("click", (ev) => {
      const card = ev.target.closest(".card"); if (!card) return;
      like(+card.dataset.id);
    });
    els.grid.addEventListener("keydown", (ev) => {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      const card = ev.target.closest(".card"); if (!card) return;
      ev.preventDefault(); like(+card.dataset.id);
    });

    // genre chips
    els.chips.addEventListener("click", (ev) => {
      const chip = ev.target.closest(".chip"); if (!chip) return;
      els.chips.querySelectorAll(".chip").forEach((c) => c.classList.remove("is-active"));
      chip.classList.add("is-active");
      activeGenre = +chip.dataset.g;
      applyFilter();
    });

    // search (debounced)
    let sT;
    els.search.addEventListener("input", (ev) => {
      clearTimeout(sT); const v = ev.target.value;
      sT = setTimeout(() => { query = v; applyFilter(); }, 130);
    });

    // filmstrip: remove pick
    els.pickReel.addEventListener("click", (ev) => {
      const f = ev.target.closest(".frame"); if (!f) return; removePick(+f.dataset.id);
    });

    // tray buttons
    els.clearBtn.addEventListener("click", clearAll);
    els.matchesBtn.addEventListener("click", () => switchView("foryou"));

    // segmented nav
    document.querySelector(".segmented").addEventListener("click", (ev) => {
      const b = ev.target.closest(".seg"); if (!b) return; switchView(b.dataset.view);
    });
    document.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => switchView(b.dataset.goto)));

    // For You rec actions
    els.recGrid.addEventListener("click", (ev) => {
      const btn = ev.target.closest(".rec-btn"); if (!btn) return;
      const id = +btn.dataset.id;
      if (btn.dataset.act === "like") { like(id); toast("Added to your picks"); }
      else { dislike(id); }
    });

    // surprise me
    els.surprise.addEventListener("click", () => {
      const pool = MOVIES.slice(0, 260);
      const picks = new Set();
      while (picks.size < 4) picks.add(pool[Math.floor(Math.random() * pool.length)].id);
      picks.forEach((id) => { state.dislikes.delete(id); state.likes.add(id); });
      afterChange(); toast("Seeded 4 crowd-pleasers — tweak away!");
      window.scrollTo({ top: 0, behavior: "smooth" });
    });

    // infinite scroll
    const io = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && currentView === "discover" && rendered < filtered.length) renderPage();
    }, { rootMargin: "600px" });
    io.observe($("scrollSentinel"));
  }

  // ── boot ────────────────────────────────────────────────────────────
  function init() {
    els.grid = $("movieGrid"); els.gridEmpty = $("gridEmpty");
    els.gridTitle = $("gridTitle"); els.gridCount = $("gridCount");
    els.chips = $("genreChips"); els.search = $("searchInput");
    els.filmstrip = $("filmstrip"); els.pickCount = $("pickCount");
    els.pickReel = $("pickReel"); els.pickPlaceholder = $("pickPlaceholder");
    els.clearBtn = $("clearBtn"); els.matchesBtn = $("matchesBtn");
    els.matchCount = $("matchCount"); els.surprise = $("surpriseBtn");
    els.foryouEmpty = $("foryouEmpty"); els.foryouContent = $("foryouContent");
    els.tasteGenres = $("tasteGenres"); els.tasteBased = $("tasteBased");
    els.recGrid = $("recGrid"); els.toast = $("toast");
    els.pickHint = $("pickHint");

    load();
    renderChips();
    applyFilter();
    renderTray();
    wire();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
