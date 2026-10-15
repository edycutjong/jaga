/* jaga overlay — vanilla JS, no build step. Reads /api/state once, then follows the /stream SSE feed.
   Every state change is also a word (ARMED, TRIPPED, ACTION FAILED), never colour alone. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const TOKEN = document.querySelector('meta[name="jaga-token"]').content;
  const TRACE_S = 120;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  const S = { snap: null, traces: {}, events: [], receipt: "", resetUntil: {}, snapped: {}, drawerSlot: null, examples: {},
    lastMsgAt: Date.now(), lost: false, rowNote: {}, plugNote: null };
  const LOST_AFTER_MS = 10000;  // no message from the watcher for this long = it is not running

  const fmt = {
    n: (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d)),
    s: (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(v < 10 ? 2 : 1)),
    signed: (v, d = 3) => (v === null || v === undefined ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d)),
    clock: (ts) => new Date(ts * 1000).toLocaleTimeString("en-GB", { hour12: false }),
    mmss: (s) => { s = Math.max(0, Math.ceil(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; },
  };
  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function post(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Jaga-Session": TOKEN },
      body: JSON.stringify(body || {}) });
    let data = {};
    try { data = await r.json(); } catch (_) { /* empty */ }
    if (!r.ok && !data.state) throw new Error(data.detail || data.error || `HTTP ${r.status}`);
    return data;
  }

  // ---------------------------------------------------------------- which tripwire the wire shows
  function activeRow(snap) {
    const rows = (snap && snap.windows) || [];
    return rows.find((w) => w.state === "tripped")
      || rows.filter((w) => w.state === "holding").sort((a, b) => b.progress - a.progress)[0]
      || rows.find((w) => w.state === "capturing") || rows.find((w) => w.state === "armed") || rows[0] || null;
  }
  const scoreOf = (w) => (w.mode === "labelled-m" ? w.m : w.z);
  const scoreName = (w) => (w.mode === "labelled-m" ? "m" : "z");

  // ---------------------------------------------------------------- render
  function renderHeader(snap) {
    const rows = snap.windows || [];
    const armed = rows.filter((w) => ["armed", "holding", "tripped"].includes(w.state)).length;
    $("hdr-armed").textContent = `${armed} armed here`;
    $("hdr-device").textContent = snap.device ? `${snap.device.toUpperCase()} fp32` : "—";
    $("hdr-spc").textContent = fmt.s(snap.s_per_clip_p50);
    $("hdr-period").textContent = fmt.s(snap.period_p50);
    const badge = $("hdr-badge");
    if (snap.compiler_down) {
      badge.hidden = false; badge.className = "badge degraded";
      badge.textContent = "TOKEN FACTORY UNREACHABLE · new sentences wait; armed ones run";
    } else if (snap.frozen_for) { badge.hidden = false; badge.className = "badge frozen"; badge.textContent = `θ FROZEN FOR ${snap.frozen_for}`; }
    else { badge.hidden = true; }
  }

  function renderCamera(snap) {
    const cam = $("cam");
    if (snap.frame_jpeg_b64) {
      $("frame").src = `data:image/jpeg;base64,${snap.frame_jpeg_b64}`;
      cam.classList.add("live");
    }
    if (snap.filmstrip_jpeg_b64 && snap.filmstrip_jpeg_b64.length) {
      const box = $("fs-frames");
      while (box.children.length < snap.filmstrip_jpeg_b64.length) { const im = document.createElement("img"); im.alt = ""; box.appendChild(im); }
      snap.filmstrip_jpeg_b64.forEach((b, i) => { box.children[i].src = `data:image/jpeg;base64,${b}`; });
    }
    const src = snap.source || {};
    if (!cam.classList.contains("live")) $("nosignal-text").textContent = src.error || "waiting for the first 8 frames…";
    const veil = $("veil");
    const row = activeRow(snap);
    let veilText = "";
    if (snap.dark) veilText = "TOO DARK TO SEE";
    else if (row && row.state === "unarmed" && /camera moved/.test(row.unarmed_reason || "")) veilText = "CAMERA MOVED — RE-ARM";
    veil.hidden = !veilText;
    $("veil-text").textContent = veilText;
    const mo = $("motion");
    $("mo-pct").textContent = snap.motion_pct === null || snap.motion_pct === undefined ? "—" : `${fmt.n(snap.motion_pct, 1)}%`;
    $("mo-still").textContent = snap.still_s === null || snap.still_s === undefined ? "—" : `${Math.round(snap.still_s)} s`;
    mo.classList.toggle("firing", (snap.motion_pct || 0) >= 0.5);
  }

  function splitSentence(text) {
    const words = String(text || "").toUpperCase().split(/\s+/).filter(Boolean);
    if (words.length < 2) return [words.join(" "), ""];
    let best = 1, bestDiff = Infinity;
    const total = words.join(" ").length;
    for (let i = 1; i < words.length; i++) {
      const left = words.slice(0, i).join(" ").length;
      const d = Math.abs(left - total * 0.55);
      if (d < bestDiff) { bestDiff = d; best = i; }
    }
    return [words.slice(0, best).join(" "), words.slice(best).join(" ")];
  }

  function renderWire(snap) {
    const tw = $("tapewire");
    const row = activeRow(snap);
    if (!row) { tw.dataset.state = "idle"; $("tracebox").hidden = true; $("dwell").hidden = true; return; }
    $("tracebox").hidden = false; $("dwell").hidden = false;
    const [l, r] = splitSentence(row.sentence);
    if ($("tape-l").textContent !== l) $("tape-l").textContent = l;
    if ($("tape-r").textContent !== r) $("tape-r").textContent = r;
    // The snap plays once, when THIS trip is acknowledged (the plug confirmed OFF for plug_off; the trip itself for
    // notify). A cached plug state never stands in for the acknowledgement (I3).
    let state = row.state;
    if (state === "tripped" && row.action === "plug_off" && !row.off_acked) state = "holding";
    tw.classList.toggle("static", reduced.matches || S.snapped[row.slot] === "done");
    if (state === "tripped" && !S.snapped[row.slot]) { S.snapped[row.slot] = "playing"; setTimeout(() => { S.snapped[row.slot] = "done"; }, 300); }
    if (state !== "tripped") delete S.snapped[row.slot];
    tw.dataset.state = state === "armed" || state === "holding" || state === "tripped" || state === "capturing" ? state : "unarmed";

    // dwell bar: windows above θ in the last dwell, out of k needed
    const cells = Math.min(Math.max(row.k || 0, 8), 24);
    const on = row.state === "tripped" ? cells : Math.round((row.progress || 0) * cells);
    const box = $("dwell-cells");
    if (box.children.length !== cells) { box.innerHTML = ""; for (let i = 0; i < cells; i++) box.appendChild(document.createElement("i")); }
    [...box.children].forEach((c, i) => c.classList.toggle("on", i < on));
    $("dwell").classList.toggle("tripped", row.state === "tripped");
    const dwell = Math.round(row.effective_dwell_s || row.dwell_s);
    const note = row.dwell_note ? ` (${row.dwell_note})` : "";
    $("dwell-text").textContent = row.state === "capturing" ? `capturing the normal state · ${row.capture_windows} windows`
      : row.state === "tripped" ? `held for the dwell · ${dwell} s${note}`
        : `dwell ${row.count}/${row.k} windows above · ${dwell} s${note}`;
    renderTrace(row, snap.ts);
  }

  function renderTrace(row, now) {
    const pts = (S.traces[row.slot] || []).filter((p) => p[0] >= now - TRACE_S && p[1] !== null && p[1] !== undefined);
    const theta = row.theta;
    $("tb-theta").textContent = theta === null || theta === undefined ? "θ —" :
      (row.mode === "labelled-m" ? `θ ${fmt.signed(theta, 4)} · v${row.theta_version ?? "?"}` : `θ z ${fmt.n(theta, 1)} · fresh`);
    const cur = scoreOf(row);
    $("tb-value").textContent = cur === null || cur === undefined ? "" : `${scoreName(row)} ${fmt.signed(cur, row.mode === "labelled-m" ? 4 : 2)}`;
    if (!pts.length || theta === null || theta === undefined) { $("trace-line").setAttribute("points", ""); $("trace-fill").setAttribute("d", ""); return; }
    const minSpan = row.mode === "labelled-m" ? 0.02 : 2.0;
    const span = Math.max(minSpan, ...pts.map((p) => Math.abs(p[1] - theta))) * 1.15;
    const X = (t) => (1000 * (t - (now - TRACE_S))) / TRACE_S;
    const Y = (v) => Math.min(97, Math.max(3, 50 - ((v - theta) / span) * 46));
    const xy = pts.map((p) => [X(p[0]).toFixed(1), Y(p[1]).toFixed(1)]);
    $("trace-line").setAttribute("points", xy.map((q) => q.join(",")).join(" "));
    const d = `M${xy[0][0]},50 ` + xy.map(([x, y]) => `L${x},${Math.min(Number(y), 50)}`).join(" ") + ` L${xy[xy.length - 1][0]},50 Z`;
    $("trace-fill").setAttribute("d", d);
  }

  // Capture countdown, then (for 90 s after it, until 3 examples) the optional check: "make it happen, press 3 times".
  function renderCapture(snap) {
    const rows = snap.windows || [];
    const box = $("capture");
    const capturing = rows.find((w) => w.state === "capturing");
    const checking = !capturing && rows.find((w) => (w.state === "armed" || w.state === "holding") && w.armed_at
      && (snap.ts - w.armed_at) < 90 && (w.examples || 0) < 3);
    const row = capturing || checking;
    if (!row) { box.hidden = true; return; }
    box.hidden = false;
    box.dataset.mode = capturing ? "capture" : "check";
    $("cap-title").textContent = capturing ? `NORMAL STATE · ${row.slot}` : `CHECK ${row.slot} · OPTIONAL`;
    $("cap-clock").textContent = capturing ? fmt.mmss(row.capture_left_s || 0) : `${row.examples || 0}/3`;
    $("cap-text").textContent = capturing ? captureHint(row.sentence)
      : "make it happen, then press three times so jaga can check it sees it";
    const btn = $("cap-example");
    btn.hidden = !!capturing;
    btn.textContent = `it's happening now (${row.examples || 0}/3)`;
    btn.dataset.slot = row.slot;
  }

  function captureHint(sentence) {
    const s = String(sentence || "").toLowerCase();
    if (/iron|setrika/.test(s)) return "iron as usual — this teaches jaga what normal looks like";
    if (/counter|meja dapur/.test(s)) return "use the counter as usual, leave nothing on it";
    if (/fridge|kulkas/.test(s)) return "use the fridge as usual and close it";
    if (/door|pintu/.test(s)) return "walk past the closed door as usual";
    return "do what normally happens here — this teaches jaga what normal looks like";
  }

  function renderBreaker(snap) {
    const p = snap.plug || {};
    const b = $("breaker");
    const rows = snap.windows || [];
    const tripped = rows.some((w) => w.action === "plug_off" && w.state === "tripped");
    const acked = rows.some((w) => w.action === "plug_off" && w.state === "tripped" && w.off_acked);
    b.dataset.state = tripped && !acked ? "unknown" : (p.state || "unknown");
    $("plug-label").textContent = p.label || "Wi-Fi plug";
    $("plug-addr").textContent = p.address || "—";
    $("flag").hidden = !acked;
    $("failed").hidden = !p.failed;
    if (p.failed) $("failed-detail").textContent = `${(p.detail || "the plug did not confirm OFF").slice(0, 90)} · push sent · retrying OFF`;
    let last = "last action: —";
    if (p.last_action === "off") last = `last action: OFF acknowledged in <span class="num">${fmt.n(p.latency_ms, 0)}</span> ms`;
    else if (p.last_action === "operator-on") last = `last action: operator test · ${esc(p.detail || "")}`;
    else if (p.last_action === "off-failed") last = "last action: OFF not acknowledged";
    if (p.kind === "none") last = "no plug configured yet — trips will show ACTION FAILED";
    if (S.plugNote && S.plugNote.until > Date.now()) last = esc(S.plugNote.text);
    $("plug-last").innerHTML = last;
  }

  const KIND = { trip: "TRIPPED", action: "PLUG OFF", action_failed: "ACTION FAILED", notify: "NOTIFIED", suspend: "SUSPENDED",
    camera_moved: "CAMERA MOVED", too_dark: "TOO DARK", watcher_stopped: "WATCHER STOPPED", watcher_ok: "WATCHER OK",
    rearmed: "RESET", armed: "ARMED", disarmed: "DISARMED", operator_on: "PLUG TEST", compile: "LOADED", unarmed: "UNARMED", notify_failed: "NOTIFY FAILED" };
  const ALARM = new Set(["trip", "action_failed", "watcher_stopped"]);

  function renderEvents(fresh) {
    const ol = $("events");
    ol.innerHTML = "";
    S.events.slice(0, 5).forEach((e) => {  // at most five one-line rows; a row that would not fit is not shown
      const li = document.createElement("li");
      if (ALARM.has(e.kind)) li.classList.add("alarm");
      if (fresh && e.id === fresh) li.classList.add("fresh");
      const slot = e.slot ? `${e.slot} ` : "";
      const word = e.kind === "armed" && /capturing/.test(e.detail || "") ? "ARMING" : (KIND[e.kind] || e.kind);
      li.innerHTML = `<span class="t">${fmt.clock(e.ts)}</span><span class="k">${esc(slot + word)}</span>` +
        `<span class="d" title="${esc(e.detail)}">${esc(e.detail)}</span>` +
        (e.kind === "trip" ? `<button class="wrong" type="button" data-id="${e.id}" aria-pressed="${e.wrong ? "true" : "false"}" title="mark this trip as wrong: kept as a labelled negative for the calibration report">✗ wrong</button>` : "<span></span>");
      ol.appendChild(li);
    });
    fitEvents();
  }
  function fitEvents() {
    const ol = $("events");
    const bottom = ol.getBoundingClientRect().bottom + 0.5;
    let cut = false;
    [...ol.children].forEach((li) => { li.hidden = false; });
    [...ol.children].forEach((li) => { if (cut || li.getBoundingClientRect().bottom > bottom) { cut = true; li.hidden = true; } });
  }
  window.addEventListener("resize", () => fitEvents());

  function scheduleRow(slot) {
    const tr = document.createElement("tr");
    tr.dataset.slot = slot;
    tr.innerHTML = '<td class="slot"></td><td class="what"><span class="sent"></span><span class="reason"></span></td>' +
      '<td class="rule"></td><td class="st"></td><td class="score num"></td><td class="act"><button type="button"></button></td>';
    return tr;
  }

  function setText(el, text) { if (el.textContent !== text) el.textContent = text; }

  // Rows are updated in place (never re-created each window), so a button under the cursor or holding keyboard
  // focus stays the same element.
  function renderSchedule(snap) {
    const tb = $("sch-rows");
    const rows = snap.windows || [];
    const soak = $("soak");
    soak.hidden = !snap.labels;
    if (snap.labels) {
      $("soak-name").textContent = `${snap.session}${snap.scored ? " · scored" : " · unscored"}${snap.frozen_for ? " · θ frozen" : ""}`;
      $("soak-keys").textContent = "log: 1 still starts · 2 still ends · 3 leaves the room · 4 steps away, in view · 5 back at the board · 6 left open · 7 closed · 0 note";
    }
    if (!rows.length) {
      if (!tb.querySelector("tr.empty")) {
        tb.innerHTML = '<tr class="empty"><td colspan="6">No tripwire yet. Type what to watch for above — Nemotron 3 Super compiles it, you check it, then arm it.</td></tr>';
      }
      return;
    }
    tb.querySelector("tr.empty")?.remove();
    const now = Date.now() / 1000;
    const seen = new Set();
    rows.forEach((w, i) => {
      seen.add(w.slot);
      let tr = tb.querySelector(`tr[data-slot="${w.slot}"]`);
      if (!tr) { tr = scheduleRow(w.slot); }
      if (tb.children[i] !== tr) tb.insertBefore(tr, tb.children[i] || null);
      let word = { armed: "ARMED", holding: "HOLDING", tripped: "TRIPPED", capturing: "CAPTURING", unarmed: "UNARMED" }[w.state] || w.state.toUpperCase();
      if (w.state === "armed" && S.resetUntil[w.slot] > now) word = "RESET";
      tr.className = w.state;
      setText(tr.querySelector(".slot"), w.slot);
      setText(tr.querySelector(".sent"), w.sentence);
      const note = S.rowNote[w.slot] && S.rowNote[w.slot].until > Date.now() ? S.rowNote[w.slot].text : "";
      setText(tr.querySelector(".reason"), note || (w.state === "unarmed" && w.unarmed_reason ? w.unarmed_reason : ""));
      const dwell = Math.round(w.effective_dwell_s || w.dwell_s);
      const rule = `dwell <span class="num">${dwell}</span> s → ${w.action === "plug_off" ? "plug OFF" : "notify"}` +
        (w.dwell_note ? ` <span class="muted">(${esc(w.dwell_note)})</span>` : "");
      const ruleTd = tr.querySelector(".rule");
      if (ruleTd.innerHTML !== rule) ruleTd.innerHTML = rule;
      setText(tr.querySelector(".st"), word + (w.state === "capturing" ? ` ${fmt.mmss(w.capture_left_s || 0)}` : ""));
      if (tr.dataset.word !== word) {  // the screen-reader line announces a change of state, not every window
        if (tr.dataset.word) $("sr-live").textContent = `tripwire ${w.slot}: ${word.toLowerCase()}`;
        tr.dataset.word = word;
      }
      const sc = scoreOf(w);
      setText(tr.querySelector(".score"), sc === null || sc === undefined ? "—" : `${scoreName(w)} ${fmt.signed(sc, w.mode === "labelled-m" ? 4 : 2)}`);
      const live = ["armed", "holding", "capturing", "tripped"].includes(w.state);
      const btn = tr.querySelector("button");
      const act = live ? "disarm" : "arm";
      if (btn.dataset.act !== act || btn.dataset.slot !== w.slot) {
        btn.dataset.act = act;
        btn.dataset.slot = w.slot;
        btn.className = live ? "btn ghost small" : "btn small";
        btn.textContent = act;
        btn.disabled = false;
        btn.setAttribute("aria-label", `${act} tripwire ${w.slot}`);
      }
    });
    [...tb.querySelectorAll("tr[data-slot]")].forEach((tr) => { if (!seen.has(tr.dataset.slot)) tr.remove(); });
  }

  function renderStopped(snap) {
    const box = $("stopped");
    document.body.classList.toggle("lost", S.lost);
    if (S.lost) {
      box.hidden = false;
      $("stopped-word").textContent = "LOST CONTACT";
      $("stopped-text").textContent = "jaga is not running — nothing is watching. Start it again.";
      return;
    }
    box.hidden = !snap || snap.watcher_ok !== false;
    $("stopped-word").textContent = "WATCHER STOPPED";
    if (!box.hidden) $("stopped-text").textContent = "no window is completing — check the camera and the laptop";
  }
  function heard() { S.lastMsgAt = Date.now(); if (S.lost) { S.lost = false; renderStopped(S.snap); } }
  setInterval(() => {
    if (!S.lost && Date.now() - S.lastMsgAt > LOST_AFTER_MS) { S.lost = true; renderStopped(S.snap); }
  }, 2000);

  function render(snap) {
    if (!snap) return;
    S.snap = snap;
    renderHeader(snap); renderCamera(snap); renderWire(snap); renderCapture(snap); renderBreaker(snap); renderSchedule(snap); renderStopped(snap);
  }

  // ---------------------------------------------------------------- data in
  function absorbWindow(snap) {
    // only a new window adds a trace point (the plug poll and the watchdog also publish snapshots)
    const fresh = (snap.n_windows || 0) > (S.lastN || 0);
    S.lastN = Math.max(S.lastN || 0, snap.n_windows || 0);
    if (!fresh) { render(snap); return; }
    (snap.windows || []).forEach((w) => {
      const sc = scoreOf(w);
      if (sc === null || sc === undefined) return;
      const arr = (S.traces[w.slot] = S.traces[w.slot] || []);
      if (!arr.length || arr[arr.length - 1][0] < snap.ts) arr.push([snap.ts, sc, w.theta, w.above]);
      while (arr.length && arr[0][0] < snap.ts - TRACE_S - 5) arr.shift();
    });
    render(snap);
  }

  async function boot() {
    try {
      const r = await fetch("/api/state");
      const d = await r.json();
      S.events = (d.events || []).map((e) => ({ ...e, slot: e.slot }));
      Object.entries(d.traces || {}).forEach(([slot, arr]) => {
        S.traces[slot] = arr.map((p) => [p[0], p[1], p[2], p[3]]).filter((p) => p[1] !== null);
      });
      if (d.receipt) $("foot-job").textContent = d.receipt;
      S.lastN = (d.snapshot && d.snapshot.n_windows) || 0;
      render(d.snapshot);
      renderEvents();
    } catch (e) { $("nosignal-text").textContent = `cannot reach the watcher: ${e.message}`; }
    const es = new EventSource("/stream");
    es.onopen = heard;
    es.addEventListener("window", (m) => { heard(); absorbWindow(JSON.parse(m.data)); });
    es.addEventListener("event", (m) => {
      heard();
      const e = JSON.parse(m.data);
      S.events.unshift(e); S.events = S.events.slice(0, 20);
      if (e.kind === "rearmed" && e.slot) S.resetUntil[e.slot] = Date.now() / 1000 + 10;
      renderEvents(e.id);
    });
    es.addEventListener("tripped", () => { if (S.snap) render(S.snap); });
    es.addEventListener("schedule", (m) => { const d = JSON.parse(m.data); $("soak-next").textContent = d.text || ""; });
    es.onerror = () => {  // the browser retries by itself; the screen says so at once instead of looking ARMED
      $("nosignal-text").textContent = "reconnecting to the watcher…";
      S.lost = true; renderStopped(S.snap);
    };
  }

  // ---------------------------------------------------------------- actions
  function openDrawer(state, d) {
    const dr = $("drawer");
    dr.hidden = false;
    dr.classList.toggle("notvisual", state === "not-visual");
    const spec = (d && d.spec) || {};
    S.drawerSlot = d && d.slot;
    $("dr-status").textContent = {
      compiling: "Nemotron 3 Super is compiling your sentence on Token Factory…",
      compiled: `compiled in ${fmt.n(d && d.latency_s, 1)} s · check it before arming`,
      cached: "Token Factory unreachable — showing the cached compile of this exact sentence",
      "not-visual": `can't see that: ${spec.reason || "name an object and where it is"}`,
      "compile-failed": (d && d.detail) || "compile failed",
    }[state] || "";
    $("dr-sentence").innerHTML = d && d.sentence ? `You typed <b>“${esc(d.sentence)}”</b>` : "";
    $("dr-pos").innerHTML = (spec.positives || []).map((t) => `<li>${esc(t)}</li>`).join("");
    $("dr-neg").innerHTML = (spec.hard_negatives || []).map((t) => `<li>${esc(t)}</li>`).join("");
    document.querySelector(".dr-cols").hidden = !(spec.positives || []).length;
    $("dr-choose").hidden = !spec.visual;
    $("dr-rule").innerHTML = spec.dwell_s ? `Super proposed: dwell <span class="num">${esc(spec.dwell_s)}</span> s → ${spec.action === "plug_off" ? "plug OFF" : "notify"} · cool-down <span class="num">${esc(spec.cooldown_s)}</span> s · ${esc(spec.name)}` : "";
    if (spec.action) {
      document.querySelectorAll('input[name="dr-action"]').forEach((r) => { r.checked = r.value === spec.action; });
      const sel = $("dr-dwell");
      if (![...sel.options].some((o) => Number(o.value) === spec.dwell_s)) { const o = document.createElement("option"); o.textContent = String(spec.dwell_s); sel.appendChild(o); }
      sel.value = String(spec.dwell_s);
    }
    $("dr-req").textContent = d && d.tf_request_id ? `${d.model || "nvidia/nemotron-3-super-120b-a12b"} · request ${d.tf_request_id}` : "";
    $("dr-title").textContent = { "compile-failed": "COMPILE FAILED · TOKEN FACTORY", compiling: "COMPILING · NEMOTRON 3 SUPER",
      "not-visual": "NOT A VISUAL TRIPWIRE · NEMOTRON 3 SUPER" }[state] || "COMPILED BY NEMOTRON 3 SUPER · TOKEN FACTORY";
    $("dr-arm").disabled = !(state === "compiled" || state === "cached");
    if (state === "compiled" || state === "cached") $("dr-arm").focus({ preventScroll: true });
    dr.scrollTop = 0;
  }
  function closeDrawer() { $("drawer").hidden = true; $("sentence").focus(); }

  $("compose").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const sentence = $("sentence").value.trim();
    if (!sentence) { $("sentence").focus(); return; }
    $("compile").disabled = true;
    openDrawer("compiling", { sentence });
    try {
      const d = await post("/tripwires", { sentence });
      openDrawer(d.state, { ...d, sentence });
    } catch (e) { openDrawer("compile-failed", { sentence, detail: e.message }); }
    finally { $("compile").disabled = false; }
  });
  $("dr-close").addEventListener("click", closeDrawer);
  $("dr-edit").addEventListener("click", () => { closeDrawer(); $("sentence").select(); });
  $("dr-arm").addEventListener("click", async () => {
    if (!S.drawerSlot) return;
    $("dr-arm").disabled = true;
    const act = (document.querySelector('input[name="dr-action"]:checked') || {}).value;
    try { await post(`/tripwires/${S.drawerSlot}/arm`, { capture_s: 120, action: act, dwell_s: Number($("dr-dwell").value) });
      closeDrawer(); $("sentence").value = ""; }
    catch (e) { $("dr-status").textContent = `could not arm: ${e.message}`; $("dr-arm").disabled = false; }
  });
  $("sch-rows").addEventListener("click", async (ev) => {
    const b = ev.target.closest("button[data-act]");
    if (!b) return;
    b.disabled = true;
    try { await post(`/tripwires/${b.dataset.slot}/${b.dataset.act}`, b.dataset.act === "arm" ? { capture_s: 120 } : {}); }
    catch (e) {
      b.textContent = "failed";
      b.title = e.message;
      S.rowNote[b.dataset.slot] = { text: `could not ${b.dataset.act}: ${e.message}`, until: Date.now() + 8000 };
      if (S.snap) renderSchedule(S.snap);
      setTimeout(() => { b.disabled = false; b.textContent = b.dataset.act; }, 2000);
    }
  });
  $("events").addEventListener("click", async (ev) => {
    const b = ev.target.closest("button.wrong");
    if (!b || b.getAttribute("aria-pressed") === "true") return;
    try { await post(`/events/${b.dataset.id}/wrong`); b.setAttribute("aria-pressed", "true");
      const e = S.events.find((x) => String(x.id) === b.dataset.id); if (e) e.wrong = 1; } catch (_) { /* shown by state */ }
  });
  $("plug-test").addEventListener("click", async () => {
    if (!window.confirm("Operator test: switch the plug OFF, then back ON?")) return;
    const b = $("plug-test"); b.disabled = true;
    try { const d = await post("/plug/test"); S.plugNote = { text: d.ok ? "test: OFF → ON acknowledged" : `test failed: ${d.detail}`, until: Date.now() + 10000 }; }
    catch (e) { S.plugNote = { text: `test failed: ${e.message}`, until: Date.now() + 10000 }; }
    finally { b.disabled = false; if (S.snap) renderBreaker(S.snap); }
  });
  $("cap-example").addEventListener("click", async (ev) => {
    const slot = ev.currentTarget.dataset.slot;
    if (!slot) return;
    try {
      const d = await post(`/tripwires/${slot}/example`);
      if (d.ok === false && d.examples === 3) $("cap-text").textContent = "can't see that — name an object and where it is";
    } catch (_) { /* the next window shows the state */ }
  });
  const SOAK_KEYS = { 1: "still_start", 2: "still_end", 3: "exit_leave", 4: "exit_inview", 5: "returned",
    6: "left_open", 7: "closed", 0: "note" };
  document.addEventListener("keydown", (ev) => {
    const t = ev.target;
    const typing = (t.tagName === "INPUT" && !["radio", "checkbox"].includes(t.type)) || t.tagName === "SELECT"
      || t.tagName === "TEXTAREA";
    if (ev.key === "Escape") {
      if (!$("drawer").hidden) { ev.preventDefault(); closeDrawer(); } else if (typing) t.blur();
      return;
    }
    if (typing) return;
    if (ev.key === "/") { ev.preventDefault(); $("sentence").focus(); return; }
    if (ev.metaKey || ev.ctrlKey || ev.altKey || ev.repeat) return;  // a label is one plain key press
    if (S.snap && S.snap.labels && SOAK_KEYS[ev.key] !== undefined) {
      post("/labels", { kind: SOAK_KEYS[ev.key], note: "key" }).then(() => {
        $("soak-next").textContent = `logged: ${SOAK_KEYS[ev.key].replace("_", " ")} at ${fmt.clock(Date.now() / 1000)}`;
      }).catch(() => {});
    }
  });
  const tick = () => { $("hdr-clock").textContent = new Date().toLocaleTimeString("en-GB", { hour12: false }); };
  tick();
  setInterval(tick, 1000);
  boot();
})();
