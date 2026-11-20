// JobPilot UI: profile, digest, search and chat (vanilla JS, no build step).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} },
};
let userId = store.get("jobpilot_user");

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText);
  return data;
}
async function busy(btn, fn) { btn.disabled = true; const t = btn.textContent; btn.textContent = "Working..."; try { await fn(); } finally { btn.disabled = false; btn.textContent = t; } }

function jobHTML(j, extra = {}) {
  const where = [j.location, j.remote ? "Remote" : ""].filter(Boolean).join(" · ");
  const why = (extra.why || j.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("");
  const miss = (extra.missing || j.missing_skills || []).map((s) => `<span class="chip miss">${esc(s)}</span>`).join("");
  const skills = (j.skills || []).map((s) => `<span class="chip">${esc(s)}</span>`).join("");
  const demo = j.source === "synthetic";
  return `<div class="job" data-id="${j.id}">
    <h3>${esc(j.title)} <span class="muted">· ${esc(j.company)}</span>
      ${extra.fit != null ? `<span class="badge fit">fit ${extra.fit}/100</span>` : ""}
      ${demo ? '<span class="badge demo">demo</span>' : ""}
      ${j.ghost_score >= 0.5 ? '<span class="badge ghost">possible ghost listing</span>' : ""}</h3>
    <div class="meta">#${j.id} · ${esc(where)} ${j.salary ? "· " + esc(j.salary) : ""} ${j.level ? "· " + esc(j.level) : ""} ${j.posted_at ? "· posted " + esc(j.posted_at) : ""}</div>
    <div>${skills}</div>
    ${why ? `<ul class="why">${why}</ul>` : ""}
    ${miss ? `<div class="muted">To work on: ${miss}</div>` : ""}
    <div class="actions">
      <button class="secondary" data-fb="1" data-job="${j.id}">👍 Save</button>
      <button class="secondary" data-fb="-1" data-job="${j.id}">👎 Not for me</button>
      ${!demo && j.url ? `<a href="${esc(j.url)}" target="_blank" rel="noopener">Apply ↗</a>` : ""}
    </div></div>`;
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-fb]");
  if (!b || !userId) return;
  await api("/api/feedback", { method: "POST", body: JSON.stringify({ user_id: userId, job_id: +b.dataset.job, signal: +b.dataset.fb }) });
  b.textContent = b.dataset.fb === "1" ? "✓ Saved" : "✓ Hidden next time";
});

function showProfile(u) {
  const p = u.profile || {};
  $("signup").classList.add("hidden"); $("profile").classList.remove("hidden"); $("app").classList.remove("hidden");
  $("pName").textContent = u.name ? `${u.name}'s profile` : "Your profile";
  $("pBody").innerHTML = `<p>${esc(p.headline)}</p>
    <div class="muted">Seniority: ${esc(p.seniority)} · Experience: ${esc(p.years_experience)} yrs · Cities: ${esc((p.locations || []).join(", ") || "any")} · Remote ok: ${p.remote_ok !== false ? "yes" : "no"}${p.min_salary_lpa ? " · Min " + esc(p.min_salary_lpa) + " LPA" : ""}</div>
    <div style="margin-top:6px">${(p.skills || []).map((s) => `<span class="chip">${esc(s)}</span>`).join("")}</div>
    <div class="muted" style="margin-top:6px">Target roles: ${esc((p.target_roles || []).join(", ") || "-")}</div>`;
}

async function loadUser() {
  if (!userId) return;
  try { showProfile(await api(`/api/users/${userId}`)); } catch { store.del("jobpilot_user"); userId = null; }
}

$("createBtn").onclick = () => busy($("createBtn"), async () => {
  $("signupMsg").textContent = "Reading your resume with the LLM...";
  try {
    const u = await api("/api/users", { method: "POST", body: JSON.stringify({ name: $("name").value, resume_text: $("resume").value }) });
    userId = u.id; store.set("jobpilot_user", userId); $("signupMsg").textContent = ""; showProfile(u);
  } catch (err) { $("signupMsg").innerHTML = `<span class="err">${esc(err.message)}</span>`; }
});
$("resetBtn").onclick = () => { store.del("jobpilot_user"); userId = null; location.reload(); };

document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
  ["digest", "search", "chat"].forEach((t) => $("tab-" + t).classList.toggle("hidden", t !== b.dataset.tab));
});

$("digestBtn").onclick = () => busy($("digestBtn"), async () => {
  $("digestList").innerHTML = ""; $("digestSummary").textContent = "Ranking jobs and asking the LLM to pick your best 3...";
  try {
    const d = await api(`/api/users/${userId}/digest`);
    $("digestSummary").textContent = d.summary;
    $("digestMeta").textContent = d.meta ? `${d.meta.source === "llm" ? "picked by LLM" : "ranking fallback"} · ${d.meta.candidates} candidates · ${d.meta.ms ?? "?"} ms` : "";
    $("digestList").innerHTML = d.picks.map((p) => jobHTML(p.job, { fit: p.fit_score, why: p.why, missing: p.missing_skills })).join("");
  } catch (err) { $("digestSummary").innerHTML = `<span class="err">${esc(err.message)}</span>`; }
});

async function doSearch() {
  const q = $("q").value.trim(); if (!q) return;
  const params = new URLSearchParams({ q, remote_only: $("remote").checked, limit: 10 });
  if (userId) params.set("user_id", userId);
  if ($("loc").value.trim()) params.set("location", $("loc").value.trim());
  await busy($("searchBtn"), async () => {
    try {
      const r = await api(`/api/search?${params}`);
      $("searchMeta").textContent = `${r.results.length} results in ${r.took_ms} ms (hybrid semantic + keyword, personalised)`;
      $("results").innerHTML = r.results.map((j) => jobHTML(j)).join("") || "<p>No results.</p>";
    } catch (err) { $("searchMeta").innerHTML = `<span class="err">${esc(err.message)}</span>`; }
  });
}
$("searchBtn").onclick = doSearch;
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });

async function sendChat() {
  const text = $("chatIn").value.trim(); if (!text) return;
  $("chatIn").value = "";
  $("chatLog").insertAdjacentHTML("beforeend", `<div class="msg user">${esc(text)}</div>`);
  await busy($("chatBtn"), async () => {
    try {
      const r = await api("/api/chat", { method: "POST", body: JSON.stringify({ user_id: userId, message: text }) });
      const trace = (r.trace || []).map((t) => `${t.tool} (${t.ms} ms)${t.ok ? "" : " ✗"}`).join(" → ");
      $("chatLog").insertAdjacentHTML("beforeend", `<div class="msg bot">${esc(r.reply)}${trace ? `<div class="trace">tools used: ${esc(trace)}</div>` : ""}</div>`
        + (r.jobs || []).slice(0, 3).map((j) => jobHTML(j)).join(""));
      loadUser();
    } catch (err) { $("chatLog").insertAdjacentHTML("beforeend", `<div class="msg bot err">${esc(err.message)}</div>`); }
  });
}
$("chatBtn").onclick = sendChat;
$("chatIn").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });

api("/api/stats").then((s) => { $("stats").textContent = `· ${s.active_jobs} live jobs · ${s.users} users · ${s.searches} searches`; }).catch(() => {});
loadUser();
