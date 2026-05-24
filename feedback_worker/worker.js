// Cloudflare Worker — feedback submission proxy.
//
// The static feedback page (docs/feedback_global.html, feedback_local.html)
// POSTs the collected entries here. This worker holds the GitHub PAT in its
// own environment variables and writes data/feedback_pending.json into the
// private logs repo on the page's behalf. The PAT never appears in any file
// committed to the public repo, so GitHub's secret scanning cannot revoke it.
//
// Configure these in the Cloudflare Worker's Settings -> Variables:
//   FEEDBACK_TOKEN    — fine-grained GitHub PAT, Contents read+write on the
//                       PRIVATE logs repo only. Set as a Secret (encrypted).
//   LOGS_REPO         — `owner/repo` of the private logs repo (e.g.
//                       moe-the-goat/job-scrapper-logs). Plain text variable.
//   ALLOWED_ORIGIN    — the GitHub Pages origin that may submit, e.g.
//                       https://moe-the-goat.github.io  (no trailing slash).
//
// Deploy: paste this file into the Cloudflare Workers editor, set the three
// variables above, save and deploy. The Worker URL it gives you back goes
// into the FEEDBACK_WORKER_URL repo variable on the GitHub side.

const PENDING_PATH = "data/feedback_pending.json";
const MAX_ENTRIES_PER_SUBMISSION = 100;

// Per RFC 6454 §4, origin scheme and host are case-insensitive. A capitalised
// hostname in either ALLOWED_ORIGIN (config typo) or the incoming Origin
// header (rare, but allowed by proxies / non-browser clients) would otherwise
// fail the equality check below and produce a 403 the user can't explain.
// Also strip any trailing slash so `https://example.com/` and `https://example.com`
// are treated as the same origin.
function normalizeOrigin(o) {
  return (o || "").trim().toLowerCase().replace(/\/+$/, "");
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(env) });
    }
    if (request.method !== "POST") {
      return json({ error: "Method not allowed" }, 405, env);
    }

    const allowedOrigin = normalizeOrigin(env.ALLOWED_ORIGIN);
    const requestOrigin = normalizeOrigin(request.headers.get("Origin"));
    if (allowedOrigin && requestOrigin !== allowedOrigin) {
      return json({ error: "Forbidden origin" }, 403, env);
    }

    if (!env.FEEDBACK_TOKEN || !env.LOGS_REPO) {
      return json({ error: "Worker not configured" }, 500, env);
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return json({ error: "Invalid JSON body" }, 400, env);
    }

    const entries = Array.isArray(payload?.entries) ? payload.entries : null;
    if (!entries || entries.length === 0) {
      return json({ error: "No entries provided" }, 400, env);
    }
    if (entries.length > MAX_ENTRIES_PER_SUBMISSION) {
      return json({ error: `Too many entries (max ${MAX_ENTRIES_PER_SUBMISSION})` }, 400, env);
    }

    const githubHeaders = {
      Authorization: `Bearer ${env.FEEDBACK_TOKEN}`,
      Accept: "application/vnd.github.v3+json",
      "User-Agent": "feedback-worker",
    };

    // Look up the current SHA of the pending file (null if it does not exist).
    let sha = null;
    const getUrl = `https://api.github.com/repos/${env.LOGS_REPO}/contents/${PENDING_PATH}`;
    const getResp = await fetch(getUrl, { headers: githubHeaders });
    if (getResp.ok) {
      const data = await getResp.json();
      sha = data.sha || null;
    } else if (getResp.status !== 404) {
      return json({ error: `GitHub read failed (${getResp.status})` }, 502, env);
    }

    // Overwrite the pending file with the new batch. The pipeline drains it
    // tomorrow morning and clears it.
    const body = JSON.stringify({ entries }, null, 2);
    const contentBase64 = utf8ToBase64(body);
    const writePayload = {
      message: `Submitted ${entries.length} feedback entries from feedback page`,
      content: contentBase64,
    };
    if (sha) writePayload.sha = sha;

    const putResp = await fetch(getUrl, {
      method: "PUT",
      headers: { ...githubHeaders, "Content-Type": "application/json" },
      body: JSON.stringify(writePayload),
    });
    if (!putResp.ok) {
      const text = await putResp.text();
      return json(
        { error: `GitHub write failed (${putResp.status}): ${text.slice(0, 200)}` },
        502,
        env,
      );
    }

    return json({ ok: true, count: entries.length }, 200, env);
  },
};

function corsHeaders(env) {
  // Echo the normalised allowed origin so the browser's case-sensitive
  // Access-Control-Allow-Origin comparison succeeds even when the env var
  // was configured with stray uppercase characters or a trailing slash.
  const allowed = normalizeOrigin(env.ALLOWED_ORIGIN);
  return {
    "Access-Control-Allow-Origin": allowed || "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}

function json(payload, status, env) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders(env) },
  });
}

function utf8ToBase64(str) {
  // Cloudflare Workers expose btoa but require Latin-1 input. Encode UTF-8
  // explicitly so notes containing non-ASCII characters survive the round trip.
  const bytes = new TextEncoder().encode(str);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}
