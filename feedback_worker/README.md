# Feedback Worker (Cloudflare)

The static feedback page on GitHub Pages cannot embed a GitHub token. Any
token committed to a public file gets revoked by GitHub's secret scanner the
moment it is pushed. This worker is the server-side proxy that holds the
token instead.

The page POSTs the collected feedback to the worker. The worker validates
the request origin, then writes `data/feedback_pending.json` into the
private logs repo using a PAT stored in its own environment. The pipeline
ingests the file the next morning.

## One-time setup

1. **Create a Cloudflare account** at `dash.cloudflare.com` if you do not
   already have one. Free tier covers everything (no credit card required).

2. **Create a new Worker**. From the dashboard, go to Workers and Pages,
   then click Create application, then Create Worker. Give it a name like
   `job-feedback`. The default URL will look like
   `https://job-feedback.<your-cf-subdomain>.workers.dev`. Save that URL.

3. **Paste the worker code**. Open the worker's online editor, replace the
   placeholder code with the contents of [`worker.js`](worker.js), then
   click Deploy.

4. **Create a new fine-grained GitHub PAT** (the old `FEEDBACK_WRITE_TOKEN`
   was revoked when it was discovered in the page source). At
   `github.com/settings/personal-access-tokens/new`:
     - Repository access: Only selected, your private logs repo
       (`moe-the-goat/job-scrapper-logs`).
     - Repository permissions: Contents → Read and Write.
     - Expiration: 1 year or longer.
   Copy the token value.

5. **Configure the worker's variables**. In the worker dashboard,
   Settings → Variables and Secrets, add:
     - `FEEDBACK_TOKEN` — paste the PAT from step 4. Mark it as a Secret
       so Cloudflare encrypts it.
     - `LOGS_REPO` — plain text, `moe-the-goat/job-scrapper-logs`.
     - `ALLOWED_ORIGIN` — plain text, `https://moe-the-goat.github.io`
       (no trailing slash). Only requests from this origin are accepted.
   Click Save and Deploy.

6. **Tell the pipeline where the worker is**. In the code repo, go to
   Settings → Secrets and variables → Actions → Variables tab, add:
     - `FEEDBACK_WORKER_URL` — the URL from step 2.
   This is a Variable, not a Secret. The URL is intentionally public,
   it ends up embedded in the static feedback page anyway.

7. **Delete the old `FEEDBACK_WRITE_TOKEN` secret** from the code repo
   Settings → Secrets. The new architecture does not use it.

## Testing the worker

After deploying, test with `curl` from your terminal:

```bash
curl -X POST https://job-feedback.<your-subdomain>.workers.dev \
  -H "Content-Type: application/json" \
  -H "Origin: https://moe-the-goat.github.io" \
  -d '{"entries":[{"job_url":"https://test","company":"Test","title":"Test","feedback":"applied"}]}'
```

Expected response: `{"ok": true, "count": 1}`. After running, the file
`data/feedback_pending.json` in the private logs repo should contain that
one entry.

## What stops abuse

The worker accepts submissions only from the configured `ALLOWED_ORIGIN`
(the GitHub Pages domain). A malicious script on another origin would be
blocked by browser CORS enforcement, and the worker double-checks the
`Origin` header. The submission cap of 100 entries per call prevents a
single request from dumping a multi-megabyte payload.

If you ever need to invalidate access, regenerate the PAT in step 4 and
update the worker's `FEEDBACK_TOKEN` variable. The Worker URL itself does
not need to change.
