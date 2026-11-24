// Thin client for /api/v2. All responses are JSON except the SSE stream,
// which is read with fetch() + ReadableStream because EventSource cannot POST.

const BASE = "/api/v2";

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || "HTTP " + status);
    this.status = status;
  }
}

async function request(method, path, body, isForm) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) {
      opts.body = body;
    } else {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
  }
  const res = await fetch(BASE + path, opts);
  if (res.status === 204) return null;
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  if (!res.ok) {
    const detail = data && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    throw new ApiError(res.status, detail);
  }
  return data;
}

export const api = {
  get: (p) => request("GET", p),
  post: (p, b) => request("POST", p, b === undefined ? {} : b),
  put: (p, b) => request("PUT", p, b),
  patch: (p, b) => request("PATCH", p, b),
  del: (p) => request("DELETE", p),
  upload: (p, form) => request("POST", p, form, true),
};

// POST and consume Server-Sent Events. onEvent(name, data) is called per
// event. Returns when the stream ends. `signal` (AbortController) stops it.
export async function streamSSE(path, body, onEvent, signal) {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = "HTTP " + res.status;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not json */ }
    throw new ApiError(res.status, detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let name = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) {
        try { onEvent(name, JSON.parse(data)); } catch (e) { console.error("bad SSE data", e); }
      }
    }
  }
}

export async function waitForJob(jobId, onProgress, timeoutMs = 300000) {
  const start = Date.now();
  for (;;) {
    const job = await api.get("/jobs/" + jobId);
    if (onProgress) onProgress(job);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) return job;
    if (Date.now() - start > timeoutMs) return job;
    await new Promise((r) => setTimeout(r, 700));
  }
}
