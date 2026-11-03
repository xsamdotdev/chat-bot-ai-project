import type {
  ChatResponse,
  DeleteResponse,
  DocumentListResponse,
  HealthResponse,
  HistoryMessage,
  UploadResponse,
} from "./types";

/**
 * Empty by default: the app calls `/api/...` on its own origin.
 * - Docker: nginx proxies `/api` to the backend container.
 * - Dev: Vite proxies `/api` to http://localhost:8000.
 * Set VITE_API_BASE_URL (build-time) to point at a separate API host instead.
 */
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(
      "Could not reach the server. Check that the backend is running.",
      0,
    );
  }

  const raw = await response.text();
  let payload: unknown = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = raw;
    }
  }

  if (!response.ok) {
    throw new ApiError(
      extractDetail(payload) ?? `Request failed (HTTP ${response.status}).`,
      response.status,
    );
  }

  return payload as T;
}

function extractDetail(payload: unknown): string | null {
  if (!payload) return null;
  if (typeof payload === "string") return payload;
  if (typeof payload !== "object") return null;

  if ("detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return JSON.stringify(item);
        })
        .join("; ");
    }
    if (detail != null) return JSON.stringify(detail);
  }

  return null;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function listDocuments(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>("/api/documents");
}

export function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<UploadResponse>("/api/documents/upload", {
    method: "POST",
    body: form,
  });
}

export function deleteDocument(docId: string): Promise<DeleteResponse> {
  return request<DeleteResponse>(`/api/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
}

export function sendChat(
  query: string,
  history: HistoryMessage[],
  topK?: number,
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      history,
      ...(topK ? { top_k: topK } : {}),
    }),
  });
}
