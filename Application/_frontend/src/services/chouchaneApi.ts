/**
 * Chouchane (chachia_agent) API client.
 * Talks to the Tunisia Tourism AI backend: Yasmine → Q&A.
 */

const BASE_URL = import.meta.env.VITE_CHOUCHANE_API_URL ?? "http://localhost:8000";

export type ChouchanePhase = "yasmine" | "qa";

export interface ChouchaneResponse {
  session_id: string;
  workflow: string;
  phase: ChouchanePhase;
  reply: string;
  partners?: string | null;
  chosen_place?: string | null;
}

async function request<T>(
  path: string,
  options: RequestInit & { body?: object } = {}
): Promise<T> {
  const { body, ...rest } = options;
  const res = await fetch(`${BASE_URL}${path}`, {
    ...rest,
    headers: {
      "Content-Type": "application/json",
      ...rest.headers,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

/** Start a new Chouchane session. Returns greeting + session_id. */
export async function sessionStart(): Promise<ChouchaneResponse> {
  return request<ChouchaneResponse>("/session/start", { method: "POST" });
}

/** Reset session back to Phase 1 (Chouchene). */
export async function sessionReset(sessionId: string): Promise<ChouchaneResponse> {
  return request<ChouchaneResponse>("/session/reset", {
    method: "POST",
    body: { session_id: sessionId, message: "" },
  });
}

/** Phase 1: Chat with Chouchene (recommendations). */
export async function choucheneChat(
  sessionId: string,
  message: string
): Promise<ChouchaneResponse> {
  return request<ChouchaneResponse>("/yasmine", {
    method: "POST",
    body: { session_id: sessionId, message },
  });
}

/** Phase 3: Tunisia Q&A — ask anything about Tunisia. */
export async function qaChat(
  sessionId: string,
  message: string
): Promise<ChouchaneResponse> {
  return request<ChouchaneResponse>("/qa", {
    method: "POST",
    body: { session_id: sessionId, message },
  });
}

/** Health check. */
export async function health(): Promise<{ status: string; workflow: string }> {
  return request("/health");
}

/** GET /session/:id — fetch session state (for restoring conversation in UI). */
export interface ChouchaneSessionState {
  session_id: string;
  phase: ChouchanePhase;
  yasmine_history: { role: string; text: string }[];
  qa_history: { role: string; text: string }[];
}

export async function getSession(sessionId: string): Promise<ChouchaneSessionState> {
  return request<ChouchaneSessionState>(`/session/${sessionId}`);
}

/** GET /places — all Tunisia places from chachia_agent (for lists, search, etc.). */
export interface PlaceFromApi {
  name: string;
  region: string;
  vibe: string;
  description: string;
  top_activities: string[];
  insider_tip: string;
  season: string;
}

export async function getPlaces(): Promise<{ places: PlaceFromApi[] }> {
  return request<{ places: PlaceFromApi[] }>("/places");
}

export { BASE_URL as chouchaneApiBaseUrl };
