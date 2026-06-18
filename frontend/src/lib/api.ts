import type { PredictionResponse } from "../types";

const BASE = "";

export async function apiGet<T>(path: string): Promise<T | null> {
  try {
    const r = await fetch(`${BASE}${path}`);
    if (!r.ok) throw new Error(`${r.status}`);
    return await r.json();
  } catch {
    return null;
  }
}

export async function apiPost<T>(
  path: string,
  body?: Record<string, unknown>
): Promise<T | null> {
  try {
    const r = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      throw new Error(`${r.status}: ${text.slice(0, 200)}`);
    }
    return await r.json();
  } catch (e) {
    console.error("[apiPost]", path, e);
    throw e;
  }
}

export async function predictCustom(
  home: string,
  away: string
): Promise<PredictionResponse> {
  return apiPost<PredictionResponse>(
    `/predict/custom?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`
  ) as Promise<PredictionResponse>;
}

export async function getHealth() {
  return apiGet<{
    status: string;
    xgboost_trained: boolean;
    llm_available: boolean;
    supported_leagues: number;
  }>("/health");
}

export async function getHistory(limit = 20) {
  return apiGet<PredictionResponse[]>(
    `/predictions/history?limit=${limit}`
  );
}
