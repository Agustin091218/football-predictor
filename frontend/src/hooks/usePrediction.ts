import { useState, useCallback } from "react";
import type { PredictionResponse, CachedPrediction } from "../types";
import { predictCustom } from "../lib/api";

const CACHE_KEY = "wc2026_preds";

function canonicalName(input: string): string {
  const aliases: Record<string, string> = {
    alemania: "Germany", germany: "Germany", inglaterra: "England",
    brasil: "Brazil", argentina: "Argentina", españa: "Spain",
    francia: "France", italia: "Italy", portugal: "Portugal",
    holanda: "Netherlands", paísesbajos: "Netherlands",
    croacia: "Croatia", uruguay: "Uruguay", belgica: "Belgium",
    colombia: "Colombia", chile: "Chile", mexico: "Mexico",
    eeuu: "United States", usa: "United States",
  };
  const lower = input.toLowerCase().replace(/\s/g, "");
  return aliases[lower] || input.trim();
}

function getCacheKey(home: string, away: string): string {
  return `${canonicalName(home)}_vs_${canonicalName(away)}`;
}

function getCached(home: string, away: string): CachedPrediction | null {
  const cache = JSON.parse(localStorage.getItem(CACHE_KEY) || "{}");
  return cache[getCacheKey(home, away)] || null;
}

function saveCache(
  home: string,
  away: string,
  prediction: PredictionResponse
): void {
  const cache = JSON.parse(localStorage.getItem(CACHE_KEY) || "{}");
  cache[getCacheKey(home, away)] = {
    home: canonicalName(home),
    away: canonicalName(away),
    prediction,
    cachedAt: new Date().toISOString(),
  };
  localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
}

export function usePrediction() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fromCache, setFromCache] = useState(false);
  const [history, setHistory] = useState<CachedPrediction[]>([]);

  const refreshHistory = useCallback(() => {
    const cache = JSON.parse(localStorage.getItem(CACHE_KEY) || "{}");
    setHistory(
      Object.values(cache as Record<string, CachedPrediction>).sort(
        (a, b) =>
          new Date(b.cachedAt).getTime() - new Date(a.cachedAt).getTime()
      )
    );
  }, []);

  const predict = useCallback(
    async (home: string, away: string, forceNew = false) => {
      const h = canonicalName(home);
      const a = canonicalName(away);
      if (!h || !a) return;

      // Check cache
      if (!forceNew) {
        const cached = getCached(h, a);
        if (cached) {
          setPrediction(cached.prediction);
          setFromCache(true);
          setLoading(false);
          return;
        }
      }

      setLoading(true);
      setError(null);
      setFromCache(false);

      try {
        const data = await predictCustom(h, a);
        saveCache(h, a, data);
        setPrediction(data);
        refreshHistory();
      } catch (e) {
        setError(`Error: ${e instanceof Error ? e.message : "desconocido"}. ¿Está el backend corriendo?`);
      }
      setLoading(false);
    },
    [refreshHistory]
  );

  return { prediction, loading, error, fromCache, history, predict, refreshHistory };
}
