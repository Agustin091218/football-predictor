import { useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, ResponsiveContainer } from "recharts";
import { Tooltip } from "./Tooltip";

const CACHE_KEY = "wc2026_preds";
const RESULTS_KEY = "wc2026_results";
const SCORES_KEY = "wc2026_scores";

interface EvalPrediction {
  home: string; away: string; matchId: string;
  prediction: { prob_home_win: number; prob_draw: number; prob_away_win: number; confidence: number; predicted_result?: string; expected_goals_home?: number; expected_goals_away?: number; simulation?: { prob_over_2_5?: number; prob_btts?: number; prob_clean_sheet_home?: number; prob_clean_sheet_away?: number; top_scores?: { score: string }[] }; signal_outputs?: Record<string, { value?: Record<string, unknown>; confidence?: number }>; model_version?: string };
  correct: boolean; actual: string; hg: number; ag: number;
  prob: number; logLoss: number; brier: number; upsetScore: number; predicted: string;
}

function getEvaluated(): EvalPrediction[] {
  try {
    const preds = Object.values(JSON.parse(localStorage.getItem(CACHE_KEY) || "{}")) as { home: string; away: string; matchId?: string; prediction: EvalPrediction["prediction"] }[];
    const results = JSON.parse(localStorage.getItem(RESULTS_KEY) || "{}");
    return preds.filter((p) => p.matchId && results[p.matchId]).map((p) => {
      const r = results[p.matchId!]; const hg = r.homeGoals, ag = r.awayGoals;
      const actual = hg > ag ? "1" : hg === ag ? "X" : "2";
      const prob = actual === "1" ? p.prediction.prob_home_win : actual === "X" ? p.prediction.prob_draw : p.prediction.prob_away_win;
      const pred = p.prediction.predicted_result || (p.prediction.prob_home_win > p.prediction.prob_draw && p.prediction.prob_home_win > p.prediction.prob_away_win ? "1" : p.prediction.prob_draw > p.prediction.prob_away_win ? "X" : "2");
      const av = actual === "1" ? [1, 0, 0] : actual === "X" ? [0, 1, 0] : [0, 0, 1];
      const pv = [p.prediction.prob_home_win || 0.33, p.prediction.prob_draw || 0.33, p.prediction.prob_away_win || 0.33];
      return { ...p, home: p.home, away: p.away, matchId: p.matchId!, correct: pred === actual, actual, hg, ag, prob: prob || 0.33, logLoss: -Math.log(Math.max(prob || 0.33, 0.001)), brier: pv.reduce((s, v, i) => s + (v - av[i]) ** 2, 0), upsetScore: 1 - (prob || 0.33), predicted: pred };
    });
  } catch { return []; }
}

function getScores() {
  try { return Object.values(JSON.parse(localStorage.getItem(SCORES_KEY) || "{}")) as { matchId: string; pts: number; home: string; away: string; realHome: number; realAway: number; breakdown: { l: string; pts: number; t: string }[] }[]; } catch { return []; }
}

const LEVELS = [{ max: 200, n: "Principiante" }, { max: 500, n: "Aprendiz" }, { max: 1000, n: "Analista" }, { max: 2000, n: "Experto" }, { max: 99999, n: "Maestro predictor" }];

export function TrainingPanel() {
  const ev = useMemo(() => getEvaluated(), []);
  const scores = useMemo(() => getScores(), []);
  const totalPts = scores.reduce((s, r) => s + (r.pts || 0), 0);
  const n = ev.length;
  const [expanded, setExpanded] = useState<number | null>(null);

  const lvlIdx = LEVELS.findIndex((l) => totalPts < l.max);
  const lvl = LEVELS[lvlIdx >= 0 ? lvlIdx : LEVELS.length - 1];
  const prevMax = lvlIdx > 0 ? LEVELS[lvlIdx - 1].max : 0;
  const nextLvl = LEVELS[lvlIdx + 1];
  const lvlPct = Math.min(100, Math.max(0, Math.round(((totalPts - prevMax) / (lvl.max - prevMax)) * 100)));

  if (n === 0 && scores.length === 0) {
    return (
      <div className="max-w-3xl mx-auto p-6 text-center py-20 space-y-6">
        <div className="text-6xl mb-2">⚽</div>
        <h2 className="text-xl font-semibold text-gray-400">Sin predicciones evaluadas todavía</h2>
        <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-6 max-w-md mx-auto text-left text-sm text-gray-400 space-y-3">
          <p><span className="text-emerald-400 font-bold">1.</span> Generá una predicción en <b>📊 Predicciones</b> o <b>🏆 Copa del Mundo</b></p>
          <p><span className="text-emerald-400 font-bold">2.</span> Cuando el partido termine, ingresá el resultado real con el botón <b>📝</b></p>
          <p><span className="text-emerald-400 font-bold">3.</span> El sistema compara tu predicción vs realidad y te da puntos</p>
          <p><span className="text-emerald-400 font-bold">4.</span> Con 5+ partidos, el <b>calibrador isotónico</b> se activa y ajusta probabilidades</p>
        </div>
        <div className="text-xs text-gray-600">
          <p><b>Sistema de puntuación:</b></p>
          <p>Marcador exacto: <span className="text-emerald-400">+100</span> · Over 2.5: <span className="text-emerald-400">+15</span> · BTTS: <span className="text-emerald-400">+10</span></p>
          <p>Error con alta confianza: <span className="text-red-400">-20 a -40</span></p>
        </div>
      </div>
    );
  }

  const correct = ev.filter((e) => e.correct).length;
  const incorrect = n - correct;
  const acc = n > 0 ? correct / n : 0;
  const ll = n > 0 ? ev.reduce((s, e) => s + e.logLoss, 0) / n : 0;
  const br = n > 0 ? ev.reduce((s, e) => s + e.brier, 0) / n : 0;
  const bestStreak = Math.max(...ev.reduce((a: number[], e) => { if (e.correct) a[a.length - 1]++; else a.push(0); return a; }, [0]));

  const curveData = scores.map((_, i) => {
    const s = scores.slice(0, i + 1);
    return { match: i + 1, pts: s.reduce((a, x) => a + (x.pts || 0), 0), random: (i + 1) * 15 };
  });

  const errors = ev.filter((e) => !e.correct);
  const drawErrs = errors.filter((e) => e.actual === "X").length;
  const hiConfErrs = errors.filter((e) => (e.prediction.confidence ?? 0) >= 0.75).length;
  const modelVersions = [...new Set(ev.map((e) => e.prediction.model_version || "poisson-v1"))];

  return (
    <div className="max-w-5xl mx-auto p-4 space-y-4 overflow-y-auto" style={{ maxHeight: "calc(100vh - 80px)" }}>
      {/* SCORE HEADER */}
      <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5 text-center">
        <div className={`text-5xl font-mono font-bold ${totalPts > 0 ? "text-emerald-400" : "text-gray-500"}`}>{totalPts} pts</div>
        <div className="text-sm text-gray-400 mt-1">Nivel: <b className="text-gray-200">{lvl.n}</b> · {scores.length} partidos evaluados</div>
        <div className="h-3 bg-[#1C2333] rounded-full mt-3 overflow-hidden">
          <div className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full transition-all duration-700" style={{ width: `${lvlPct}%` }} />
        </div>
        <div className="text-[11px] text-gray-500 mt-1">{totalPts - prevMax}/{lvl.max - prevMax} pts para <b>{nextLvl?.n || "máximo"}</b></div>
      </div>

      {/* METRICS ROW with tooltips */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        <Tooltip term="Accuracy"><div className="bg-[#161B26] border border-[#252D3D] rounded-lg p-3 text-center hover:border-emerald-500/50 transition-colors cursor-help">
          <div className="text-xl font-mono font-bold text-emerald-400">{n}</div><div className="text-[10px] text-gray-500">Evaluados</div></div></Tooltip>
        <Tooltip term="Accuracy"><div className="bg-[#161B26] border border-[#252D3D] rounded-lg p-3 text-center hover:border-emerald-500/50 transition-colors cursor-help">
          <div className="text-lg font-mono font-bold">{(acc * 100).toFixed(0)}%</div><div className="text-[10px] text-gray-500">Accuracy</div>
          <div className="text-[9px] text-gray-600 mt-0.5">{correct}✅ {incorrect}❌</div></div></Tooltip>
        <Tooltip term="LogLoss"><div className="bg-[#161B26] border border-[#252D3D] rounded-lg p-3 text-center hover:border-blue-500/50 transition-colors cursor-help">
          <div className="text-lg font-mono font-bold">{ll.toFixed(3)}</div><div className="text-[10px] text-gray-500">LogLoss</div>
          <div className="text-[9px] text-gray-600 mt-0.5">aleatorio 1.099</div></div></Tooltip>
        <Tooltip term="Brier Score"><div className="bg-[#161B26] border border-[#252D3D] rounded-lg p-3 text-center hover:border-blue-500/50 transition-colors cursor-help">
          <div className="text-lg font-mono font-bold">{br.toFixed(3)}</div><div className="text-[10px] text-gray-500">Brier</div>
          <div className="text-[9px] text-gray-600 mt-0.5">aleatorio 0.667</div></div></Tooltip>
        <div className="bg-[#161B26] border border-[#252D3D] rounded-lg p-3 text-center">
          <div className="text-lg font-mono font-bold text-amber-400">{bestStreak}</div><div className="text-[10px] text-gray-500">Mejor racha</div></div>
        <div className="bg-[#161B26] border border-[#252D3D] rounded-lg p-3 text-center">
          <div className="text-lg font-mono font-bold">{n > 0 ? Math.round(totalPts / n) : 0}</div><div className="text-[10px] text-gray-500">Pts/partido</div></div>
      </div>

      {/* HOW IT WORKS — models & calibration */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-4">
          <h3 className="text-sm font-semibold mb-2">🧠 ¿Cómo predice?</h3>
          <div className="text-xs text-gray-400 space-y-1.5">
            <p><span className="text-emerald-400">Poisson (Dixon-Coles):</span> estima goles desde ELO</p>
            <p><span className="text-blue-400">Monte Carlo:</span> 10,000 simulaciones</p>
            <p><span className="text-amber-400">XGBoost:</span> 54 features, 3,000 partidos INTL</p>
            <p><span className="text-purple-400">Gemini:</span> análisis + Google Search</p>
            <p><span className="text-gray-500">Usados:</span> {modelVersions.join(", ")}</p>
          </div>
        </div>
        <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-4">
          <h3 className="text-sm font-semibold mb-2">📐 ¿Cómo aprende?</h3>
          <div className="text-xs text-gray-400 space-y-1.5">
            <p>1. Predice → guarda en localStorage</p>
            <p>2. Resultado real → compara predicción</p>
            <p>3. Calcula <span className="text-emerald-400">log-loss</span> y <span className="text-blue-400">Brier</span></p>
            <p>4. Da <span className="text-amber-400">puntos</span> según precisión</p>
            <p>5. Con ≥5 partidos: <span className="text-purple-400">calibrador isotónico</span></p>
          </div>
        </div>
        <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-4">
          <h3 className="text-sm font-semibold mb-2">⚙️ Calibración</h3>
          <div className="text-xs text-gray-400 space-y-1.5">
            {n >= 5 ? (
              <>
                <p><span className="text-emerald-400">✅ ACTIVA</span> — {n} partidos</p>
                <p>Ajusta probabilidades automáticamente</p>
                <p className="text-gray-500">Prob &gt;70%: reduce sobre-confianza</p>
                <p className="text-gray-500">Prob &lt;30%: aumenta sub-confianza</p>
                <p className="text-gray-500">Usa <span className="text-blue-400">IsotonicRegression</span> (one-vs-rest)</p>
              </>
            ) : (
              <>
                <p><span className="text-amber-400">⚠️ INACTIVA</span></p>
                <div className="h-2 bg-[#1C2333] rounded-full mt-1 overflow-hidden"><div className="h-full bg-amber-500 rounded-full" style={{ width: `${(n / 5) * 100}%` }} /></div>
                <p className="mt-1">Faltan {5 - n} partidos para activar</p>
              </>
            )}
          </div>
        </div>
      </div>

        {/* CHARTS ROW */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {scores.length >= 2 && (
          <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-4">
            <h3 className="text-sm font-semibold mb-3">📈 Curva de aprendizaje</h3>
            <p className="text-[10px] text-gray-500 mb-2">Puntos acumulados partido a partido vs referencia aleatoria (~15 pts/partido)</p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={curveData}>
                <CartesianGrid stroke="#252D3D" strokeDasharray="3 3" />
                <XAxis dataKey="match" tick={{ fill: "#4A5568", fontSize: 10 }} />
                <YAxis tick={{ fill: "#4A5568", fontSize: 10 }} />
                <ReTooltip contentStyle={{ background: "#0D0F14", border: "1px solid #252D3D", borderRadius: 8, fontSize: 11 }} />
                <Line type="monotone" dataKey="pts" stroke="#00C896" strokeWidth={2} dot={{ r: 3 }} name="Puntos" />
                <Line type="monotone" dataKey="random" stroke="#4A5568" strokeDasharray="5 5" dot={false} name="Aleatorio" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* ERROR ANALYSIS */}
      {errors.length >= 1 && (
        <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-4">
          <h3 className="text-sm font-semibold mb-3">🔍 Análisis de errores</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div>
              <div className="text-gray-500 mb-1">Por nivel de confianza</div>
              {[{ l: "Alta (>75%)", lo: 0.75, hi: 1 }, { l: "Media (50-75%)", lo: 0.5, hi: 0.75 }, { l: "Baja (<50%)", lo: 0, hi: 0.5 }].map((b) => {
                const inBin = errors.filter((e) => (e.prediction.confidence ?? 0) >= b.lo && (e.prediction.confidence ?? 0) < b.hi);
                const total = ev.filter((e) => (e.prediction.confidence ?? 0) >= b.lo && (e.prediction.confidence ?? 0) < b.hi);
                const pct = total.length > 0 ? inBin.length / total.length : 0;
                return <div key={b.l} className="flex items-center gap-2 my-1"><span className="w-24 text-gray-400">{b.l}</span><div className="flex-1 h-2 bg-[#1C2333] rounded-full overflow-hidden"><div className="h-full bg-red-500 rounded-full" style={{ width: `${pct * 100}%` }} /></div><span className="text-gray-500 w-10 text-right">{inBin.length}/{total.length}</span></div>;
              })}
            </div>
            <div>
              <div className="text-gray-500 mb-1">Tipo de error</div>
              {drawErrs > 0 && <div className="text-amber-400 my-1">⚠️ {drawErrs} empates no predichos</div>}
              {hiConfErrs > 0 && <div className="text-amber-400 my-1">⚠️ {hiConfErrs} errores con confianza alta</div>}
              <div className="text-gray-400 my-1">Predijo "1" → erró {errors.filter((e) => e.predicted === "1").length}x</div>
              <div className="text-gray-400 my-1">Predijo "X" → erró {errors.filter((e) => e.predicted === "X").length}x</div>
              <div className="text-gray-400 my-1">Predijo "2" → erró {errors.filter((e) => e.predicted === "2").length}x</div>
            </div>
            <div>
              <div className="text-gray-500 mb-1">Señal más contradictoria</div>
              {(() => {
                const sigErrs: Record<string, number> = {};
                for (const e of errors) {
                  const so = e.prediction.signal_outputs || {};
                  for (const [nm, sig] of Object.entries(so)) {
                    const v = sig.value as Record<string, number> | undefined || {};
                    const act = e.actual;
                    const p = act === "1" ? v.prob_home_win : act === "X" ? v.prob_draw : v.prob_away_win;
                    if (p != null && p < 0.5) sigErrs[nm] = (sigErrs[nm] || 0) + 1;
                  }
                }
                return Object.entries(sigErrs).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([nm, c]) => (
                  <div key={nm} className="flex items-center gap-2 my-1"><span className="w-16 text-gray-400 capitalize">{nm}</span><div className="flex-1 h-2 bg-[#1C2333] rounded-full overflow-hidden"><div className="h-full bg-red-500/60 rounded-full" style={{ width: `${Math.min(100, (c / Math.max(1, errors.length)) * 100)}%` }} /></div><span className="text-gray-500">{c}x</span></div>
                ));
              })()}
            </div>
          </div>
        </div>
      )}

      {/* SCORE BREAKDOWN */}
      {scores.length > 0 && (
        <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-4">
          <h3 className="text-sm font-semibold mb-3">📋 Historial de predicciones evaluadas</h3>
          <div className="space-y-1">
            {[...scores].reverse().slice(0, 10).map((s, i) => {
              const evMatch = ev.find((e) => e.matchId === s.matchId);
              const isOpen = expanded === i;
              return (
                <details key={i} className="text-xs" open={isOpen} onToggle={() => setExpanded(isOpen ? null : i)}>
                  <summary className="cursor-pointer flex justify-between py-1.5 border-b border-[#1a2030] hover:bg-[#1C2333]/50 px-2 rounded">
                    <span>{s.pts >= 0 ? "✅" : "❌"} <b>{s.home}</b> {s.realHome}-{s.realAway} <b>{s.away}</b></span>
                    <span className={`font-bold font-mono ${s.pts >= 0 ? "text-emerald-400" : "text-red-400"}`}>{s.pts >= 0 ? "+" : ""}{s.pts} pts</span>
                  </summary>
                  <div className="pl-4 py-2 space-y-1.5 bg-[#1C2333]/30 rounded-b-lg">
                    <div className="text-gray-400">Desglose de puntos:</div>
                    {s.breakdown?.map((b, j) => (
                      <div key={j} className={b.t === "success" ? "text-emerald-400" : b.t === "error" ? "text-red-400" : "text-gray-500"}>
                        {b.t === "success" ? "✅" : b.t === "error" ? "❌" : "•"} {b.l}: <span className="font-mono">{b.pts > 0 ? "+" : ""}{b.pts} pts</span>
                      </div>
                    ))}
                    {evMatch && (
                      <div className="mt-2 pt-2 border-t border-[#252D3D] text-gray-500 space-y-0.5">
                        <div>Predijo: <span className="text-gray-400">{evMatch.predicted === "1" ? "Victoria local" : evMatch.predicted === "X" ? "Empate" : "Victoria visitante"}</span> · Confianza {(evMatch.prediction.confidence * 100).toFixed(0)}%</div>
                        <div>xG: {(evMatch.prediction.expected_goals_home ?? 0).toFixed(1)} — {(evMatch.prediction.expected_goals_away ?? 0).toFixed(1)} · LogLoss: {evMatch.logLoss.toFixed(3)}</div>
                        {evMatch.upsetScore > 0.5 && <div className="text-amber-400">⚡ Sorpresa: el modelo solo daba {(evMatch.prob * 100).toFixed(0)}% a este resultado</div>}
                      </div>
                    )}
                  </div>
                </details>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
