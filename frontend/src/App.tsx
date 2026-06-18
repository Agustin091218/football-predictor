import { useEffect, useState, useCallback } from "react";
import { usePrediction } from "./hooks/usePrediction";
import { MatchBar } from "./components/MatchBar";
import { Callout } from "./components/Callout";
import { MonteCarloCard } from "./components/MonteCarloCard";
import { GeminiCard } from "./components/GeminiCard";
import { Tooltip } from "./components/Tooltip";
import { WorldCupPanel } from "./components/WorldCupPanel";
import { TrainingPanel } from "./components/TrainingPanel";
import { AnalyticsDashboard } from "./components/AnalyticsDashboard";
import { getHealth } from "./lib/api";

type Tab = "wc" | "predict" | "training" | "analytics";
const RESULTS_KEY = "wc2026_results";
const SCORES_KEY = "wc2026_scores";

function calculateScore(pred: { prob_home_win: number; prob_draw: number; prob_away_win: number; confidence: number; predicted_result?: string; expected_goals_home?: number; expected_goals_away?: number; simulation?: { prob_over_2_5?: number; prob_btts?: number; prob_clean_sheet_home?: number; prob_clean_sheet_away?: number; top_scores?: { score: string }[] } }, realHG: number, realAG: number) {
  const conf = pred.confidence || 0;
  const predRes = pred.predicted_result || (pred.prob_home_win > pred.prob_draw && pred.prob_home_win > pred.prob_away_win ? "1" : pred.prob_draw > pred.prob_away_win ? "X" : "2");
  const realRes = realHG > realAG ? "1" : realHG === realAG ? "X" : "2";
  const correct = predRes === realRes;
  const sim = pred.simulation || {};
  let pts = 0;
  const bd: { l: string; pts: number; t: string }[] = [];
  if (correct) {
    const top = sim.top_scores?.[0]?.score?.split("-");
    const exact = top && parseInt(top[0]) === realHG && parseInt(top[1]) === realAG;
    const xgDiff = Math.abs((pred.expected_goals_home || 0) - (pred.expected_goals_away || 0) - (realHG - realAG));
    if (exact) { pts += 100; bd.push({ l: "Marcador exacto", pts: 100, t: "success" }); }
    else if (xgDiff <= 1) { pts += 60; bd.push({ l: "Resultado + goles", pts: 60, t: "success" }); }
    else { pts += 30; bd.push({ l: "Solo 1X2", pts: 30, t: "partial" }); }
  } else {
    if (conf >= 0.9) { pts -= 40; bd.push({ l: "Error crítico", pts: -40, t: "error" }); }
    else if (conf >= 0.75) { pts -= 20; bd.push({ l: "Error grave", pts: -20, t: "error" }); }
    else { bd.push({ l: "Resultado incorrecto", pts: 0, t: "neutral" }); }
  }
  const xgErr = Math.abs((pred.expected_goals_home || 0) - (pred.expected_goals_away || 0) - (realHG - realAG));
  if (xgErr <= 0.5) { pts += 20; bd.push({ l: "xG muy preciso", pts: 20, t: "success" }); }
  else if (xgErr <= 1.0) { pts += 10; bd.push({ l: "xG aproximado", pts: 10, t: "partial" }); }
  else if (xgErr > 2.0) { pts -= 10; bd.push({ l: "xG impreciso", pts: -10, t: "error" }); }
  const overR = (realHG + realAG) > 2.5, overP = (sim.prob_over_2_5 || 0) > 0.5; if (overR === overP) { pts += 15; bd.push({ l: "Over/Under 2.5", pts: 15, t: "success" }); }
  const bttsR = realHG > 0 && realAG > 0, bttsP = (sim.prob_btts || 0) > 0.5; if (bttsR === bttsP) { pts += 10; bd.push({ l: "BTTS", pts: 10, t: "success" }); }
  const cleanHR = realAG === 0, cleanHP = (sim.prob_clean_sheet_home || 0) > 0.5; if (cleanHR === cleanHP) { pts += 10; bd.push({ l: "Clean sheet local", pts: 10, t: "success" }); }
  const cleanAR = realHG === 0, cleanAP = (sim.prob_clean_sheet_away || 0) > 0.5; if (cleanAR === cleanAP) { pts += 10; bd.push({ l: "Clean sheet visit", pts: 10, t: "success" }); }
  return { pts, breakdown: bd, correct, realRes };
}

export default function App() {
  const { prediction, loading, history, predict, refreshHistory } = usePrediction();
  const [home, setHome] = useState("Germany");
  const [away, setAway] = useState("Curaçao");
  const [backendOnline, setBackendOnline] = useState(false);
  const [xgboostTrained, setXgboostTrained] = useState(false);
  const [llmAvailable, setLlmAvailable] = useState(false);
  const [tab, setTab] = useState<Tab>("wc");
  const [resultModal, setResultModal] = useState<{ matchId: string; home: string; away: string } | null>(null);
  const [resultHG, setResultHG] = useState(0);
  const [resultAG, setResultAG] = useState(0);
  const [, forceUpdate] = useState(0);

  useEffect(() => {
    getHealth().then((h) => { if (h) { setBackendOnline(true); setXgboostTrained(h.xgboost_trained); setLlmAvailable(h.llm_available); } });
    refreshHistory();
  }, [refreshHistory]);

  const handlePredict = (e: React.FormEvent) => { e.preventDefault(); predict(home, away, true); };
  const doPredict = useCallback((h: string, a: string) => { setHome(h); setAway(a); setTab("predict"); predict(h, a); }, [predict]);

  const submitResult = () => {
    if (!resultModal) return;
    const { matchId, home: ht, away: at } = resultModal;
    const results = JSON.parse(localStorage.getItem(RESULTS_KEY) || "{}");
    results[matchId] = { homeGoals: resultHG, awayGoals: resultAG, enteredAt: new Date().toISOString() };
    localStorage.setItem(RESULTS_KEY, JSON.stringify(results));
    const cache = JSON.parse(localStorage.getItem("wc2026_preds") || "{}");
    const cachedArr = Object.values(cache) as { home: string; away: string; prediction?: typeof prediction }[];
    const cached = cachedArr.find((v) => v.home === ht && v.away === at);
    if (cached?.prediction) {
      const sc = calculateScore(cached.prediction, resultHG, resultAG);
      const scores = JSON.parse(localStorage.getItem(SCORES_KEY) || "{}");
      scores[matchId] = { ...sc, matchId, home: ht, away: at, realHome: resultHG, realAway: resultAG, evaluatedAt: new Date().toISOString() };
      localStorage.setItem(SCORES_KEY, JSON.stringify(scores));
    }
    setResultModal(null); forceUpdate((n) => n + 1); refreshHistory();
  };

  return (
    <div className="min-h-screen bg-[#0D0F14] text-gray-200">
      <header className="sticky top-0 z-50 bg-[#161B26] border-b border-[#252D3D] px-6 h-16 flex items-center justify-between">
        <div><h1 className="text-lg font-bold">⚽ Football Predictor</h1><p className="text-[10px] text-gray-600">Poisson · Monte Carlo · Gemini</p></div>
        <div className="flex gap-1">{(["wc","predict","training","analytics"] as Tab[]).map((t)=>(<button key={t} onClick={()=>setTab(t)} className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${tab===t?"bg-emerald-500 text-black":"text-gray-400 hover:text-gray-200"}`}>{t==="wc"?"🏆 Copa del Mundo":t==="predict"?"📊 Predicciones":t==="training"?"🧠 Entrenamiento":"📈 Analytics"}</button>))}</div>
        <div className="flex gap-2"><span className={`text-[10px] px-2 py-1 rounded-full ${backendOnline?"bg-emerald-900 text-emerald-400":"bg-red-900 text-red-400"}`}>{backendOnline?"✅ Online":"❌ Offline"}</span>{!xgboostTrained&&<span className="text-[10px] px-2 py-1 rounded-full bg-amber-900 text-amber-400">⚠️ XGBoost sin entrenar</span>}{!llmAvailable&&<span className="text-[10px] px-2 py-1 rounded-full bg-gray-800 text-gray-400">LLM off</span>}</div>
      </header>
      {tab==="wc"&&<WorldCupPanel onPredict={doPredict} onAddResult={(mid,h,a)=>{setResultModal({matchId:mid,home:h,away:a});setResultHG(0);setResultAG(0)}}/>}
      {tab==="predict"&&(
        <div className="grid grid-cols-[280px_1fr_320px] min-h-[calc(100vh-64px)] max-lg:grid-cols-1">
          <aside className="bg-[#161B26] border-r border-[#252D3D] p-5 flex flex-col gap-4 max-lg:border-r-0 max-lg:border-b">
            <form onSubmit={handlePredict} className="space-y-3"><input className="w-full bg-[#0D0F14] border border-[#252D3D] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500" placeholder="Equipo local" value={home} onChange={(e)=>setHome(e.target.value)}/><input className="w-full bg-[#0D0F14] border border-[#252D3D] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500" placeholder="Equipo visitante" value={away} onChange={(e)=>setAway(e.target.value)}/><button type="submit" disabled={loading} className="w-full bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-black font-semibold py-2.5 rounded-lg text-sm transition-colors">{loading?"⏳ Analizando...":"🔮 Predecir"}</button></form>
            <div><h3 className="text-xs font-semibold text-gray-500 mb-2">📋 Historial</h3><div className="space-y-1 max-h-[350px] overflow-y-auto">{history.slice(0,12).map((item)=>(<button key={item.cachedAt} onClick={()=>{predict(item.home,item.away);setHome(item.home);setAway(item.away)}} className="w-full text-left p-2 rounded-lg hover:bg-[#1C2333] transition-colors text-xs"><div className="font-medium">{item.home} vs {item.away}</div><div className="text-gray-500 text-[10px]">{new Date(item.cachedAt).toLocaleDateString("es-AR",{day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"})}</div></button>))}{history.length===0&&<p className="text-xs text-gray-600 text-center py-4">Sin predicciones aún</p>}</div></div>
          </aside>
          <main className="p-6 flex flex-col items-center gap-5 overflow-y-auto">
            {loading&&<div className="text-center py-20 text-gray-500"><div className="text-4xl mb-4">⏳</div><p>Generando predicción...</p></div>}
            {prediction&&!loading&&(<div className="w-full max-w-2xl space-y-5"><div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5 text-center"><div className="text-2xl font-bold">{prediction.match.home_team.name}<span className="text-gray-600 mx-2">vs</span>{prediction.match.away_team.name}</div><div className="text-xs text-gray-500 mt-1">{new Date(prediction.predicted_at).toLocaleString("es-AR")}</div></div><div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5"><MatchBar pred={prediction}/><div className="mt-4"><Callout pred={prediction}/></div></div><MonteCarloCard pred={prediction}/><div className="grid grid-cols-4 gap-3">{[["xG",`${(prediction.expected_goals_home??0).toFixed(1)} — ${(prediction.expected_goals_away??0).toFixed(1)}`],["Confianza",`${(prediction.confidence*100).toFixed(0)}%`],["Over 2.5",prediction.simulation?`${((prediction.simulation.prob_over_2_5??0)*100).toFixed(1)}%`:"—"],["BTTS",prediction.simulation?`${((prediction.simulation.prob_btts??0)*100).toFixed(1)}%`:"—"]].map(([t,v])=>(<Tooltip key={t} term={t}><div className="bg-[#161B26] border border-[#252D3D] rounded-lg p-3 text-center"><div className="text-lg font-mono font-bold">{v}</div><div className="text-[10px] text-gray-500">{t}</div></div></Tooltip>))}</div><GeminiCard pred={prediction}/></div>)}
            {!prediction&&!loading&&<div className="text-center py-20 text-gray-600"><div className="text-5xl mb-4">⚽</div><h2 className="text-xl font-semibold text-gray-500 mb-2">Ingresá dos equipos</h2><p className="text-sm">ELO · Poisson · Monte Carlo · Gemini</p></div>}
          </main>
          <aside className="bg-[#161B26] border-l border-[#252D3D] p-4 overflow-y-auto max-lg:border-l-0 max-lg:border-t"><h3 className="text-xs font-semibold text-gray-500 mb-3">📡 Señales del modelo</h3>{prediction?.signal_outputs?Object.entries(prediction.signal_outputs).filter(([k])=>!k.startsWith("_")).map(([name,sig])=>(<details key={name} open={name==="elo"||name==="poisson"} className="bg-[#1C2333] rounded-lg mb-2"><summary className="p-3 text-xs font-semibold cursor-pointer flex justify-between items-center"><span className="capitalize">{name}</span><span className="text-[10px] text-gray-500">{((sig.confidence??0)*100).toFixed(0)}%</span></summary><div className="px-3 pb-3 text-xs text-gray-400">{sig.summary}</div></details>)):<p className="text-xs text-gray-600 text-center py-8">Sin señales</p>}</aside>
        </div>
      )}
      {tab==="training"&&<TrainingPanel/>}{tab==="analytics"&&<AnalyticsDashboard/>}
      {resultModal&&(<div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center" onClick={()=>setResultModal(null)}><div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-6 max-w-sm w-full mx-4" onClick={(e)=>e.stopPropagation()}><h2 className="text-lg font-bold mb-1">📝 Ingresar resultado</h2><p className="text-sm text-gray-400 mb-4">{resultModal.home} vs {resultModal.away}</p><div className="flex items-center justify-center gap-3 mb-4"><span className="font-semibold">{resultModal.home}</span><input type="number" min={0} max={20} value={resultHG} onChange={(e)=>setResultHG(parseInt(e.target.value)||0)} className="w-16 bg-[#0D0F14] border border-[#252D3D] rounded-lg px-3 py-2 text-center text-lg font-mono focus:outline-none focus:border-emerald-500"/><span className="text-xl font-bold">—</span><input type="number" min={0} max={20} value={resultAG} onChange={(e)=>setResultAG(parseInt(e.target.value)||0)} className="w-16 bg-[#0D0F14] border border-[#252D3D] rounded-lg px-3 py-2 text-center text-lg font-mono focus:outline-none focus:border-emerald-500"/><span className="font-semibold">{resultModal.away}</span></div><button onClick={submitResult} className="w-full bg-emerald-500 hover:bg-emerald-400 text-black font-semibold py-2.5 rounded-lg text-sm">💾 Guardar</button><button onClick={()=>setResultModal(null)} className="w-full mt-2 bg-[#1C2333] border border-[#252D3D] text-gray-400 py-2 rounded-lg text-sm">Cancelar</button></div></div>)}
    </div>
  );
}
