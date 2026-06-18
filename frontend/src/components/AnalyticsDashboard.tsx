import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar, Legend } from "recharts";
import { Tooltip } from "./ui/Tooltip";

// Mock data — in production this comes from localStorage + backend
const MOCK_EVALUATED = [
  { match: "Germany vs Curaçao", date: "14 Jun", predicted: "1", prob: 93, confidence: 74, realHome: 8, realAway: 0, correct: true, logLoss: 0.07, brier: 0.008 },
  { match: "Mexico vs South Africa", date: "11 Jun", predicted: "1", prob: 89, confidence: 68, realHome: 2, realAway: 0, correct: true, logLoss: 0.12, brier: 0.012 },
  { match: "Canada vs Bosnia Herz.", date: "12 Jun", predicted: "1", prob: 87, confidence: 72, realHome: 1, realAway: 1, correct: false, logLoss: 1.89, brier: 0.74 },
  { match: "Qatar vs Switzerland", date: "12 Jun", predicted: "2", prob: 66, confidence: 24, realHome: 1, realAway: 1, correct: false, logLoss: 0.98, brier: 0.31 },
  { match: "Brazil vs Morocco", date: "13 Jun", predicted: "1", prob: 71, confidence: 42, realHome: 1, realAway: 1, correct: false, logLoss: 1.24, brier: 0.38 },
  { match: "USA vs Paraguay", date: "12 Jun", predicted: "1", prob: 49, confidence: 8, realHome: 4, realAway: 1, correct: true, logLoss: 0.71, brier: 0.26 },
  { match: "Australia vs Turkey", date: "13 Jun", predicted: "1", prob: 52, confidence: 15, realHome: 2, realAway: 0, correct: true, logLoss: 0.65, brier: 0.23 },
  { match: "South Korea vs Czechia", date: "11 Jun", predicted: "1", prob: 66, confidence: 28, realHome: 2, realAway: 1, correct: true, logLoss: 0.42, brier: 0.11 },
];

export function AnalyticsDashboard() {
  const n = MOCK_EVALUATED.length;
  const correct = MOCK_EVALUATED.filter((e) => e.correct).length;
  const acc = n > 0 ? correct / n : 0;
  const ll = n > 0 ? MOCK_EVALUATED.reduce((s, e) => s + e.logLoss, 0) / n : 0;
  const br = n > 0 ? MOCK_EVALUATED.reduce((s, e) => s + e.brier, 0) / n : 0;

  const radarData = [
    { dim: "1X2", modelo: acc * 100, aleatorio: 33 },
    { dim: "Goles", modelo: 62, aleatorio: 50 },
    { dim: "Over/Under", modelo: 75, aleatorio: 50 },
    { dim: "BTTS", modelo: 71, aleatorio: 50 },
    { dim: "Confianza", modelo: 55, aleatorio: 33 },
  ];

  const barData = [
    { tipo: "Victoria\nlocal", pred: MOCK_EVALUATED.filter((e) => e.predicted === "1").length, ok: MOCK_EVALUATED.filter((e) => e.predicted === "1" && e.correct).length },
    { tipo: "Empate", pred: MOCK_EVALUATED.filter((e) => e.predicted === "X").length, ok: MOCK_EVALUATED.filter((e) => e.predicted === "X" && e.correct).length },
    { tipo: "Victoria\nvisitante", pred: MOCK_EVALUATED.filter((e) => e.predicted === "2").length, ok: MOCK_EVALUATED.filter((e) => e.predicted === "2" && e.correct).length },
  ];

  return (
    <div className="min-h-screen bg-[#0D0F14] text-gray-200">
      {/* Sidebar + Main */}
      <div className="flex">
        {/* Sidebar */}
        <aside className="w-56 bg-[#161B26] border-r border-[#252D3D] min-h-screen p-4 flex flex-col gap-2 sticky top-0">
          <div className="mb-6">
            <h1 className="text-lg font-bold">⚽ Football Predictor</h1>
            <p className="text-[10px] text-gray-600 mt-0.5">Analytics Dashboard</p>
          </div>
          {[
            { icon: "📊", label: "Dashboard", active: true },
            { icon: "🏆", label: "Copa del Mundo" },
            { icon: "🔮", label: "Predicciones" },
            { icon: "🧠", label: "Entrenamiento" },
            { icon: "⚙️", label: "Configuración" },
          ].map((item) => (
            <button
              key={item.label}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-left ${item.active ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "text-gray-400 hover:text-gray-200 hover:bg-[#1C2333]"}`}
            >
              <span>{item.icon}</span>
              {item.label}
            </button>
          ))}
          <div className="mt-auto pt-4 border-t border-[#252D3D]">
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Sistema online
            </div>
            <div className="text-[10px] text-gray-600 mt-1">v2.1.0 · 39 tests OK</div>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 p-6 space-y-5 overflow-y-auto max-h-screen">
          {/* Page header */}
          <div>
            <h2 className="text-xl font-bold">Football Predictor Analytics</h2>
            <p className="text-sm text-gray-500">Rendimiento del modelo · Mundial 2026 · {n} partidos evaluados</p>
          </div>

          {/* TOP ROW — Educational Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Card 1: How it predicts */}
            <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5 hover:border-emerald-500/30 transition-colors group">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xl">🧠</span>
                <h3 className="text-sm font-semibold">¿Cómo predice?</h3>
              </div>
              <div className="space-y-2 text-xs text-gray-400">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <Tooltip term="Poisson"><span className="font-medium text-gray-300">Poisson</span></Tooltip>
                  <span className="text-gray-600">— Dixon & Coles (1997)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-400" />
                  <Tooltip term="Monte Carlo"><span className="font-medium text-gray-300">Monte Carlo</span></Tooltip>
                  <span className="text-gray-600">— 10,000 simulaciones</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-400" />
                  <Tooltip term="XGBoost"><span className="font-medium text-gray-300">XGBoost</span></Tooltip>
                  <span className="text-gray-600">— 54 features · 3,000 partidos</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-purple-400" />
                  <Tooltip term="Gemini"><span className="font-medium text-gray-300">Gemini 2.5 Flash</span></Tooltip>
                  <span className="text-gray-600">— Análisis + Google Search</span>
                </div>
              </div>
            </div>

            {/* Card 2: How it learns */}
            <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5 hover:border-blue-500/30 transition-colors">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xl">📐</span>
                <h3 className="text-sm font-semibold">¿Cómo aprende?</h3>
              </div>
              <div className="space-y-2 text-xs text-gray-400">
                <div className="flex items-center gap-2">
                  <span className="text-emerald-400 font-mono text-[10px]">1</span>
                  <span>Genera predicción → guarda en localStorage</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-400 font-mono text-[10px]">2</span>
                  <span>Resultado real → compara predicción vs realidad</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-400 font-mono text-[10px]">3</span>
                  <span>Calcula <Tooltip term="log-loss"><span className="text-blue-400">log-loss</span></Tooltip> y <Tooltip term="Brier Score"><span className="text-blue-400">Brier Score</span></Tooltip></span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-400 font-mono text-[10px]">4</span>
                  <span>Asigna <span className="text-amber-400">puntuación</span> según precisión</span>
                </div>
              </div>
            </div>

            {/* Card 3: Calibration */}
            <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5 hover:border-emerald-500/30 transition-colors">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xl">⚙️</span>
                <h3 className="text-sm font-semibold">
                  <Tooltip term="Calibración Isotónica">Calibración</Tooltip>
                </h3>
              </div>
              <div className="mb-3">
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950 text-emerald-400 text-[11px] font-semibold border border-emerald-500/30">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  ACTIVA — {n} partidos
                </span>
              </div>
              <div className="space-y-1.5 text-xs text-gray-400">
                <p>Ajusta probabilidades automáticamente</p>
                <p className="text-gray-500">· Prob &gt;70%: reduce sobre-confianza</p>
                <p className="text-gray-500">· Prob &lt;30%: aumenta sub-confianza</p>
                <p className="text-gray-500">· Usa <span className="text-blue-400">IsotonicRegression</span> (one-vs-rest)</p>
              </div>
            </div>
          </div>

          {/* FULL-WIDTH CHARTS SECTION */}
          <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5 w-full">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold">📊 Rendimiento del modelo</h3>
                <p className="text-[10px] text-gray-500 mt-0.5">
                  <Tooltip term="Accuracy"><span className="text-emerald-400">{(acc * 100).toFixed(0)}% accuracy</span></Tooltip>
                  {" · "}
                  <Tooltip term="log-loss"><span className="text-blue-400">LogLoss {ll.toFixed(3)}</span></Tooltip>
                  {" · "}
                  <Tooltip term="Brier Score"><span className="text-blue-400">Brier {br.toFixed(3)}</span></Tooltip>
                </p>
              </div>
              <div className="flex gap-4 text-xs">
                <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-emerald-400" /> Modelo</div>
                <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-gray-600" /> Aleatorio</div>
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* LEFT: Radar Chart */}
              <div>
                <h4 className="text-xs font-semibold text-gray-400 mb-3">🎯 Precisión por dimensión</h4>
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#252D3D" />
                    <PolarAngleAxis dataKey="dim" tick={{ fill: "#8B95A8", fontSize: 11, fontWeight: 500 }} />
                    <Radar name="Modelo" dataKey="modelo" stroke="#00C896" fill="#00C896" fillOpacity={0.12} strokeWidth={2} />
                    <Radar name="Aleatorio" dataKey="aleatorio" stroke="#4A5568" fill="none" strokeDasharray="4 4" strokeWidth={1.5} />
                    <Legend wrapperStyle={{ fontSize: 11, color: "#8B95A8", paddingTop: 16 }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* RIGHT: Bar Chart */}
              <div>
                <h4 className="text-xs font-semibold text-gray-400 mb-3">📊 Predicciones por tipo</h4>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={barData} barCategoryGap="30%">
                    <CartesianGrid stroke="#252D3D" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="tipo" tick={{ fill: "#8B95A8", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#4A5568", fontSize: 10 }} allowDecimals={false} axisLine={false} tickLine={false} />
                    <ReTooltip contentStyle={{ background: "#0D0F14", border: "1px solid #252D3D", borderRadius: 10, fontSize: 12, padding: "8px 12px" }} cursor={{ fill: "#1C2333" }} />
                    <Bar dataKey="pred" fill="#252D3D" name="Predichos" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="ok" fill="#00C896" name="Acertados" radius={[6, 6, 0, 0]} />
                    <Legend wrapperStyle={{ fontSize: 11, color: "#8B95A8", paddingTop: 16 }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* TRAINING HISTORY TABLE */}
          <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold">🧪 Historial de Entrenamiento</h3>
                <p className="text-[10px] text-gray-500 mt-0.5">Partidos evaluados por el modelo durante el Mundial 2026</p>
              </div>
              <div className="flex gap-3 text-[10px] text-gray-500">
                <span>{correct} ✅ aciertos</span>
                <span>{n - correct} ❌ errores</span>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-[#252D3D] text-gray-500">
                    <th className="text-left py-3 px-3 font-medium">Fecha</th>
                    <th className="text-left py-3 px-3 font-medium">Partido</th>
                    <th className="text-center py-3 px-3 font-medium">Predicción</th>
                    <th className="text-center py-3 px-3 font-medium">Confianza</th>
                    <th className="text-center py-3 px-3 font-medium">Real</th>
                    <th className="text-center py-3 px-3 font-medium">Estado</th>
                    <th className="text-right py-3 px-3 font-medium">
                      <Tooltip term="log-loss">LogLoss</Tooltip>
                    </th>
                    <th className="text-right py-3 px-3 font-medium">
                      <Tooltip term="Brier Score">Brier</Tooltip>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {MOCK_EVALUATED.map((row, i) => (
                    <tr key={i} className="border-b border-[#1a2030] hover:bg-[#1C2333]/40 transition-colors">
                      <td className="py-3 px-3 text-gray-400">{row.date}</td>
                      <td className="py-3 px-3 font-medium">{row.match}</td>
                      <td className="py-3 px-3 text-center">
                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ${row.predicted === "1" ? "bg-emerald-950 text-emerald-400" : row.predicted === "X" ? "bg-amber-950 text-amber-400" : "bg-red-950 text-red-400"}`}>
                          {row.predicted === "1" ? "Local" : row.predicted === "X" ? "Empate" : "Visit."} {row.prob}%
                        </span>
                      </td>
                      <td className="py-3 px-3 text-center font-mono text-gray-400">{row.confidence}%</td>
                      <td className="py-3 px-3 text-center font-mono font-bold">{row.realHome} — {row.realAway}</td>
                      <td className="py-3 px-3 text-center text-lg">{row.correct ? "✅" : "❌"}</td>
                      <td className="py-3 px-3 text-right font-mono text-gray-400">{row.logLoss.toFixed(3)}</td>
                      <td className="py-3 px-3 text-right font-mono text-gray-400">{row.brier.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
