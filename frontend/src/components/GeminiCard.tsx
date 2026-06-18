import type { PredictionResponse } from "../types";

export function GeminiCard({ pred }: { pred: PredictionResponse }) {
  const so = pred.signal_outputs || {};
  const la = (so._llm_analysis as unknown as Record<string, unknown>) || {};
  const gm = (so._gemini_matches as unknown as Record<string, { opponent: string; score: string }[]>) || {};

  if (!pred.llm_explanation) {
    return (
      <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5 border-l-[3px] border-l-gray-600">
        <h3 className="text-sm font-semibold text-gray-400">🤖 Análisis LLM no disponible</h3>
        <p className="text-xs text-gray-500 mt-1">Configurá GEMINI_API_KEY para activar explicaciones</p>
        {renderRecentMatches(gm, pred)}
      </div>
    );
  }

  return (
    <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5 border-l-[3px] border-l-blue-500">
      <h3 className="text-sm font-semibold text-gray-200">🤖 Análisis Gemini</h3>
      <p className="text-sm text-gray-300 mt-2 leading-relaxed">{pred.llm_explanation}</p>

      {renderRecentMatches(gm, pred)}

      {renderList("🔄 Escenarios alternativos", la.alternative_scenarios as string[], "text-amber-400", "border-amber-500")}
      {renderField("⏱️ Goles", la.goal_timing as string)}
      {renderField("📈 Desarrollo", la.match_flow as string)}
      {renderField("⚔️ Duelos clave", la.key_matchups as string)}
      {renderField("🟨 Disciplina", la.discipline as string)}

      {(pred.llm_actions || []).length > 0 && (
        <div className="mt-3">
          <div className="text-xs text-gray-500 mb-1">⚡ Acciones sugeridas</div>
          <div className="flex flex-wrap gap-1">
            {pred.llm_actions!.map((a) => (
              <span key={a} className="text-[10px] px-2 py-0.5 rounded-full border border-blue-500 text-blue-400">
                {a.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function renderRecentMatches(
  gm: Record<string, { opponent: string; score: string }[]>,
  pred: PredictionResponse
) {
  if (!gm.home?.length && !gm.away?.length) return null;
  return (
    <div className="mt-3">
      <div className="text-xs text-gray-500 mb-1">📋 Últimos partidos (Google Search)</div>
      {gm.home?.length ? (
        <div className="text-xs text-gray-400">
          <b>{pred.match.home_team.name}:</b>{" "}
          {gm.home.map((m) => `${m.opponent} ${m.score}`).join(", ")}
        </div>
      ) : null}
      {gm.away?.length ? (
        <div className="text-xs text-gray-400">
          <b>{pred.match.away_team.name}:</b>{" "}
          {gm.away.map((m) => `${m.opponent} ${m.score}`).join(", ")}
        </div>
      ) : null}
    </div>
  );
}

function renderList(title: string, items: string[] | undefined, color: string, border: string) {
  if (!items?.length) return null;
  return (
    <div className="mt-3">
      <div className="text-xs text-gray-500 mb-1">{title}</div>
      {items.map((s, i) => (
        <div key={i} className={`text-xs ${color} pl-2 border-l-2 ${border} my-1`}>{s}</div>
      ))}
    </div>
  );
}

function renderField(label: string, value: string | undefined) {
  if (!value) return null;
  return (
    <div className="mt-2 text-xs">
      <span className="text-gray-500">{label}: </span>
      <span className="text-gray-300">{value}</span>
    </div>
  );
}
