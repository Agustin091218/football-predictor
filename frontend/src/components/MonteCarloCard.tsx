import type { PredictionResponse } from "../types";

export function MonteCarloCard({ pred }: { pred: PredictionResponse }) {
  const s = pred.simulation;
  if (!s) return null;

  const topScores = (s.top_scores || []).slice(0, 5);
  const maxProb = topScores[0]?.probability || 1;

  return (
    <div className="bg-[#161B26] border border-[#252D3D] rounded-xl p-5 space-y-4">
      <h3 className="text-sm font-semibold text-gray-200">
        📊 Monte Carlo ({(s.n_simulations ?? 0).toLocaleString()} simulaciones)
      </h3>

      <div className="grid grid-cols-4 gap-3">
        {[
          { label: "Over 2.5", value: s.prob_over_2_5, color: "text-emerald-400" },
          { label: "BTTS", value: s.prob_btts, color: "text-amber-400" },
          { label: `Clean ${pred.match.home_team.name}`, value: s.prob_clean_sheet_home },
          { label: `Clean ${pred.match.away_team.name}`, value: s.prob_clean_sheet_away },
        ].map((m) => (
          <div key={m.label} className="bg-[#1C2333] rounded-lg p-3 text-center">
            <div className={`text-lg font-mono font-bold ${m.color || "text-gray-200"}`}>
              {((m.value ?? 0) * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] text-gray-500 mt-1">{m.label}</div>
          </div>
        ))}
      </div>

      {topScores.length > 0 && (
        <div>
          <div className="text-xs text-gray-500 mb-2">Marcadores más probables</div>
          <div className="space-y-1">
            {topScores.map((sc, i) => {
              const colors = ["#00C896", "#27ae60", "#1abc9c", "#16a085", "#0e6655"];
              return (
                <div key={sc.score} className="flex items-center gap-3 text-sm">
                  <span className="font-mono w-8">{sc.score}</span>
                  <div className="flex-1 h-3 bg-[#1C2333] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${((sc.probability / maxProb) * 100).toFixed(0)}%`,
                        backgroundColor: colors[i],
                      }}
                    />
                  </div>
                  <span className="font-mono text-xs text-gray-400 w-12 text-right">
                    {(sc.probability * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
