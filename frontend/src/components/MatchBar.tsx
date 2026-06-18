import type { PredictionResponse } from "../types";

export function MatchBar({ pred }: { pred: PredictionResponse }) {
  const { prob_home_win: ph, prob_draw: pd, prob_away_win: pa } = pred;

  return (
    <div className="space-y-2">
      <div className="flex h-10 rounded-lg overflow-hidden text-sm font-bold">
        <div
          className="bg-emerald-500 flex items-center justify-center text-white transition-all duration-700"
          style={{ width: `${(ph * 100).toFixed(0)}%` }}
        >
          {(ph * 100).toFixed(0)}%
        </div>
        <div
          className="bg-amber-500 flex items-center justify-center text-white transition-all duration-700"
          style={{ width: `${(pd * 100).toFixed(0)}%` }}
        >
          {(pd * 100).toFixed(0)}%
        </div>
        <div
          className="bg-red-500 flex items-center justify-center text-white transition-all duration-700"
          style={{ width: `${(pa * 100).toFixed(0)}%` }}
        >
          {(pa * 100).toFixed(0)}%
        </div>
      </div>
      <div className="flex justify-between text-xs text-gray-500">
        <span>🏠 {pred.match.home_team.name}</span>
        <span>🤝 Empate</span>
        <span>{pred.match.away_team.name} 🏠</span>
      </div>
      <div className="grid grid-cols-3 gap-4 text-center mt-3">
        <div>
          <div className="text-2xl font-mono font-bold text-emerald-400">
            {(ph * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">{pred.match.home_team.name}</div>
        </div>
        <div>
          <div className="text-2xl font-mono font-bold text-amber-400">
            {(pd * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">Empate</div>
        </div>
        <div>
          <div className="text-2xl font-mono font-bold text-red-400">
            {(pa * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">{pred.match.away_team.name}</div>
        </div>
      </div>
    </div>
  );
}
