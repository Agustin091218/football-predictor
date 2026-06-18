import type { PredictionResponse } from "../types";

export function Callout({ pred }: { pred: PredictionResponse }) {
  const { prob_home_win: ph, prob_draw: pd, prob_away_win: pa, confidence: conf } = pred;
  const sim = pred.simulation;
  const maxProb = Math.max(ph, pd, pa);
  const predRes = maxProb === ph ? "1" : maxProb === pd ? "X" : "2";
  const pick = predRes === "1" ? pred.match.home_team.name : predRes === "X" ? "Empate" : pred.match.away_team.name;
  const isHome = predRes === "1";
  const isDraw = predRes === "X";

  const bg = isHome
    ? "bg-emerald-950 border-emerald-500"
    : isDraw
    ? "bg-amber-950 border-amber-500"
    : "bg-red-950 border-red-500";

  const confColor = conf > 0.55 ? "bg-emerald-900 text-emerald-400" : conf > 0.25 ? "bg-amber-900 text-amber-400" : "bg-red-900 text-red-400";
  const confLabel = conf > 0.55 ? "Alta" : conf > 0.25 ? "Media" : "Baja";

  const bestScore = sim?.most_likely_score || "—";
  const calloutText = conf < 0.2
    ? `Partido muy parejo (confianza ${(conf * 100).toFixed(0)}%)`
    : `🏆 Victoria de ${pick}`;

  return (
    <div className={`rounded-xl p-4 text-center border ${bg}`}>
      <div className="text-lg font-bold">{calloutText}</div>
      <div className="text-sm text-gray-400 mt-1">
        Marcador más probable: <span className="font-mono text-lg">{bestScore}</span>
      </div>
      <div className="font-mono text-lg mt-1">
        xG: <span className="text-emerald-400">{(pred.expected_goals_home ?? 0).toFixed(2)}</span>
        {" — "}
        <span className="text-red-400">{(pred.expected_goals_away ?? 0).toFixed(2)}</span>
      </div>
      <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold mt-2 ${confColor}`}>
        Confianza {confLabel} ({(conf * 100).toFixed(0)}%)
      </span>
    </div>
  );
}
