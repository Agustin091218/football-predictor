export interface TeamInfo {
  id: string;
  name: string;
  short_name: string;
  tla: string;
  country: string;
}

export interface MatchInfo {
  id: string;
  home_team: TeamInfo;
  away_team: TeamInfo;
  league: { id: string; name: string; country: string; season?: string };
  match_date: string;
  status: string;
  matchday?: number;
}

export interface MonteCarloSim {
  n_simulations: number;
  prob_home_win: number;
  prob_draw: number;
  prob_away_win: number;
  ci_home_win: [number, number];
  most_likely_score: string;
  top_scores: { score: string; probability: number; count?: number }[];
  prob_over_2_5: number;
  prob_btts: number;
  prob_clean_sheet_home: number;
  prob_clean_sheet_away: number;
  goals_p50_home: number;
  goals_p50_away: number;
}

export interface PredictionResponse {
  match_id: string;
  match: MatchInfo;
  prob_home_win: number;
  prob_draw: number;
  prob_away_win: number;
  predicted_result: string;
  expected_goals_home: number;
  expected_goals_away: number;
  confidence: number;
  model_version: string;
  predicted_at: string;
  simulation?: MonteCarloSim;
  signal_outputs?: Record<string, SignalOutput>;
  llm_explanation?: string;
  llm_actions?: string[];
}

export interface SignalOutput {
  signal: string;
  weight: number;
  confidence: number;
  summary: string;
  value: Record<string, unknown>;
}

export interface CachedPrediction {
  home: string;
  away: string;
  matchId: string;
  prediction: PredictionResponse;
  cachedAt: string;
  apiMatchId?: string;
}

export interface RealResult {
  homeGoals: number;
  awayGoals: number;
  enteredAt: string;
}

export interface ScoredResult {
  pts: number;
  breakdown: { l: string; pts: number; t: string }[];
  correct: boolean;
  realRes: string;
  maxPossible: number;
  pct: number;
  home: string;
  away: string;
  realHome: number;
  realAway: number;
  matchId: string;
}
