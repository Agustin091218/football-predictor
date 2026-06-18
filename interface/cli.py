"""CLI entry point for the football predictor.

Uses the Container singleton for all dependencies.
"""

from __future__ import annotations

import typer

from football_predictor.application.predict_match import (
    InsufficientDataError,
    MatchNotFoundError,
)
from football_predictor.domain.entities import (
    MatchResult,
)
from football_predictor.infrastructure.container import container

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(name="football", help="Football match predictor — CLI")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def fetch(
    league: str = typer.Option("PL", help="Código de liga: PL, PD, SA, BL1, FL1"),
    season: str = typer.Option("2024", help="Año de inicio de la temporada"),
) -> None:
    """Descarga partidos de la API y los guarda en la base de datos."""
    if not container.football_data_token:
        typer.echo("❌ FOOTBALL_DATA_TOKEN no configurado. Exportá la variable de entorno.")
        raise typer.Exit(1)

    result = container.fetch_and_store_uc.execute(league, season)

    if result["status"] == "ok":
        typer.echo(f"✅ {result['matches_fetched']} partidos guardados para {league} {season}")
    else:
        typer.echo(f"❌ Error: {result.get('error', 'desconocido')}")


@app.command()
def compute_stats(
    league: str = typer.Option("PL", help="Código de liga: PL, PD, SA, BL1, FL1"),
    season: str = typer.Option("2024", help="Año de inicio de la temporada"),
) -> None:
    """Calcula estadísticas de equipos a partir de partidos finalizados."""
    result = container.compute_stats_uc.execute(league, season)

    status = result["status"]
    if status == "ok":
        typer.echo(
            f"✅ Stats calculadas para {result['teams_computed']} equipos "
            f"usando {result['matches_used']} partidos"
        )
    elif status == "insufficient_data":
        typer.echo(f"⚠️  Solo {result['matches_found']} partidos finalizados. Necesitás al menos 5.")
    else:
        typer.echo(f"❌ Error inesperado: {result}")


@app.command()
def predict(
    match_id: str = typer.Argument(help="ID del partido a predecir"),
) -> None:
    """Genera una predicción para un partido."""

    try:
        prediction = container.predict_uc.execute(match_id)
    except MatchNotFoundError as exc:
        typer.echo(f"❌ {exc}")
        raise typer.Exit(1) from exc
    except InsufficientDataError as exc:
        typer.echo(f"⚠️  {exc}")
        raise typer.Exit(1) from exc

    home_name = prediction.match.home_team.name
    away_name = prediction.match.away_team.name
    league_name = prediction.match.league.name or str(prediction.match.league.id)
    match_date_str = prediction.match.match_date.strftime("%d/%m/%Y %H:%M")

    result_label = {
        MatchResult.HOME_WIN: "Victoria local",
        MatchResult.DRAW: "Empate",
        MatchResult.AWAY_WIN: "Victoria visitante",
    }

    typer.echo(f"\n{'=' * 50}")
    typer.echo(f"  {home_name} vs {away_name}")
    typer.echo(f"  {league_name} — {match_date_str}")
    typer.echo(f"{'=' * 50}")
    typer.echo(f"  Victoria local:   {prediction.prob_home_win:.1%}")
    typer.echo(f"  Empate:           {prediction.prob_draw:.1%}")
    typer.echo(f"  Victoria visit.:  {prediction.prob_away_win:.1%}")
    typer.echo(
        f"  Goles esperados:  {prediction.expected_goals_home:.2f}"
        f" — {prediction.expected_goals_away:.2f}"
    )
    typer.echo(
        f"  Resultado más probable: "
        f"{result_label.get(prediction.predicted_result, prediction.predicted_result.value)}"
    )
    typer.echo(f"  Confianza del modelo:   {prediction.confidence:.1%}")
    typer.echo(f"  Modelo usado:           {prediction.model_version}")
    typer.echo(f"{'=' * 50}")

    if prediction.simulation:
        mc = prediction.simulation
        typer.echo(f"\n  SIMULACIÓN MONTE CARLO ({mc.n_simulations:,} simulaciones):")
        typer.echo(
            f"  Victoria local:     {mc.prob_home_win:.1%} "
            f"(IC 95%: {mc.ci_home_win_low:.1%}–{mc.ci_home_win_high:.1%})"
        )
        typer.echo(f"  Empate:             {mc.prob_draw:.1%}")
        typer.echo(f"  Victoria visitante: {mc.prob_away_win:.1%}")
        if mc.top_scores:
            top3 = ", ".join(f"{s['score']} ({s['probability']:.1%})" for s in mc.top_scores[:3])
            typer.echo(f"  Top 3 marcadores:   {top3}")
        typer.echo(f"  Over 2.5 goles:     {mc.prob_over_2_5:.1%}")
        typer.echo(f"  Ambos marcan:       {mc.prob_btts:.1%}")

    if prediction.signal_outputs and "poisson" in prediction.signal_outputs:
        p_val = prediction.signal_outputs["poisson"]["value"]
        typer.echo("\n  📐 Desglose por modelo:")
        typer.echo(
            f"  Poisson:  "
            f"1={p_val['prob_home_win']:.1%} "
            f"X={p_val['prob_draw']:.1%} "
            f"2={p_val['prob_away_win']:.1%}"
        )
        typer.echo(
            f"  Ensemble: "
            f"1={prediction.prob_home_win:.1%} "
            f"X={prediction.prob_draw:.1%} "
            f"2={prediction.prob_away_win:.1%}"
        )

    if prediction.llm_explanation:
        typer.echo(f"\n  💬 ANÁLISIS: {prediction.llm_explanation}")

    typer.echo("")


@app.command()
def list_upcoming(
    league: str = typer.Option("PL", help="Código de liga: PL, PD, SA, BL1, FL1"),
    days: int = typer.Option(7, help="Días hacia adelante"),
) -> None:
    """Lista los próximos partidos con sus IDs."""
    matches: list = container.match_repo.get_upcoming_matches(league, days_ahead=days)

    if not matches:
        typer.echo(f"No hay partidos programados para {league} en los próximos {days} días.")
        return

    typer.echo(f"\nPróximos partidos — {league}:\n")
    for m in matches:
        date_str = m.match_date.strftime("%d/%m %H:%M")
        typer.echo(f"  [{m.id}]  {date_str}  {m.home_team.name} vs {m.away_team.name}")
    typer.echo("")


@app.command()
def evaluate(
    league: str = typer.Option("PL", help="Código de liga: PL, PD, SA, BL1, FL1"),
    season: str = typer.Option("2024", help="Año de inicio de la temporada"),
) -> None:
    """Evalúa predicciones pasadas comparando con resultados reales."""
    finished = container.match_repo.get_finished_matches(league, season)

    evaluated = 0
    for m in finished:
        if m.score is None or m.score.result is None:
            continue
        pred = container.prediction_repo.get_by_match_id(str(m.id))
        if pred is not None:
            container.prediction_repo.update_result(str(m.id), m.score.result)
            evaluated += 1

    if evaluated == 0:
        typer.echo("No hay predicciones para evaluar en esta liga/temporada.")
        return

    stats = container.prediction_repo.get_accuracy_stats(league_id=league)

    typer.echo(f"\n{'=' * 50}")
    typer.echo("  ✅ Evaluación completada")
    typer.echo(f"  Predicciones evaluadas: {evaluated}")
    typer.echo(f"{'=' * 50}")

    if stats["total_evaluated"] > 0:
        typer.echo(
            f"  Accuracy general: {stats['accuracy']:.1%} "
            f"({stats['correct']}/{stats['total_evaluated']} predicciones)"
        )
    typer.echo("  Por resultado:")
    for key, label in [
        ("1", "Victoria local (1)"),
        ("X", "Empate (X)"),
        ("2", "Victoria visit (2)"),
    ]:
        by = stats["by_result"].get(key, {})
        count = by.get("predicted", 0)
        acc = by.get("accuracy", 0.0)
        corr = by.get("correct", 0)
        if count > 0:
            typer.echo(f"    {label}: {acc:.1%} ({corr}/{count})")
        else:
            typer.echo(f"    {label}: —")
    typer.echo(f"{'=' * 50}\n")


@app.command()
def set_weights(
    poisson: float = typer.Option(0.5, help="Peso del modelo Poisson (0.0-1.0)"),
    xgboost: float = typer.Option(0.5, help="Peso del modelo XGBoost (0.0-1.0)"),
) -> None:
    """Ajusta los pesos del ensemble en runtime sin reiniciar."""
    try:
        container.ensemble.set_weights(poisson, xgboost)
        typer.echo(f"✅ Pesos actualizados: Poisson={poisson:.0%} | XGBoost={xgboost:.0%}")
    except ValueError as exc:
        typer.echo(f"❌ {exc}")
        raise typer.Exit(1)


@app.command()
def backtest(
    league: str = typer.Option("PL"),
    season: str = typer.Option("2024"),
    train_until: int = typer.Option(20, help="Entrenar hasta jornada N, testear el resto"),
) -> None:
    """Evalúa el modelo históricamente sin ver datos futuros."""
    from football_predictor.application.backtesting import BacktestingUseCase
    from football_predictor.infrastructure.ml.calibration import ProbabilityCalibrator

    calibrator = ProbabilityCalibrator()
    bt_uc = BacktestingUseCase(
        match_repo=container.match_repo,
        stats_repo=container.stats_repo,
        poisson_service=container.poisson_service,
        feature_engineer=container.feature_engineer,
        xgboost_predictor=container.xgboost_predictor,
        monte_carlo=container.mc_simulator,
        calibrator=calibrator,
    )

    typer.echo(
        f"🔍 Backtesting {league} {season} "
        f"(train: jornadas 1-{train_until}, test: jornadas {train_until + 1}+)..."
    )

    try:
        result = bt_uc.execute(league, season, train_until)

        typer.echo(f"\n{'=' * 55}")
        typer.echo(f"  Backtesting — {league} {season}")
        typer.echo(f"{'=' * 55}")
        typer.echo(f"  Partidos train:    {result.n_matches_train}")
        typer.echo(f"  Partidos test:     {result.n_matches_test}")
        typer.echo(f"  Accuracy:          {result.accuracy:.1%}")
        typer.echo(f"  Log-loss:          {result.log_loss:.4f}")
        typer.echo(f"  Brier score:       {result.brier_score:.4f}")
        typer.echo(f"  ROI flat betting:  {result.roi_flat_betting:+.1f}%")
        typer.echo("\n  Breakdown por resultado:")
        for res, data in result.breakdown.items():
            label = {"1": "Victoria local", "X": "Empate", "2": "Victoria visit."}[res]
            typer.echo(
                f"    {label}: {data['accuracy']:.1%} ({data['correct']}/{data['predicted']})"
            )
        typer.echo(
            f"\n  Threshold óptimo: {result.best_threshold:.0%} "
            f"→ {result.best_threshold_accuracy:.1%} accuracy "
            f"({result.best_threshold_n} predicciones)"
        )
        typer.echo(f"  Modelo usado: {result.model_used}")
        typer.echo(f"{'=' * 55}\n")

    except ValueError as exc:
        typer.echo(f"⚠️  {exc}")
        raise typer.Exit(1)


@app.command()
def scheduler_start() -> None:
    """Inicia el scheduler en foreground (bloqueante)."""
    import time

    from football_predictor.infrastructure.scheduler import PredictionScheduler

    sched = PredictionScheduler(container)
    sched.start()
    typer.echo("✅ Scheduler iniciado. Ctrl+C para detener.")
    typer.echo("Jobs activos:")
    for job in sched.get_jobs_status():
        typer.echo(f"  - {job['name']} | próxima: {job['next_run']}")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        sched.stop()
        typer.echo("\n🛑 Scheduler detenido.")


@app.command()
def wc_calibrate() -> None:
    """Evalúa predicciones contra resultados reales del Mundial 2026."""
    from football_predictor.application.world_cup_calibration import WorldCupCalibrationUseCase
    from football_predictor.infrastructure.ml.calibration import ProbabilityCalibrator

    calibrator = ProbabilityCalibrator()
    cal_uc = WorldCupCalibrationUseCase(
        prediction_repo=container.prediction_repo,
        match_repo=container.match_repo,
        calibrator=calibrator,
    )
    result = cal_uc.execute()

    typer.echo(f"\n{'=' * 55}")
    typer.echo("  🌍 Calibración — Mundial 2026")
    typer.echo(f"{'=' * 55}")
    cov = result["mundial_2026"]
    typer.echo(
        f"  Partidos con predicción: {cov['partidos_con_prediccion']}/{cov['partidos_jugados']}"
    )

    if cov["partidos_con_prediccion"] > 0:
        m = result["metricas"]
        typer.echo(f"  Accuracy:    {m['accuracy']:.1%} ({m['n_correctos']}/{m['n_evaluados']})")
        typer.echo(f"  Log-loss:    {m['log_loss']:.4f}")
        typer.echo(f"  Brier score: {m['brier_score']:.4f}")

        if result["upsets"]:
            typer.echo("\n  ⚡ Resultados sorpresivos:")
            for u in result["upsets"][:3]:
                typer.echo(
                    f"    {u['partido']}: {u['resultado']} "
                    f"(modelo daba {u['prob_resultado_real']:.0%})"
                )

        cal = result["calibration"]
        typer.echo(
            f"\n  Calibración: {'✅ actualizada' if cal['fitted'] else '⚠️ ' + (cal['note'] or '')}"
        )
    else:
        typer.echo("  Sin predicciones para evaluar. Predecí partidos primero.")
    typer.echo(f"{'=' * 55}\n")


@app.command()
def wc_add_result(
    home: str = typer.Argument(help="Equipo local"),
    away: str = typer.Argument(help="Equipo visitante"),
    score: str = typer.Argument(help="Marcador: '3-0'"),
    group: str = typer.Option("?", help="Letra del grupo: A-L"),
) -> None:
    """Registra un resultado real del Mundial y recalibra."""
    from datetime import date

    from football_predictor.application.world_cup_calibration import (
        WorldCupCalibrationUseCase,
        WorldCupTracker,
    )
    from football_predictor.infrastructure.ml.calibration import ProbabilityCalibrator

    try:
        hg, ag = map(int, score.split("-"))
    except ValueError:
        typer.echo("❌ Formato de score inválido. Usar '3-0'.")
        raise typer.Exit(1)

    calibrator = ProbabilityCalibrator()
    cal_uc = WorldCupCalibrationUseCase(
        prediction_repo=container.prediction_repo,
        match_repo=container.match_repo,
        calibrator=calibrator,
    )
    tracker = WorldCupTracker(cal_uc, container.prediction_repo)

    r = tracker.add_result(
        home_team=home,
        away_team=away,
        home_goals=hg,
        away_goals=ag,
        group=group,
        match_date=str(date.today()),
    )
    typer.echo(f"✅ {r['result_added']}")
    typer.echo(f"   Partidos: {r['total_results']}")
    if r.get("current_accuracy"):
        typer.echo(f"   Accuracy: {r['current_accuracy']:.1%}")


@app.command()
def wc_standings() -> None:
    """Tabla de posiciones del Mundial 2026."""
    from football_predictor.application.world_cup_calibration import (
        WorldCupCalibrationUseCase,
        WorldCupTracker,
    )
    from football_predictor.infrastructure.ml.calibration import ProbabilityCalibrator

    calibrator = ProbabilityCalibrator()
    cal_uc = WorldCupCalibrationUseCase(
        prediction_repo=container.prediction_repo,
        match_repo=container.match_repo,
        calibrator=calibrator,
    )
    tracker = WorldCupTracker(cal_uc, container.prediction_repo)
    standings = tracker.get_standings()

    for group, teams in sorted(standings.items()):
        typer.echo(f"\n  Grupo {group}")
        typer.echo(f"  {'Equipo':<20} PJ  PG  PE  PP  GF  GC  Pts")
        typer.echo(f"  {'-' * 52}")
        for t in teams:
            typer.echo(
                f"  {t['team']:<20} {t['pj']:>2}  {t['pg']:>2}  {t['pe']:>2}  "
                f"{t['pp']:>2}  {t['gf']:>2}  {t['gc']:>2}  {t['pts']:>3}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
