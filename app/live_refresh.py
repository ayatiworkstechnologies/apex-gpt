"""Live data refresh and model tuning orchestration."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from threading import Lock

from app.city_rates import reload_city_db
from app.predictor import get_model_catalog, reload_runtime_artifacts
from data.update_prices import update_prices
from model.auto_tune import run_tuning

_STATUS_LOCK = Lock()
_STATUS = {
    "state": "idle",
    "started_at": None,
    "finished_at": None,
    "only_verified": True,
    "dry_run": False,
    "summary": None,
    "error": None,
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _set_status(**changes):
    with _STATUS_LOCK:
        _STATUS.update(changes)


def get_refresh_status() -> dict:
    with _STATUS_LOCK:
        return deepcopy(_STATUS)


def run_live_refresh(
    *,
    only_verified: bool = True,
    dry_run: bool = False,
    version: str | None = None,
) -> dict:
    """Fetch latest rate data, retune the model, and reload runtime artifacts."""
    _set_status(
        state="running",
        started_at=_now_iso(),
        finished_at=None,
        only_verified=only_verified,
        dry_run=dry_run,
        summary=None,
        error=None,
    )

    try:
        price_summary = update_prices(dry_run=dry_run, only_verified=only_verified)
        tune_summary = None
        runtime_reload = None
        city_reload = None

        if not dry_run:
            city_reload = {"cities_loaded": len(reload_city_db())}
            tune_summary = run_tuning(save_best=True, version=version)
            runtime_reload = reload_runtime_artifacts()

        summary = {
            "price_update": price_summary,
            "model_tuning": tune_summary,
            "city_rate_reload": city_reload,
            "runtime_reload": runtime_reload,
            "model_catalog": get_model_catalog(),
        }
        _set_status(
            state="completed",
            finished_at=_now_iso(),
            summary=summary,
            error=None,
        )
        return get_refresh_status()
    except Exception as exc:
        _set_status(
            state="failed",
            finished_at=_now_iso(),
            error=str(exc),
        )
        return get_refresh_status()
