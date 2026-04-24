import dataclasses
import json
import os
import shutil
from datetime import datetime
from statistics import fmean, stdev

import equinox as eqx

from taming_the_ito_lyon.models import Model
from taming_the_ito_lyon.training.results_gathering_fns import ResultsDict


LOSS_METRICS: dict[str, dict[str, float | str]] = {
    "mse": {"scale": 1.0, "unit": ""},
    "rge": {"scale": 1.0, "unit": ""},
    "sigker": {"scale": 1.0, "unit": ""},
    "sigker_branched": {"scale": 1.0, "unit": ""},
    "frobenius": {"scale": 1.0, "unit": ""},
}


def loss_meta(loss_label: str, value: float) -> tuple[float, str, float]:
    info = LOSS_METRICS.get(loss_label, {"scale": 1.0, "unit": ""})
    scale = float(info.get("scale", 1.0))
    unit = str(info.get("unit", ""))
    return value * scale, unit, scale


def format_loss(loss_label: str, value: float) -> str:
    scaled_value, unit, _ = loss_meta(loss_label, value)
    suffix = f" {unit}" if unit else ""
    return f"{scaled_value:.3f}{suffix}"


def get_run_dirname(model_name: str) -> str:
    """Generate a human-readable directory name like 'nrde_10_25pm_26_11_25'."""
    now = datetime.now()
    time_str = now.strftime("%I_%M%p").lower()
    date_str = now.strftime("%d_%m_%y")
    return f"{model_name}_{time_str}_{date_str}"


def _best_value_and_epoch(values: list[float]) -> tuple[float | None, int | None]:
    if len(values) == 0:
        return None, None
    best_epoch, best_value = min(enumerate(values), key=lambda item: item[1])
    return float(best_value), int(best_epoch)


def _scale_loss_history(loss_label: str, values: list[float]) -> tuple[list[float], str, float]:
    _, unit, scale = loss_meta(loss_label, 0.0)
    return [float(v) * scale for v in values], unit, scale


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
def _extract_seed_test_metric(seed_payload: dict[str, object]) -> float | None:
    test_section = seed_payload.get("test")
    if not isinstance(test_section, dict):
        return None
    metric_section = test_section.get("metric")
    if not isinstance(metric_section, dict):
        return None
    return _to_float(metric_section.get("value"))


def _extract_seed_results(seed_payload: dict[str, object]) -> list[tuple[float, float]]:
    test_section = seed_payload.get("test")
    if not isinstance(test_section, dict):
        return []
    results_dict = test_section.get("results_dict")
    if not isinstance(results_dict, dict):
        return []
    raw_times = results_dict.get("results_times")
    raw_values = results_dict.get("results")
    if not isinstance(raw_times, list) or not isinstance(raw_values, list):
        return []

    time_value_pairs: list[tuple[float, float]] = []
    for raw_time, raw_value in zip(raw_times, raw_values):
        time_value = _to_float(raw_time)
        result_value = _to_float(raw_value)
        if time_value is None or result_value is None:
            continue
        time_value_pairs.append((float(time_value), float(result_value)))
    return time_value_pairs


def _build_seed_aggregate(
    *, seed_metrics: dict[str, object], eval_metric_name: str
) -> dict[str, object] | None:
    metric_values: list[float] = []
    results_by_time: dict[float, list[float]] = {}
    seed_count = 0

    for seed_payload in seed_metrics.values():
        if not isinstance(seed_payload, dict):
            continue
        seed_count += 1

        metric_value = _extract_seed_test_metric(seed_payload)
        if metric_value is not None:
            metric_values.append(float(metric_value))

        for time_value, result_value in _extract_seed_results(seed_payload):
            bucket = results_by_time.setdefault(float(time_value), [])
            bucket.append(float(result_value))

    if seed_count == 0:
        return None

    aggregate_test: dict[str, object] = {}
    if len(metric_values) > 0:
        aggregate_test["metric"] = {
            "name": eval_metric_name,
            "mean": float(fmean(metric_values)),
            "std_1sigma": float(stdev(metric_values)) if len(metric_values) > 1 else None,
            "count": len(metric_values),
        }

    if len(results_by_time) > 0:
        sorted_times = sorted(results_by_time.keys())
        results_mean: list[float] = []
        results_std_1sigma: list[float | None] = []
        counts: list[int] = []
        for time_value in sorted_times:
            values = results_by_time[time_value]
            results_mean.append(float(fmean(values)))
            results_std_1sigma.append(float(stdev(values)) if len(values) > 1 else None)
            counts.append(len(values))
        aggregate_test["results_dict"] = {
            "results_times": sorted_times,
            "results_mean": results_mean,
            "results_std_1sigma": results_std_1sigma,
            "counts": counts,
        }

    if len(aggregate_test) == 0:
        return None

    return {
        "num_seeds": seed_count,
        "std_ddof": 1,
        "test": aggregate_test,
    }


def build_training_metrics_payload(
    *,
    run_dirname: str,
    model_name: str,
    num_params: int,
    final_epoch: int,
    best_epoch: int,
    training_elapsed: float,
    time_to_best_epoch: float | None,
    inference_elapsed: float,
    loss_label: str,
    eval_metric_name: str,
    train_loss_history: list[float],
    val_loss_history: list[float],
    val_metric_history: list[float],
    test_loss: float,
    test_eval_metric: float,
    test_results_dict: ResultsDict,
    xla_scratch_size_mib: float | None = None,
) -> dict[str, object]:
    scaled_train_history, unit, scale = _scale_loss_history(loss_label, train_loss_history)
    scaled_val_history, _, _ = _scale_loss_history(loss_label, val_loss_history)
    scaled_test_loss, _, _ = loss_meta(loss_label, float(test_loss))
    best_train_loss, best_train_epoch = _best_value_and_epoch(scaled_train_history)
    best_val_loss, best_val_loss_epoch = _best_value_and_epoch(scaled_val_history)
    best_val_metric, best_val_metric_epoch = _best_value_and_epoch(val_metric_history)
    test_results = dataclasses.asdict(test_results_dict)
    metrics: dict[str, object] = {
        "run": {
            "name": run_dirname,
            "total_epochs": final_epoch + 1,
            "best_epoch": best_epoch,
        },
        "model": {
            "type": model_name,
            "num_params": num_params,
        },
        "timings": {
            "training_s": training_elapsed,
            "time_to_best_epoch_s": time_to_best_epoch,
            "inference_s": inference_elapsed,
        },
        "memory": {
            "xla_scratch_size_mib": xla_scratch_size_mib,
        },
        "train": {
            "loss": {
                "name": loss_label,
                "best": best_train_loss,
                "best_epoch": best_train_epoch,
                "last": scaled_train_history[-1] if scaled_train_history else None,
                "history": scaled_train_history,
                "unit": unit,
                "scale": scale,
            }
        },
        "validation": {
            "loss": {
                "name": loss_label,
                "best": best_val_loss,
                "best_epoch": best_val_loss_epoch,
                "last": scaled_val_history[-1] if scaled_val_history else None,
                "history": scaled_val_history,
                "unit": unit,
                "scale": scale,
            },
            "metric": {
                "name": eval_metric_name,
                "used_for_best_epoch": True,
                "best": best_val_metric,
                "best_epoch": best_val_metric_epoch,
                "last": val_metric_history[-1] if val_metric_history else None,
                "history": [float(v) for v in val_metric_history],
            },
        },
        "test": {
            "loss": {
                "name": loss_label,
                "value": scaled_test_loss,
                "unit": unit,
                "scale": scale,
            },
            "metric": {
                "name": eval_metric_name,
                "value": float(test_eval_metric),
            },
            "results_dict": test_results,
        },
    }
    return metrics


def finalize_training_run(
    *,
    run_dirname: str,
    model_name: str,
    model: Model,
    temp_best_path: str,
    config_path: str | None,
    num_params: int,
    final_epoch: int,
    best_epoch: int,
    training_elapsed: float,
    time_to_best_epoch: float | None,
    inference_elapsed: float,
    loss_label: str,
    eval_metric_name: str,
    train_loss_history: list[float],
    val_loss_history: list[float],
    val_metric_history: list[float],
    test_loss: float,
    test_eval_metric: float,
    test_results_dict: ResultsDict,
    xla_scratch_size_mib: float | None = None,
) -> str:
    run_dir = os.path.join("saved_models", run_dirname)
    os.makedirs(run_dir, exist_ok=True)

    best_path = os.path.join(run_dir, "best.eqx")
    last_path = os.path.join(run_dir, "last.eqx")
    metrics_path = os.path.join(run_dir, "metrics.json")
    config_save_path = os.path.join(run_dir, "config.toml")

    # Move temp best to final location
    os.rename(temp_best_path, best_path)

    # Save last model
    eqx.tree_serialise_leaves(last_path, model)

    # Copy config file if provided
    if config_path is not None and os.path.exists(config_path):
        shutil.copy2(config_path, config_save_path)

    metrics = build_training_metrics_payload(
        run_dirname=run_dirname,
        model_name=model_name,
        num_params=num_params,
        final_epoch=final_epoch,
        best_epoch=best_epoch,
        training_elapsed=training_elapsed,
        time_to_best_epoch=time_to_best_epoch,
        inference_elapsed=inference_elapsed,
        loss_label=loss_label,
        eval_metric_name=eval_metric_name,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_metric_history=val_metric_history,
        test_loss=test_loss,
        test_eval_metric=test_eval_metric,
        test_results_dict=test_results_dict,
        xla_scratch_size_mib=xla_scratch_size_mib,
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return run_dir


def write_test_metrics(
    *,
    run_dir: str,
    model_name: str,
    num_params: int,
    inference_elapsed: float,
    loss_label: str,
    eval_metric_name: str,
    test_eval_metric: float,
    test_results_dict: ResultsDict,
    checkpoint_path: str,
    metrics_name: str = "test_metrics.json",
    metrics_seed: int | None = None,
    xla_scratch_size_mib: float | None = None,
) -> str:
    metrics_path = os.path.join(run_dir, metrics_name)
    scaled_test, unit, scale = loss_meta(loss_label, test_eval_metric)
    metrics = {
        "run": {
            "name": os.path.basename(run_dir),
            "checkpoint": checkpoint_path,
        },
        "model": {
            "type": model_name,
            "num_params": num_params,
        },
        "timings": {
            "inference_s": inference_elapsed,
        },
        "memory": {
            "xla_scratch_size_mib": xla_scratch_size_mib,
        },
        "test": {
            "loss": {
                "name": loss_label,
                "value": scaled_test,
                "unit": unit,
                "scale": scale,
            },
            "metric": {
                "name": eval_metric_name,
                "value": float(test_eval_metric),
            },
            "results_dict": dataclasses.asdict(test_results_dict),
        },
    }

    if metrics_seed is None:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics_path

    combined_metrics: dict[str, object] = {
        "run": metrics["run"],
        "model": metrics["model"],
        "seed_metrics": {},
    }
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                combined_metrics = loaded
        except json.JSONDecodeError:
            pass

    seed_metrics_obj = combined_metrics.get("seed_metrics", {})
    seed_metrics: dict[str, object] = (
        seed_metrics_obj if isinstance(seed_metrics_obj, dict) else {}
    )
    combined_metrics["run"] = metrics["run"]
    combined_metrics["model"] = metrics["model"]
    combined_metrics["memory"] = metrics["memory"]
    combined_metrics["seed_metrics"] = seed_metrics
    seed_metrics[str(int(metrics_seed))] = metrics
    combined_metrics["seeds"] = sorted(int(seed) for seed in seed_metrics.keys())
    aggregate = _build_seed_aggregate(
        seed_metrics=seed_metrics,
        eval_metric_name=eval_metric_name,
    )
    if aggregate is None:
        combined_metrics.pop("aggregate", None)
    else:
        combined_metrics["aggregate"] = aggregate

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)
    return metrics_path
