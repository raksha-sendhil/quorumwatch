"""Runs the trained QuorumWatch model on one scenario or one uploaded case file, for the demo app.

Everything shown in the app comes from the same code as the experiment: the trained Isolation
Forest, the calibrated signal parameters and activation threshold from outputs/results, and the
same alert and evaluation rules.
"""
import csv
import datetime as dt
import hashlib
import io
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config, SignalParams
from data.generator import inject_outbreak, select_cluster, simulate_rates
from data.reporting import apply_reporting, draw_reporting, make_profile
from experiments.calibration import qw_mask, simulate_full
from experiments.evaluation import evaluate_mask
from experiments.world import _process, build_world

# Synthetic day 0 is Monday 1 January 2024, so day % 7 matches the weekday profile (0 = Monday).
EPOCH = dt.date(2024, 1, 1)
MAX_UPLOAD_DAYS = 400
REAL_DEMO_DIR = Path(__file__).resolve().parent / "demo_data"
# Real Karnataka dengue examples: (title, note, first week, last week). Each file starts with the
# model's 14 warm-up weeks. They were chosen after checking how the model behaves on each year.
REAL_DEMOS = {
    "karnataka_2022_monsoon": ("2022 monsoon season", "Dengue rises from July", "2021-09-20", "2022-12-26"),
    "karnataka_2016_monsoon": ("2016 monsoon season", "Malnad and coastal districts", "2015-09-21", "2016-12-26"),
    "karnataka_2024_early_surge": ("Early 2024 surge", "Cases rise from February", "2023-09-25", "2024-06-24"),
    "karnataka_2020_epidemic_tail": ("Early 2020", "End of the 2019 epidemic", "2019-09-23", "2020-06-29"),
    "karnataka_2018_quiet_year": ("2018, a quiet year", "For comparison", "2017-09-25", "2018-12-31"),
}
UNITS = {
    "synthetic": {"step": "Day", "site": "Hospital", "value": "cases", "typical": "cases on a normal day"},
    "karnataka": {"step": "Week", "site": "District", "value": "dengue cases",
                  "typical": "lab-confirmed dengue cases in a typical week"},
}
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
SEED_OFFSET = 7_000_000      # keeps demo runs apart from the study's validation and test seeds


def to_date(day):
    return EPOCH + dt.timedelta(days=int(day))


def to_day(date):
    return (date - EPOCH).days


def csv_facilities(text):
    """The set of facility IDs named in a case file (used to tell which dataset it belongs to)."""
    rows = csv.DictReader(io.StringIO(text.strip()))
    column = next((c for c in (rows.fieldnames or []) if c and c.strip().lower() == "facility_id"), None)
    return {(row[column] or "").strip() for row in rows} if column else set()


def parse_cases_csv(text, facility_ids, step=1):
    """Parse `date,facility_id,cases` (or `day` instead of `date`). Blank cases mean a missing report.
    Rows are `step` days apart (1 for daily data, 7 for weekly). Returns (absolute days, reported
    values with NaN for missing). Raises ValueError with a readable message."""
    rows = list(csv.DictReader(io.StringIO(text.strip())))
    if not rows:
        raise ValueError("The file is empty.")
    columns = {c.strip().lower(): c for c in rows[0].keys() if c}
    if "facility_id" not in columns or "cases" not in columns or not ({"date", "day"} & columns.keys()):
        raise ValueError("Columns needed: date (or day), facility_id, cases.")
    index = {f: k for k, f in enumerate(facility_ids)}
    entries = []
    for line, row in enumerate(rows, start=2):
        facility = (row[columns["facility_id"]] or "").strip()
        if facility not in index:
            raise ValueError(f"Line {line}: unknown facility '{facility}'.")
        try:
            if "date" in columns:
                day = to_day(dt.date.fromisoformat(row[columns["date"]].strip()))
            else:
                day = int(float(row[columns["day"]]))
        except (TypeError, ValueError):
            raise ValueError(f"Line {line}: could not read the date or day.")
        raw = (row[columns["cases"]] or "").strip()
        try:
            cases = float(raw) if raw else np.nan
        except ValueError:
            raise ValueError(f"Line {line}: cases must be a number or blank.")
        if cases < 0:
            raise ValueError(f"Line {line}: cases cannot be negative.")
        entries.append((day, index[facility], cases))
    first = min(e[0] for e in entries)
    last = max(e[0] for e in entries)
    if any((e[0] - first) % step for e in entries):
        raise ValueError("Dates must be exactly one week apart." if step == 7 else "Dates must be whole days.")
    T = (last - first) // step + 1
    if T > MAX_UPLOAD_DAYS:
        raise ValueError(f"Please upload at most {MAX_UPLOAD_DAYS} days.")
    reported = np.full((T, len(facility_ids)), np.nan)
    for day, k, cases in entries:
        reported[(day - first) // step, k] = cases
    return first + step * np.arange(T), reported


class ModelService:
    def __init__(self, results_dir=ROOT / "outputs" / "results"):
        results_dir = Path(results_dir)
        selected = results_dir / "selected_parameters.json"
        if not selected.exists():
            raise SystemExit("No calibrated model found. Run `python run_experiments.py` first.")
        chosen = json.loads(selected.read_text(encoding="utf8"))
        self.cfg = Config()
        self.params = SignalParams(**chosen["selected_signal_parameters"])
        self.theta_on = float(chosen["activation_threshold"]["theta_on"])
        self.world = self._load_world()
        self.karnataka = self._load_karnataka()
        self.metrics = self._metrics(results_dir)

    def _load_karnataka(self):
        """Karnataka dengue data and district map, if present. Models are built per year on demand."""
        try:
            from real_data import karnataka
            matrix = karnataka.weekly_matrix()
            reference = karnataka.build_karnataka_world(self.cfg, 2022, matrix)
        except Exception as error:
            print(f"Karnataka data unavailable: {error}")
            return None
        return {"module": karnataka, "matrix": matrix, "reference": reference, "worlds": {},
                "map": karnataka.district_map()}

    def _karnataka_world(self, test_year):
        """The model for one test year (trained on preceding years), cached in memory and on disk."""
        k = self.karnataka
        year = int(min(max(test_year, 2014), int(k["matrix"].index.max().year)))
        if year not in k["worlds"]:
            data_file = k["module"].WEEKLY
            cache = CACHE_DIR / f"karnataka_{year}.pkl"
            world = None
            if cache.exists() and cache.stat().st_mtime >= data_file.stat().st_mtime:
                try:
                    with cache.open("rb") as f:
                        world = pickle.load(f)
                except Exception:
                    world = None
            if world is None:
                world = k["module"].build_karnataka_world(self.cfg, year, k["matrix"])
                CACHE_DIR.mkdir(exist_ok=True)
                with cache.open("wb") as f:
                    pickle.dump(world, f)
            k["worlds"][year] = world
        return k["worlds"][year]

    def _load_world(self):
        """Build the world once and cache it; later launches load in seconds."""
        key = hashlib.sha256(json.dumps(self.cfg.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]
        cache = CACHE_DIR / f"world_{key}.pkl"
        if cache.exists():
            try:
                with cache.open("rb") as f:
                    return pickle.load(f)
            except Exception:
                cache.unlink(missing_ok=True)
        world = build_world(self.cfg)
        CACHE_DIR.mkdir(exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump(world, f)
        return world

    # ---- static information -------------------------------------------------------------
    def _metrics(self, results_dir):
        path = results_dir / "false_alert_matched_comparison.csv"
        if not path.exists():
            return None
        target = self.cfg.experiment.target_false_alerts_per_year
        frame = pd.read_csv(path)
        frame = frame[(frame.scenario == "all_scenarios") & np.isclose(frame.false_alerts_per_year, target)]
        out = {"false_alerts_per_year": target, "test_runs": 300}
        for method, key in (("full", "model"), ("forest_alone", "forest_alone")):
            row = frame[frame.method == method].iloc[0]
            out[key] = {name: [float(row[col]), float(row[f"{col}_ci_low"]), float(row[f"{col}_ci_high"])]
                        for name, col in (("detection", "attributable_detection_rate"),
                                          ("delay", "attributable_restricted_delay_days"))}
        return out

    def results(self):
        """The simulated study's results, for the app's Results view (read from the output files)."""
        target = self.cfg.experiment.target_false_alerts_per_year
        out = {"target": target, "geographies": []}
        for key, folder, label in (("primary", ROOT / "outputs" / "results", "Primary geography"),
                                   ("second", ROOT / "outputs_world777" / "results", "Second geography")):
            if not (folder / "false_alert_matched_comparison.csv").exists():
                continue
            matched = pd.read_csv(folder / "false_alert_matched_comparison.csv")
            at = matched[np.isclose(matched.false_alerts_per_year, target)]

            def triple(row, col):
                return [float(row[col]), float(row[f"{col}_ci_low"]), float(row[f"{col}_ci_high"])]

            geo = {"key": key, "label": label, "scenarios": []}
            for scenario in ("all_scenarios", "high_reliability", "medium_reliability", "low_reliability"):
                rows = at[at.scenario == scenario].set_index("method")
                geo["scenarios"].append({"scenario": scenario, **{
                    m: {"detection": triple(rows.loc[m], "attributable_detection_rate"),
                        "delay": triple(rows.loc[m], "attributable_restricted_delay_days")}
                    for m in ("full", "forest_alone")}})
            paired = pd.read_csv(folder / "paired_differences.csv")
            paired = paired[(paired.scenario == "all_scenarios") & (paired.versus == "forest_alone")].set_index("metric")
            geo["paired"] = {m: [float(paired.loc[m, "difference"]), float(paired.loc[m, "ci_low"]),
                                 float(paired.loc[m, "ci_high"])]
                             for m in ("attributable_detection_rate", "attributable_restricted_delay_days")}
            agg = pd.read_csv(folder / "aggregated_metrics.csv")
            agg = agg[(agg.scenario == "all_scenarios") & (agg.metric == "messages_per_day")].set_index("method")
            geo["messages"] = {m: float(agg.loc[m, "mean"]) for m in ("full", "forest_alone")}
            if key == "primary":
                curves = pd.read_csv(folder / "tradeoff_curves.csv")
                curves = curves[(curves.scenario == "all_scenarios") & curves.usable_for_matching]
                out["tradeoff"] = {m: g.sort_values("false_alerts_per_year")[
                    ["false_alerts_per_year", "attributable_detection_rate"]].round(4).values.tolist()
                    for m, g in curves.groupby("method")}
                types = pd.read_csv(folder / "results_by_outbreak_type.csv")
                out["by_type"] = [{"dimension": dim, "value": str(value), "n": int(g.n_runs.iloc[0]),
                                   **{m: float(g[g.method == m].attributable_detection_rate.iloc[0])
                                      for m in ("full", "forest_alone")}}
                                  for (dim, value), g in types.groupby(["dimension", "value"], sort=False)]
            out["geographies"].append(geo)
        out["test_runs"] = 300
        out["params"] = {"D": self.params.D, "delta": self.params.delta, "beta": self.params.beta,
                         "K": self.params.K, "n": self.params.n, "quorum": self.params.quorum,
                         "theta_on": self.theta_on}
        return out

    @staticmethod
    def _dataset_meta(world, name, outlines=None):
        W = world.W
        i, j = np.nonzero(np.triu(W > 0))
        frame = world.facilities
        labels = frame["label"] if "label" in frame else frame["facility_id"]
        meta = {
            "name": name, "unit": UNITS[name],
            "facilities": [{"id": r.facility_id, "x": round(float(r.x_km), 2), "y": round(float(r.y_km), 2),
                            "volume": round(float(r.baseline_daily_count), 1), "label": str(label),
                            "short": str(label) if name != "synthetic" else r.facility_id}
                           for r, label in zip(frame.itertuples(), labels)],
            "edges": [{"s": int(a), "t": int(b), "w": round(float(W[a, b]), 4)} for a, b in zip(i, j)],
        }
        if outlines is not None:
            meta["outlines"] = outlines
        return meta

    def meta(self):
        exp, p = self.cfg.experiment, self.params
        theta_off = self.theta_on * exp.theta_off_ratio
        datasets = {"synthetic": self._dataset_meta(self.world, "synthetic")}
        if self.karnataka is not None:
            k = self.karnataka
            outlines = [k["map"][d]["path"] for d in k["module"].DISTRICTS]
            datasets["karnataka"] = self._dataset_meta(k["reference"], "karnataka", outlines)
        demos = [{"id": key, "title": t, "note": n} for key, (t, n, _, _) in REAL_DEMOS.items()
                 if self.karnataka is not None and (REAL_DEMO_DIR / f"{key}.csv").exists()]
        return {
            "datasets": datasets,
            "demos": demos,
            "params": {"D": p.D, "delta": p.delta, "beta": p.beta, "K": p.K, "n": p.n, "quorum": p.quorum,
                       "theta_on": self.theta_on, "theta_off": theta_off,
                       "signal_on": self._signal_level(self.theta_on), "min_sites": exp.min_quorum_count},
            "limits": {"eval_days": exp.eval_days, "warmup_days": exp.burn_in_days, "warmup_weeks": 14},
            "metrics": self.metrics,
            "defaults": {"outbreak": True, "epicentre": 14, "size": 6, "peak": 35, "onset": 7, "duration": 28,
                         "start": 40, "unreliable": 30, "seed": 7},
        }

    def _signal_level(self, theta):
        return float(self.params.K * (theta / (1 - theta)) ** (1 / self.params.n))

    # ---- scenarios ------------------------------------------------------------------------
    def _spec(self, spec):
        exp = self.cfg.experiment
        N = self.cfg.world.n_facilities

        def number(key, low, high, default, kind=float):
            try:
                value = kind(spec.get(key, default))
            except (TypeError, ValueError):
                value = default
            return min(max(value, low), high)

        return {
            "outbreak": bool(spec.get("outbreak", True)),
            "epicentre": number("epicentre", 0, N - 1, 14, int),
            "size": number("size", 1, 12, 6, int),
            "peak": number("peak", 5, 150, 35) / 100.0,
            "onset": number("onset", 1, 28, 7, int),
            "duration": number("duration", 7, 60, 28, int),
            "start": number("start", 7, exp.eval_days - 14, 40, int),
            "unreliable": number("unreliable", 0, 90, 30) / 100.0,
            "seed": number("seed", 0, 10 ** 6, 7, int),
        }

    def _generate(self, raw_spec):
        """Simulate one scenario with the study's data generator and reporting process."""
        s = self._spec(raw_spec)
        cfg, exp, w = self.cfg, self.cfg.experiment, self.world
        T = exp.n_days
        base = cfg.world.seed
        baseline = w.facilities["baseline_daily_count"].to_numpy()
        nominal = w.facilities["nominal_reliability"].to_numpy()

        rng = np.random.default_rng([base, SEED_OFFSET + s["seed"], 10])
        start = cfg.world.history_days + cfg.world.score_calibration_days + int(rng.integers(365))
        days = start + np.arange(T)
        rates = simulate_rates(baseline, days, rng, cfg.world)
        counts = rng.poisson(rates).astype(float)

        cluster = select_cluster(s["epicentre"], w.W, s["size"])
        t0 = exp.burn_in_days + s["start"]
        truth = None
        if s["outbreak"]:
            ob = replace(cfg.outbreak, ramp_days=s["onset"], duration_days=s["duration"])
            counts = inject_outbreak(counts, rates, cluster, t0, s["peak"],
                                     np.random.default_rng([base, SEED_OFFSET + s["seed"], 11]), ob)
            truth = {"cluster": cluster, "t0": t0,
                     "t_end": min(t0 + s["duration"] + cfg.outbreak.grace_days, T),
                     "end": min(t0 + s["duration"], T)}

        rng_rep = np.random.default_rng([base, SEED_OFFSET + s["seed"], 12])
        profile = make_profile(nominal, s["unreliable"], rng_rep, cfg.reporting)
        arrival, factor = draw_reporting(profile, T, rng_rep, cfg.reporting)
        reported = apply_reporting(counts, factor, arrival)
        return days, counts, reported, arrival, truth, profile.unreliable

    def simulate(self, raw_spec):
        days, counts, reported, arrival, truth, unreliable = self._generate(raw_spec)
        return self._run(days, reported, arrival, counts, truth, unreliable, source="simulated")

    def template_csv(self, raw_spec):
        """The scenario's reported counts, including warm-up days, as an editable case file."""
        days, _, reported, _, _, _ = self._generate(raw_spec)
        ids = self.world.facilities.facility_id.tolist()
        out = io.StringIO()
        out.write("date,facility_id,cases\n")
        for t, day in enumerate(days):
            date = to_date(day).isoformat()
            for k, fid in enumerate(ids):
                value = reported[t, k]
                out.write(f"{date},{fid},{'' if np.isnan(value) else int(value)}\n")
        return out.getvalue()

    def upload(self, text):
        if self.karnataka is not None:
            k = self.karnataka["module"]
            known = {k.site_id(d) for d in k.DISTRICTS}
            named = {k.site_id(n.upper()) for n in csv_facilities(text)}
            if named and named <= known:
                return self._upload_karnataka(text)
        exp = self.cfg.experiment
        ids = self.world.facilities.facility_id.tolist()
        days, reported = parse_cases_csv(text, ids)
        if len(days) < exp.burn_in_days + 14:
            raise ValueError(f"Please upload at least {exp.burn_in_days + 14} days: the first "
                             f"{exp.burn_in_days} are used to warm up the model.")
        arrival = np.where(np.isnan(reported), -1, np.arange(len(days))[:, None]).astype(np.int16)
        return self._run(days, reported, arrival, None, None, None, source="uploaded")

    def _upload_karnataka(self, text):
        k = self.karnataka["module"]
        rows = list(csv.DictReader(io.StringIO(text.strip())))
        column = next(c for c in rows[0] if c and c.strip().lower() == "facility_id")
        for row in rows:
            row[column] = k.site_id((row[column] or "").strip().upper())
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        ids = [k.site_id(d) for d in k.DISTRICTS]
        days, reported = parse_cases_csv(out.getvalue(), ids, step=7)
        warmup = k.WARMUP_WEEKS
        if len(days) < warmup + 4:
            raise ValueError(f"Please upload at least {warmup + 4} weeks: the first {warmup} warm up the model.")
        dates = [to_date(d) for d in days]
        # The model is the one for the year that most of the evaluated weeks fall in.
        years = pd.Series([d.year for d in dates[warmup:]])
        world = self._karnataka_world(int(years.mode().iloc[0]))
        arrival = np.where(np.isnan(reported), -1, np.arange(len(days))[:, None]).astype(np.int16)
        result = self._run(k.model_days(pd.to_datetime(dates)), reported, arrival, None, None, None,
                           source="uploaded", world=world, dates=dates, dataset="karnataka")
        result["model_note"] = (f"Model trained on {', '.join(world.train_period)} "
                                f"and calibrated on {world.calibration_period[0][:4]}")
        return result

    def demo(self, name):
        if name not in REAL_DEMOS or self.karnataka is None:
            raise ValueError("Unknown example.")
        path = REAL_DEMO_DIR / f"{name}.csv"
        if not path.exists():
            raise ValueError("This example file is missing. Run `python app/make_demo_data.py`.")
        result = self._upload_karnataka(path.read_text(encoding="utf8"))
        result["title"] = REAL_DEMOS[name][0]
        return result

    # ---- the model --------------------------------------------------------------------------
    def _run(self, days, reported, arrival, counts, truth, unreliable, source, world=None, dates=None,
             dataset="synthetic"):
        w = world if world is not None else self.world
        cfg, p = w.cfg, self.params
        exp = cfg.experiment
        T, N = reported.shape
        burn = exp.burn_in_days
        run = {"days": days, "arrival": arrival, "outbreak_reported": reported}
        part = _process(w, [run], "outbreak")
        A, H, _ = simulate_full(part, w.W, p)
        alert, active = qw_mask(H, self.theta_on, exp, w.neighbourhoods, p.quorum)
        production = (part.production_ml * part.weight)[0]
        # "Unusual" for display: the top 5% of a site's normal scores (calibrated tail probability <= 0.05).
        flagged = part.production_ml[0] >= np.log(cfg.features.production_tail / 0.05)

        cluster_mask = np.zeros((1, N), bool)
        if truth is not None:
            cluster_mask[0, truth["cluster"]] = True
            res = evaluate_mask(alert, burn, exp.refractory_days, np.array([truth["t0"]]),
                                np.array([truth["t_end"]]), cluster_mask)
        else:
            res = evaluate_mask(alert, burn, exp.refractory_days)
        events = self._events(alert[0], burn, exp.refractory_days)

        view = slice(burn, T)
        A_view = A[0, view]
        scale = float(max(np.percentile(A_view, 99.5), self._signal_level(self.theta_on) * 1.6, 1e-3))

        def grid(values, digits):
            rounded = np.round(np.asarray(values, float), digits)
            return [[None if np.isnan(v) else (int(v) if digits == 0 else float(v)) for v in row] for row in rounded]

        detection = None
        if truth is not None:
            day = int(res["detection_day"][0])
            detection = {"day": day - burn if day >= 0 else None, "delay": day - truth["t0"] if day >= 0 else None,
                         "sites": np.nonzero(alert[0, day])[0].tolist() if day >= 0 else []}
        return {
            "source": source,
            "dataset": dataset,
            "dates": [(dates[t] if dates is not None else to_date(days[t])).isoformat() for t in range(burn, T)],
            "warmup_days": burn,
            "scale": scale,
            "series": {
                "reported": grid(reported[view], 0),
                "cases": grid(counts[view], 0) if counts is not None else None,
                "signal": grid(A_view, 2),
                "activation": grid(H[0, view], 3),
                "production": grid(production[view], 2),
                "reliability": grid(part.reliability[0, view], 2),
                "flagged": flagged[view].astype(int).tolist(),
                "active": active[0, view].astype(int).tolist(),
                "alert": alert[0, view].astype(int).tolist(),
            },
            "truth": None if truth is None else {
                "cluster": truth["cluster"].tolist(), "start": truth["t0"] - burn, "end": truth["end"] - burn},
            "unreliable": None if unreliable is None else np.nonzero(unreliable)[0].tolist(),
            "events": [{"day": e["day"] - burn, "sites": e["sites"]} for e in events],
            "detection": detection,
            "false_alerts": int(res["n_false"][0]),
        }

    @staticmethod
    def _events(alert, burn, refractory):
        """Alert events after warm-up: onsets closer than the refractory period are one event."""
        onset = alert.copy()
        onset[1:] &= ~alert[:-1]
        onset[:burn] = False
        days = np.nonzero(onset.any(axis=1))[0]
        events = []
        for d in days:
            if events and d - events[-1]["last"] <= refractory:
                events[-1]["last"] = d
                continue
            events.append({"day": int(d), "last": int(d), "sites": np.nonzero(alert[d])[0].tolist()})
        return events
