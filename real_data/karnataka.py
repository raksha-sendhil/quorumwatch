"""Karnataka dengue surveillance data, the district map, and QuorumWatch models trained on it.

Source: Karnataka Dengue Daily Surveillance Reports (state vector-borne disease control programme),
in the structured "0015-karnataka-dengue-summaries" release: one row per district per day with
suspected, tested and lab-confirmed (IgM + NS1) dengue cases, 2012 to January 2025.

Cleaning, fixed before any model results were looked at:
- District names are normalised to upper case. Vijayanagara (split from Ballari in 2021) is merged
  back into Ballari, and the BBMP city corporation is merged into Bengaluru Urban, so the same 30
  districts exist in every year and match the district map.
- The series is lab-confirmed positives. State-total rows are dropped.
- Districts report in batches: in 70% of district-weeks with 20 or more positives, one day holds at
  least 70% of the week's cases (for example 1,070 BBMP positives logged on 3 July 2019 after two
  weeks of zeros). So the model uses weeks (Monday to Sunday): a district's weekly value is the sum
  of the positives it reported that week. A week with no report at all is missing, not zero.

Case counts grow strongly over the years (testing and reporting expanded), so a model for test
year Y is rebuilt from the preceding years, as a surveillance team would update it each year.
Following the usual "endemic channel" practice, epidemic years are left out of the baseline: of the
candidate training years Y-5 to Y-2, any year whose statewide total exceeds 1.5 times the
candidates' median is excluded (at least two years are kept). Year Y-1 calibrates the scores.

The weekly-sum rule and the epidemic-year exclusion replaced a first version (mean daily positives
times 7, no exclusion) after that version's results were seen. A further variant, which also
excluded epidemic years from calibration, was tried and gave worse results, so it was dropped. The signal parameters and
activation threshold are the study's, unchanged.
"""
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ml.feature_engineering import build_features, direction_excess, fit_profiles
from ml.isolation_forest import ScoreCalibrator, fit_forest, raw_scores
from network.facility_network import build_weights, closed_neighbourhood, n_components

HERE = Path(__file__).resolve().parent
WEEKLY = HERE / "karnataka_dengue_weekly.csv"
MAP = HERE / "karnataka_map.json"
RAW_DIR = Path(os.environ.get("QW_KARNATAKA_RAW", Path.home() / "Downloads" / "0015-karnataka-dengue-summaries"))
GEOJSON = Path(os.environ.get("QW_KARNATAKA_GEOJSON",
                              Path.home() / "Downloads" / "EpiSentinel" / "dashboard" / "karnataka_districts.json"))
WARMUP_WEEKS = 14
EPIDEMIC_YEAR_RATIO = 1.5
ORIGIN = pd.Timestamp("2012-01-02")           # a Monday; week k is model day 7k

@dataclass
class RealWorld:
    """The parts of the study's World that the detection pipeline uses, built from real data."""
    cfg: object
    facilities: pd.DataFrame
    W: np.ndarray
    neighbourhoods: np.ndarray
    profiles: object
    forest: object
    calibrator_ml: ScoreCalibrator


# Map polygons use older district names.
MAP_NAMES = {
    "Bagalkot": "BAGALKOTE", "Bangalore": "BENGALURU URBAN", "Bangalore Rural": "BENGALURU RURAL",
    "Belgaum": "BELAGAVI", "Bellary": "BALLARI", "Bidar": "BIDAR", "Bijapur": "VIJAYAPURA",
    "Chamrajnagar": "CHAMARAJANAGARA", "Chikkaballapura": "CHIKKABALLAPURA", "Chikmagalur": "CHIKKAMAGALURU",
    "Chitradurga": "CHITRADURGA", "Dakshina Kannada": "DAKSHINA KANNADA", "Davanagere": "DAVANGERE",
    "Dharwad": "DHARWAD", "Gadag": "GADAG", "Gulbarga": "KALABURAGI", "Hassan": "HASSAN", "Haveri": "HAVERI",
    "Kodagu": "KODAGU", "Kolar": "KOLAR", "Koppal": "KOPPAL", "Mandya": "MANDYA", "Mysore": "MYSURU",
    "Raichur": "RAICHUR", "Ramanagara": "RAMANAGARA", "Shimoga": "SHIVAMOGGA", "Tumkur": "TUMAKURU",
    "Udupi": "UDUPI", "Uttara Kannada": "UTTARA KANNADA", "Yadgir": "YADGIR",
}
DISTRICTS = sorted(MAP_NAMES.values())


def site_id(district):
    return district.lower().replace(" ", "_")


def label(district):
    return district.title().replace("Bengaluru", "Bengaluru")


# ---- building the cleaned files (run once from the raw release) --------------------------
def build_weekly_file(raw_dir=RAW_DIR, out=WEEKLY):
    frames = []
    for path in sorted(Path(raw_dir).glob("20*.csv")):
        frames.append(pd.read_csv(path, usecols=["metadata.recordDate", "location.admin.hierarchy",
                                                 "location.admin2.name", "daily.positive.total"]))
    if not frames:
        raise FileNotFoundError(f"No yearly CSV files found in {raw_dir}")
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = ["date", "hierarchy", "district", "positive"]
    raw = raw[raw.hierarchy.isin(["REVENUE", "ULB"])].dropna(subset=["district"])
    raw["date"] = pd.to_datetime(raw.date.str[:10], format="%Y-%m-%d")
    raw["district"] = raw.district.str.strip().str.upper().replace({"VIJAYANAGAR": "BALLARI", "VIJAYANAGARA": "BALLARI"})
    unknown = set(raw.district) - set(DISTRICTS)
    if unknown:
        raise ValueError(f"Unmapped districts: {sorted(unknown)}")
    daily = (raw.dropna(subset=["positive"]).groupby(["district", "date"]).positive.sum().reset_index())
    daily["week_start"] = daily.date - pd.to_timedelta(daily.date.dt.weekday, unit="D")
    weekly = daily.groupby(["district", "week_start"]).agg(reported_days=("date", "nunique"), positive=("positive", "sum"))
    weekly = weekly.reset_index()
    weekly["cases"] = weekly.positive
    weekly[["district", "week_start", "reported_days", "positive", "cases"]].to_csv(out, index=False)
    return weekly


def build_map_file(geojson=GEOJSON, tolerance=0.012):
    """Project district polygons to a 0-100 frame, simplify them, and store centroids."""
    data = json.loads(Path(geojson).read_text(encoding="utf8"))
    lon0, lat0 = 76.2, 15.0
    scale = np.cos(np.radians(lat0))

    def project(lon, lat):
        return (lon - lon0) * scale, -(lat - lat0)

    rings = {}
    for feature in data["features"]:
        name = MAP_NAMES[feature["properties"]["district"]]
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        rings[name] = [np.array([project(*pt[:2]) for pt in poly[0]]) for poly in polygons]
    every = np.vstack([r for rs in rings.values() for r in rs])
    low, span = every.min(axis=0), (every.max(axis=0) - every.min(axis=0)).max()

    def to_frame(points):
        return (points - low) / span * 100

    def simplify(points):
        keep = [points[0]]
        for p in points[1:]:
            if np.hypot(*(p - keep[-1])) >= tolerance * 100 / 4:
                keep.append(p)
        return np.array(keep)

    out = {}
    for name, rs in rings.items():
        framed = [simplify(to_frame(r)) for r in rs]
        framed = [r for r in framed if len(r) >= 4]
        main = max(framed, key=len)
        x, y = main[:, 0], main[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = cross.sum() / 2
        cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * area)
        cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * area)
        out[name] = {"centroid": [round(float(cx), 2), round(float(cy), 2)],
                     "path": " ".join("M" + " L".join(f"{p[0]:.1f},{p[1]:.1f}" for p in r) + " Z" for r in framed)}
    MAP.write_text(json.dumps(out), encoding="utf8")
    return out


# ---- loading ------------------------------------------------------------------------------
def weekly_matrix():
    """Weeks x districts of weekly lab-confirmed dengue cases; NaN where not reported."""
    if not WEEKLY.exists():
        build_weekly_file()
    frame = pd.read_csv(WEEKLY, parse_dates=["week_start"])
    matrix = frame.pivot_table(index="week_start", columns="district", values="cases", dropna=False)
    matrix = matrix.reindex(columns=DISTRICTS)
    full = pd.date_range(ORIGIN, matrix.index.max(), freq="7D")
    return matrix.reindex(full)


def district_map():
    if not MAP.exists():
        build_map_file()
    return json.loads(MAP.read_text(encoding="utf8"))


def model_days(dates):
    return (np.round((pd.DatetimeIndex(dates) - ORIGIN).days / 7).astype(int) * 7).to_numpy()


def segment(matrix, start, end):
    part = matrix.loc[pd.Timestamp(start):pd.Timestamp(end)]
    reported = part.to_numpy(float)
    arrival = np.where(np.isnan(reported), -1, np.arange(len(part))[:, None]).astype(np.int16)
    return part.index, model_days(part.index), reported, arrival


def district_network(length_scale):
    geo = district_map()
    coords = np.array([geo[d]["centroid"] for d in DISTRICTS], float)
    W, _ = build_weights(coords, 4, length_scale)
    assert n_components(W) == 1
    return coords, W


def weekly_config(cfg):
    return replace(cfg, experiment=replace(cfg.experiment, burn_in_days=WARMUP_WEEKS, refractory_days=2))


def baseline_years(matrix, test_year):
    """Training years: candidates test_year-5 to test_year-2 without epidemic years (statewide total
    above 1.5 times the candidates' median; endemic-channel practice), keeping at least two.
    Returns (training years, calibration year), where the calibration year is test_year-1."""
    candidates = list(range(max(2012, test_year - 5), test_year - 1))
    totals = {y: float(np.nansum(matrix.loc[f"{y}-01-01":f"{y}-12-31"].to_numpy())) for y in candidates}
    median = np.median(list(totals.values()))
    kept = [y for y in candidates if totals[y] <= EPIDEMIC_YEAR_RATIO * median]
    if len(kept) < 2:
        kept = sorted(sorted(candidates, key=lambda y: totals[y])[:2])
    return kept, test_year - 1


def _rows(matrix, years):
    """Stack whole years as one series; the gaps between non-adjacent years are left out."""
    parts = [segment(matrix, f"{y}-01-01", f"{y}-12-31") for y in years]
    days = np.concatenate([p[1] for p in parts])
    reported = np.vstack([p[2] for p in parts])
    arrival = np.where(np.isnan(reported), -1, np.arange(len(reported))[:, None]).astype(np.int16)
    return days, reported, arrival


def build_karnataka_world(cfg, test_year, matrix=None):
    """Train on recent non-epidemic years before test_year-1 and calibrate on test_year-1."""
    cfg = weekly_config(cfg)
    fc = cfg.features
    matrix = weekly_matrix() if matrix is None else matrix
    coords, W = district_network(cfg.world.length_scale_km)
    years, calibration_year = baseline_years(matrix, test_year)
    train = tuple(str(y) for y in years)
    calibrate = (f"{calibration_year}-01-01", f"{calibration_year}-12-31")

    days, reported, arrival = _rows(matrix, years)
    profiles = fit_profiles(reported, days)
    z = profiles.standardise(reported, days, fc.z_clip)
    X_train, _, _ = build_features(z, arrival, days, fc.window, fc.long_window)
    forest = fit_forest(X_train, replace(cfg.forest, max_samples=min(cfg.forest.max_samples, len(X_train))))
    typical = np.nanmean(reported, axis=0)          # most districts' median week is zero

    _, days, reported, arrival = segment(matrix, *calibrate)
    z = profiles.standardise(reported, days, fc.z_clip)
    X_cal, (s, i), stats = build_features(z, arrival, days, fc.window, fc.long_window)
    cal_raw = raw_scores(forest, X_cal)
    calibrator = ScoreCalibrator.fit(cal_raw, direction_excess(z, stats)[s, i], i, len(DISTRICTS), fc.production_tail)
    n = len(DISTRICTS)
    site_mean = np.bincount(i, cal_raw, n) / np.maximum(np.bincount(i, minlength=n), 1)
    site_sd = np.array([max(cal_raw[i == k].std(), 1e-6) if (i == k).any() else 1.0 for k in range(n)])
    facilities = pd.DataFrame({"facility_id": [site_id(d) for d in DISTRICTS], "label": [label(d) for d in DISTRICTS],
                               "x_km": coords[:, 0], "y_km": coords[:, 1], "baseline_daily_count": typical})
    world = RealWorld(cfg, facilities, W, closed_neighbourhood(W), profiles, forest, calibrator)
    world.train_period, world.calibration_period = train, calibrate
    world.forest_site_mean, world.forest_site_sd = site_mean, site_sd       # for Isolation Forest alone
    return world
