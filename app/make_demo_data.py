"""Write ready-to-upload demo case files for the app, and record what the model does on each.

    python app/make_demo_data.py

Real files: Karnataka dengue surveillance, weekly lab-confirmed cases per district, exported from
real_data/karnataka_dengue_weekly.csv without changes. They were chosen after checking how the model
behaves on each year (real_data/README.md lists every year checked).

Synthetic files (in synthetic/): generated with the study's data generator, with answer keys.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_service import REAL_DEMOS, ModelService, to_date

OUT = Path(__file__).resolve().parent / "demo_data"
SYNTHETIC = OUT / "synthetic"

SCENARIOS = {
    "01_clear_outbreak": {"title": "Clear outbreak", "outbreak": True, "epicentre": 14, "size": 6, "peak": 60,
                          "onset": 7, "duration": 28, "start": 40, "unreliable": 10, "seed": 101},
    "02_messy_reporting": {"title": "Outbreak with half the hospitals reporting unreliably", "outbreak": True,
                           "epicentre": 22, "size": 10, "peak": 40, "onset": 7, "duration": 28, "start": 60,
                           "unreliable": 50, "seed": 404},
}


def write_real(service):
    k = service.karnataka["module"]
    matrix = service.karnataka["matrix"]
    lines = ["## Real data: Karnataka dengue surveillance", "",
             "Weekly lab-confirmed dengue cases (IgM + NS1) per district, from the Karnataka Dengue Daily Surveillance",
             "Reports, cleaned as described in `real_data/karnataka.py` and exported without further changes. Each file",
             "starts with 14 warm-up weeks. For each file the app rebuilds the model from earlier years only (see the",
             "model column). These periods were chosen after checking how the model behaves on every year from 2016",
             "to 2024; `real_data/README.md` lists them all.", "",
             "| File | Weeks | What it shows | Model | First regional alert |", "|---|---|---|---|---|"]
    for name, (title, note, start, end) in REAL_DEMOS.items():
        part = matrix.loc[start:end]
        rows = ["date,facility_id,cases"]
        for date, values in part.iterrows():
            for district, value in values.items():
                rows.append(f"{date.date().isoformat()},{k.label(district)},{'' if value != value else int(value)}")
        text = "\n".join(rows) + "\n"
        (OUT / f"{name}.csv").write_text(text, encoding="utf8")
        result = service.upload(text)
        events = result["events"]
        names = [f["label"] for f in service.meta()["datasets"]["karnataka"]["facilities"]]
        if events:
            first = events[0]
            outcome = f"{result['dates'][first['day']]}: {', '.join(names[i] for i in first['sites'])}"
            if len(events) > 1:
                outcome += f" (+{len(events) - 1} later alert{'s' if len(events) > 2 else ''})"
        else:
            outcome = "No alert"
        lines.append(f"| `{name}.csv` | {start} to {end} | {title}: {note} | {result['model_note']} | {outcome} |")
        print(name, "|", result["model_note"], "|", outcome)
    return lines


def write_synthetic(service):
    SYNTHETIC.mkdir(exist_ok=True)
    ids = service.world.facilities.facility_id.tolist()
    lines = ["## Synthetic data (in `synthetic/`)", "",
             "Generated with the study's data generator. The answer key is what was simulated; the last column is",
             "what the model did when the file was uploaded (uploaded reports count as on time).", "",
             "| File | Scenario | Outbreak starts | Hospitals affected | Model result |", "|---|---|---|---|---|"]
    for name, spec in SCENARIOS.items():
        csv_text = service.template_csv(spec)
        (SYNTHETIC / f"{name}.csv").write_text(csv_text, encoding="utf8")
        days, _, _, _, truth, _ = service._generate(spec)
        result = service.upload(csv_text)
        start_view = truth["t0"] - result["warmup_days"]
        hits = [e for e in result["events"]
                if start_view <= e["day"] < truth["end"] - result["warmup_days"] + 7 and set(e["sites"]) & set(truth["cluster"])]
        outcome = (f"Alert on {result['dates'][hits[0]['day']]}, {hits[0]['day'] - start_view} days after onset"
                   if hits else "Not detected")
        others = len(result["events"]) - (1 if hits else 0)
        if others:
            outcome += f" (+{others} false alert{'s' if others > 1 else ''})"
        key = {"title": spec["title"], "settings": spec, "outbreak_start": to_date(days[truth["t0"]]).isoformat(),
               "affected_hospitals": [ids[i] for i in truth["cluster"]], "model_result": outcome}
        (SYNTHETIC / f"{name}_answer_key.json").write_text(json.dumps(key, indent=1), encoding="utf8")
        lines.append(f"| `synthetic/{name}.csv` | {spec['title']} | {key['outbreak_start']} | "
                     f"{', '.join(key['affected_hospitals'])} | {outcome} |")
        print(name, outcome)
    return lines


def main():
    OUT.mkdir(exist_ok=True)
    service = ModelService()
    lines = ["# Demo case files", "",
             "Upload any of these with the app's *Upload CSV* button; the Karnataka examples are also in the data menu at the top of the Explore view.", ""]
    lines += write_real(service) + [""] + write_synthetic(service)
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
