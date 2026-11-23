from datetime import datetime
import json
import math
from pathlib import Path
import re
import threading
from typing import Any, Union

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

if __package__:
    from . import matching_func
else:
    import matching_func


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
STORE = BACKEND_DIR / "intake_store"
STORE.mkdir(exist_ok=True)
DATA_FILE = BACKEND_DIR / "user_requests_filtered_activities_formatted.csv"
CLUSTERS_FILE = BACKEND_DIR / "clusters.csv"
DATA_LOCK = threading.RLock()

LANGUAGE_INPUTS = {
    "english": "English",
    "chinese": "Chinese",
    "both": "Both",
}
AVAILABILITY_PATTERN = re.compile(
    r"Available:\s*(\d{4}-\d{2}-\d{2})\s+at\s+(\d{1,2}:\d{2})"
    r"\s*[–—-]\s*(\d{4}-\d{2}-\d{2})\s+at\s+(\d{1,2}:\d{2})",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=422, detail=f"'{key}' is required.")
    return value.strip()


def _number(record: dict[str, Any], key: str, *, integer: bool = False) -> Union[int, float]:
    value = record.get(key)
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"'{key}' must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"'{key}' must be a number.") from None
    if not math.isfinite(number) or (integer and not number.is_integer()):
        raise HTTPException(status_code=422, detail=f"'{key}' has an invalid value.")
    return int(number) if integer else number


def _parse_availability(hard_constraints: str) -> str:
    match = AVAILABILITY_PATTERN.search(hard_constraints)
    if match is None:
        raise HTTPException(
            status_code=422,
            detail="Could not parse an 'Available: start – end' time range from hardConstraints.",
        )
    try:
        start = datetime.strptime(
            f"{match.group(1)} {match.group(2)}",
            "%Y-%m-%d %H:%M",
        )
        end = datetime.strptime(
            f"{match.group(3)} {match.group(4)}",
            "%Y-%m-%d %H:%M",
        )
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="hardConstraints contains an invalid date or time.",
        ) from None

    if start.date() != end.date():
        raise HTTPException(
            status_code=422,
            detail="Availability must start and end on the same day to match the CSV format.",
        )
    if end <= start:
        raise HTTPException(
            status_code=422,
            detail="Availability end must be later than its start.",
        )
    return f"{start:%Y/%m/%d %H:%M}-{end:%H:%M}"


def _record_to_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    email = _required_string(record, "email").lower()
    if not EMAIL_PATTERN.fullmatch(email):
        raise HTTPException(status_code=422, detail="'email' must be a valid email address.")

    hard_constraints = _required_string(record, "hardConstraints")
    language = _required_string(record, "language").lower()
    if language not in LANGUAGE_INPUTS:
        raise HTTPException(
            status_code=422,
            detail="'language' must be English, Chinese, or Both.",
        )

    budget_min = _number(record, "budgetMin")
    budget_max = _number(record, "budgetMax")
    min_party = _number(record, "minParty", integer=True)
    max_party = _number(record, "maxParty", integer=True)
    if budget_min < 0 or budget_max < budget_min:
        raise HTTPException(status_code=422, detail="Budget range is invalid.")
    if min_party < 1 or max_party < min_party:
        raise HTTPException(status_code=422, detail="Participant range is invalid.")

    return {
        "email": email,
        "Activity_Type": _required_string(record, "activity"),
        "Availability": _parse_availability(hard_constraints),
        "Location_Zone": _required_string(record, "location"),
        "Budget_Min_USD": budget_min,
        "Budget_Max_USD": budget_max,
        "Min_Group_Size": min_party,
        "Max_Group_Size": max_party,
        "Language_Preference": LANGUAGE_INPUTS[language],
        "Social_Energy": _required_string(record, "socialEnergy"),
        "Soft_Preferences": str(record.get("softPreferences") or "None specified").strip(),
        "Reference_Group_ID": "",
    }


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> Path:
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    return temporary


def _load_or_create_cluster_snapshot() -> pd.DataFrame:
    snapshot = None
    if CLUSTERS_FILE.exists():
        snapshot = pd.read_csv(
            CLUSTERS_FILE,
            dtype={
                "cluster_id": str,
                "email": str,
                "cluster_availability": str,
                "cluster_location_zone": str,
                "cluster_language_preference": str,
            },
        )
        cluster_columns = [
            column
            for column in snapshot.columns
            if column.startswith("cluster_") and column != "cluster_id"
        ]
        stale_snapshot = (
            "cluster_activity_type" not in snapshot.columns
            or snapshot[cluster_columns].astype(str).eq("NOT A MATCH").any().any()
        )
        if not stale_snapshot:
            return snapshot

    requests = matching_func.open_file(DATA_FILE)
    clusters = matching_func.cluster_requests(requests)
    snapshot = matching_func.cluster_members_frame(requests, clusters)
    if _has_not_a_match_constraint(snapshot):
        raise ValueError("Matching produced a cluster with incompatible hard constraints.")
    temporary = _write_csv_atomic(snapshot, CLUSTERS_FILE)
    temporary.replace(CLUSTERS_FILE)
    return snapshot


def _has_not_a_match_constraint(snapshot: pd.DataFrame) -> bool:
    cluster_columns = [
        column
        for column in snapshot.columns
        if column.startswith("cluster_") and column != "cluster_id"
    ]
    return snapshot[cluster_columns].astype(str).eq("NOT A MATCH").any().any()


def _text(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def _plans_for_email(snapshot: pd.DataFrame, email: str) -> list[dict[str, Any]]:
    matching_rows = snapshot[
        snapshot["email"].fillna("").str.casefold() == email.casefold()
    ].sort_values("row_index")
    plans = []

    for _, request in matching_rows.iterrows():
        cluster_id = _text(request["cluster_id"])
        members = snapshot[snapshot["cluster_id"] == cluster_id]
        member_summaries = [
            {
                "email": _text(member["email"]),
                "activity": _text(member["Activity_Type"]),
                "availability": _text(member["Availability"]),
                "socialEnergy": _text(member["Social_Energy"]),
                "softPreferences": _text(member["Soft_Preferences"]),
                "isCurrentUser": int(member["row_index"]) == int(request["row_index"]),
            }
            for _, member in members.iterrows()
        ]
        plans.append({
            "planId": int(request["row_index"]),
            "activity": _text(request["Activity_Type"]),
            "availability": _text(request["Availability"]),
            "location": _text(request["Location_Zone"]),
            "budgetMin": float(request["Budget_Min_USD"]),
            "budgetMax": float(request["Budget_Max_USD"]),
            "minParty": int(request["Min_Group_Size"]),
            "maxParty": int(request["Max_Group_Size"]),
            "language": _text(request["Language_Preference"]),
            "socialEnergy": _text(request["Social_Energy"]),
            "softPreferences": _text(request["Soft_Preferences"]),
            "cluster": {
                "id": cluster_id,
                "memberCount": len(members),
                "availability": _text(request["cluster_availability"]),
                "location": _text(request["cluster_location_zone"]),
                "minGroupSize": (
                    int(request["cluster_min_group_size"])
                    if pd.notna(request["cluster_min_group_size"])
                    else None
                ),
                "maxGroupSize": (
                    int(request["cluster_max_group_size"])
                    if pd.notna(request["cluster_max_group_size"])
                    else None
                ),
                "budgetMin": (
                    float(request["cluster_min_budget_usd"])
                    if pd.notna(request["cluster_min_budget_usd"])
                    else None
                ),
                "budgetMax": (
                    float(request["cluster_max_budget_usd"])
                    if pd.notna(request["cluster_max_budget_usd"])
                    else None
                ),
                "language": _text(request["cluster_language_preference"]),
                "members": member_summaries,
            },
        })

    return plans


@app.get("/api/plans")
def get_plans(email: str = Query(..., min_length=3)):
    normalized_email = email.strip().lower()
    if not EMAIL_PATTERN.fullmatch(normalized_email):
        raise HTTPException(status_code=422, detail="'email' must be a valid email address.")

    with DATA_LOCK:
        try:
            snapshot = _load_or_create_cluster_snapshot()
        except ImportError as error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The NLI model dependency is missing. Install "
                    "pennside_backend/requirements.txt before loading plans."
                ),
            ) from error
        except matching_func.NLIModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "email": normalized_email,
            "plans": _plans_for_email(snapshot, normalized_email),
        }


@app.post("/api/intake")
def receive_intake(payload: dict[str, Any] = Body(...)):
    record = payload.get("record", payload)
    if not isinstance(record, dict):
        raise HTTPException(status_code=422, detail="'record' must be a JSON object.")
    csv_row = _record_to_csv_row(record)

    with DATA_LOCK:
        requests = matching_func.open_file(DATA_FILE)
        row_index = len(requests)
        updated_requests = pd.concat(
            [
                requests,
                pd.DataFrame([csv_row], columns=matching_func.CSV_COLUMNS),
            ],
            ignore_index=True,
        )
        try:
            clusters = matching_func.cluster_requests(updated_requests)
        except ImportError as error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The NLI model dependency is missing. Install "
                    "pennside_backend/requirements.txt before submitting a plan."
                ),
            ) from error
        except matching_func.NLIModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        updated_requests = matching_func.assign_cluster_ids_to_reference_groups(
            updated_requests,
            clusters,
        )
        snapshot = matching_func.cluster_members_frame(updated_requests, clusters)
        requests_temp = _write_csv_atomic(updated_requests, DATA_FILE)
        clusters_temp = _write_csv_atomic(snapshot, CLUSTERS_FILE)
        requests_temp.replace(DATA_FILE)
        clusters_temp.replace(CLUSTERS_FILE)

        current_plans = _plans_for_email(snapshot, csv_row["email"])
        submitted_plan = next(
            plan for plan in current_plans if plan["planId"] == row_index
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        saved_name = f"{timestamp}.json"
        (STORE / saved_name).write_text(
            json.dumps(
                {
                    "key": payload.get("key"),
                    "record": record,
                    "plan": submitted_plan,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"[intake] {timestamp} email={csv_row['email']!r} "
        f"activity={csv_row['Activity_Type']!r} "
        f"cluster={submitted_plan['cluster']['id']}"
    )
    return {
        "ok": True,
        "saved": saved_name,
        "plan": submitted_plan,
        "plans": current_plans,
    }
