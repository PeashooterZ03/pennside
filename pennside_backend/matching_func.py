from datetime import datetime
import re

import pandas as pd

CSV_COLUMNS = (
    "email",
    "Activity_Type",
    "Availability",
    "Location_Zone",
    "Budget_Min_USD",
    "Budget_Max_USD",
    "Min_Group_Size",
    "Max_Group_Size",
    "Language_Preference",
    "Social_Energy",
    "Soft_Preferences",
    "Reference_Group_ID",
)

LANGUAGE_OPTIONS = {
    "English": frozenset({"English"}),
    "Chinese": frozenset({"Chinese"}),
    "Both": frozenset({"English", "Chinese"}),
}

def open_file(file_path):
    df = pd.read_csv(
        file_path,
        dtype={
            "email": str,
            "Availability": str,
            "Language_Preference": str,
            "Reference_Group_ID": str,
        },
    )
    if tuple(df.columns) != CSV_COLUMNS:
        raise ValueError(
            "CSV columns must match user_requests_filtered_activities_formatted.csv "
            f"in name and order: {', '.join(CSV_COLUMNS)}"
        )
    df["Reference_Group_ID"] = df["Reference_Group_ID"].fillna("")
    return df


def extract_hard_constraint_columns(df):
    columns = [
        "Activity_Type",
        "Availability",
        "Location_Zone",
        "Budget_Min_USD",
        "Budget_Max_USD",
        "Min_Group_Size",
        "Max_Group_Size",
        "Language_Preference",
    ]
    return df[columns]

# return the match result or "NOT A MATCH" if the two inputs do not match.
# 1 if match, 0 if not match
def match_Activity_Type(Activity_Type_1, Activity_Type_2):
    if Activity_Type_1.strip() == Activity_Type_2.strip():
        return Activity_Type_1, 1
    else:
        return "NOT A MATCH", 0
    
def _parse_availability(s):
    if not isinstance(s, str):
        return None
    m = re.fullmatch(
        r"(\d{4}/\d{2}/\d{2}) (\d{2}:\d{2})-(\d{2}:\d{2})",
        s.strip(),
    )
    if not m:
        return None
    try:
        day = datetime.strptime(m.group(1), "%Y/%m/%d").date()
        start = datetime.strptime(m.group(2), "%H:%M").time()
        end = datetime.strptime(m.group(3), "%H:%M").time()
    except ValueError:
        return None
    if end <= start:
        return None
    return day, start, end


def match_Availability(Availability_1, Availability_2):
    a = _parse_availability(Availability_1)
    b = _parse_availability(Availability_2)

    if a is None or b is None:
        return "NOT A MATCH", 0

    day1, start1, end1 = a
    day2, start2, end2 = b

    if day1 != day2:
        return "NOT A MATCH", 0

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start < overlap_end:
        overlap_str = (
            f"{day1:%Y/%m/%d} {overlap_start:%H:%M}-{overlap_end:%H:%M}"
        )
        return overlap_str, 1

    return "NOT A MATCH", 0


def match_Location_Zone(Location_Zone_1, Location_Zone_2):
    if Location_Zone_1.strip() == Location_Zone_2.strip():
        return Location_Zone_1, 1
    else:
        return "NOT A MATCH", 0

def match_Language_Preference(Language_Preference_1, Language_Preference_2):
    languages_1 = LANGUAGE_OPTIONS.get(Language_Preference_1)
    languages_2 = LANGUAGE_OPTIONS.get(Language_Preference_2)
    if languages_1 is None or languages_2 is None:
        return "NOT A MATCH", 0

    common_languages = languages_1 & languages_2
    if not common_languages:
        return "NOT A MATCH", 0
    if len(common_languages) == 2:
        return "Both", 1
    return next(iter(common_languages)), 1

def match_Group_Size(Min_Group_Size_1, Max_Group_Size_1, Min_Group_Size_2, Max_Group_Size_2):
    if (Min_Group_Size_1 <= Max_Group_Size_2) and (Min_Group_Size_2 <= Max_Group_Size_1):
        return (max(Min_Group_Size_1, Min_Group_Size_2), min(Max_Group_Size_1, Max_Group_Size_2), 1)
    else:
        return "NOT A MATCH", 0

def match_Budget_Range(Budget_Min_1, Budget_Max_1, Budget_Min_2, Budget_Max_2):
    if (Budget_Min_1 <= Budget_Max_2) and (Budget_Min_2 <= Budget_Max_1):
        return (max(Budget_Min_1, Budget_Min_2), min(Budget_Max_1, Budget_Max_2), 1)
    return "NOT A MATCH", 0

def merge_soft_preferences(Social_Energy, Soft_Preference):
    se = str(Social_Energy).strip() if pd.notna(Social_Energy) else "Unrestricted"
    sp = str(Soft_Preference).strip() if pd.notna(Soft_Preference) else "Unrestricted"

    return f"social energy: {se}, soft preference: {sp}"
        
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
_NLI_MODEL_CACHE = {}
_DURATION_BOUND_PATTERNS = (
    re.compile(
        r"^\s*(?P<operator>less than|under|below|up to|at most|no more than|"
        r"over|more than|above|at least|no less than)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>minutes?|mins?|hours?|hrs?)\s*\.?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?P<operator>少于|不到|不超过|至多|最多|以内|超过|多于|至少|不低于)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>分钟|分|小时|时)\s*。?\s*$"
    ),
)
_MAX_DURATION_OPERATORS = {
    "less than", "under", "below", "up to", "at most", "no more than",
    "少于", "不到", "不超过", "至多", "最多", "以内",
}


def _parse_duration_bound(text):
    for pattern in _DURATION_BOUND_PATTERNS:
        match = pattern.fullmatch(str(text))
        if match is None:
            continue

        value = float(match.group("value"))
        unit = match.group("unit").lower()
        if unit in {"hour", "hours", "hr", "hrs", "小时", "时"}:
            value *= 60
        bound_type = (
            "maximum"
            if match.group("operator").lower() in _MAX_DURATION_OPERATORS
            else "minimum"
        )
        return bound_type, value
    return None


def _duration_compatibility(text1, text2):
    first_bound = _parse_duration_bound(text1)
    second_bound = _parse_duration_bound(text2)
    if first_bound is None or second_bound is None:
        return None

    first_type, first_value = first_bound
    second_type, second_value = second_bound
    if first_type == second_type:
        smaller, larger = sorted((first_value, second_value))
        return 1.0 if larger == 0 else smaller / larger

    upper_bound = first_value if first_type == "maximum" else second_value
    lower_bound = first_value if first_type == "minimum" else second_value
    scale = max(upper_bound, lower_bound, 1.0)
    if lower_bound < upper_bound:
        return (upper_bound - lower_bound) / scale
    if lower_bound == upper_bound:
        return 0.0
    return -min((lower_bound - upper_bound) / scale, 1.0)


class NLIModelUnavailableError(RuntimeError):
    pass


def _get_nli_model(model_name=NLI_MODEL_NAME):
    if model_name not in _NLI_MODEL_CACHE:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except OSError as error:
            raise NLIModelUnavailableError(
                f"Unable to load NLI model {model_name!r}; download it or ensure "
                "Hugging Face model access is available."
            ) from error
        model.to("cpu")
        model.eval()

        label_indices = {}
        for index, label in model.config.id2label.items():
            normalized_label = str(label).strip().lower()
            for nli_label in ("entailment", "neutral", "contradiction"):
                if nli_label in normalized_label:
                    label_indices[nli_label] = int(index)
        if set(label_indices) != {"entailment", "neutral", "contradiction"}:
            raise ValueError(
                f"NLI model {model_name!r} must define entailment, neutral, "
                "and contradiction labels."
            )

        _NLI_MODEL_CACHE[model_name] = (tokenizer, model, label_indices)
    return _NLI_MODEL_CACHE[model_name]


def _predict_nli_probabilities(text1, text2, model_name=NLI_MODEL_NAME):
    import torch

    tokenizer, model, label_indices = _get_nli_model(model_name)
    encoded = tokenizer(
        [text1, text2],
        [text2, text1],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1).tolist()
    return [
        {
            label: float(probabilities[pair_index][label_index])
            for label, label_index in label_indices.items()
        }
        for pair_index in range(2)
    ]


def _score_bidirectional_nli(forward_probabilities, reverse_probabilities):
    forward_label = max(forward_probabilities, key=forward_probabilities.get)
    reverse_label = max(reverse_probabilities, key=reverse_probabilities.get)
    labels = {forward_label, reverse_label}

    if labels == {"entailment"}:
        return min(
            forward_probabilities["entailment"],
            reverse_probabilities["entailment"],
        )
    if labels == {"contradiction"}:
        return -min(
            forward_probabilities["contradiction"],
            reverse_probabilities["contradiction"],
        )
    if labels == {"entailment", "neutral"}:
        return max(
            forward_probabilities["entailment"],
            reverse_probabilities["entailment"],
        )
    if labels == {"entailment", "contradiction"}:
        return -max(
            forward_probabilities["contradiction"],
            reverse_probabilities["contradiction"],
        )
    return 0.0


def calculate_compatibility(text1, text2, model_name=NLI_MODEL_NAME):
    duration_score = _duration_compatibility(text1, text2)
    if duration_score is not None:
        return duration_score

    forward_probabilities, reverse_probabilities = _predict_nli_probabilities(
        text1,
        text2,
        model_name,
    )
    return _score_bidirectional_nli(
        forward_probabilities,
        reverse_probabilities,
    )


def calculate_overall_score(row, candidate_cluster):
    if not check_hard_constraints(row, candidate_cluster):
        return float("-inf")

    compatibilities = []
    for member_row in candidate_cluster.get_members():
        social_energy_score = calculate_compatibility(
            row["Social_Energy"],
            member_row["Social_Energy"],
        )
        soft_preference_score = calculate_compatibility(
            row["Soft_Preferences"],
            member_row["Soft_Preferences"],
        )
        compatibilities.append(
            0.2 * social_energy_score + 0.8 * soft_preference_score
        )

    average_compatibility = sum(compatibilities) / len(compatibilities)
    required_minimum = max(row["Min_Group_Size"], candidate_cluster.min_group_size)
    reached_minimum = len(candidate_cluster.get_members()) >= required_minimum
    group_size_score = -1.0 if reached_minimum else 0.0
    return 0.7 * average_compatibility + 0.3 * group_size_score


def check_hard_constraints(row, candidate_cluster):
    if not candidate_cluster.hard_constraints_valid:
        return False

    email = str(row.get("email", "")).casefold()
    if email and any(
        str(member.get("email", "")).casefold() == email
        for member in candidate_cluster.get_members()
    ):
        return False

    if match_Activity_Type(row['Activity_Type'], candidate_cluster.activity_type)[1] == 0:
        return False

    if match_Availability(row['Availability'], candidate_cluster.availability)[1] == 0:
        return False

    if match_Location_Zone(row['Location_Zone'], candidate_cluster.location_zone)[1] == 0:
        return False

    if match_Group_Size(row['Min_Group_Size'], row['Max_Group_Size'], candidate_cluster.min_group_size, candidate_cluster.max_group_size)[1] == 0:
        return False

    budget_match = match_Budget_Range(
        row["Budget_Min_USD"],
        row["Budget_Max_USD"],
        candidate_cluster.min_budget,
        candidate_cluster.max_budget,
    )
    if budget_match[0] == "NOT A MATCH":
        return False

    if match_Language_Preference(row['Language_Preference'], candidate_cluster.language_preference)[1] == 0:
        return False

    max_group_size = min(row["Max_Group_Size"], candidate_cluster.max_group_size)
    if len(candidate_cluster.get_members()) >= max_group_size:
        return False

    return True


class cluster:
    def __init__(self, row, idx, cluster_id=""):
        self.rows = [row]
        self.idx_members = [idx]
        self.cluster_id = cluster_id
        self.hard_constraints_valid = True
        self.activity_type = row['Activity_Type']
        self.availability = row['Availability']
        self.location_zone = row['Location_Zone']
        self.min_group_size = row['Min_Group_Size']
        self.max_group_size = row['Max_Group_Size']
        self.language_preference = row['Language_Preference']
        self.min_budget = row["Budget_Min_USD"]
        self.max_budget = row["Budget_Max_USD"]

    def add_member(self, member_row, member_idx):
        if not check_hard_constraints(member_row, self):
            raise ValueError("Cannot add a member who violates the cluster's hard constraints")

        budget_range = match_Budget_Range(
            self.min_budget,
            self.max_budget,
            member_row["Budget_Min_USD"],
            member_row["Budget_Max_USD"],
        )
        if budget_range[0] == "NOT A MATCH":
            raise ValueError("Cannot add a member with a non-overlapping budget range")

        self.rows.append(member_row)
        self.idx_members.append(member_idx)
        self.availability = match_Availability(self.availability, member_row['Availability'])[0]
        self.min_group_size = max(self.min_group_size, member_row['Min_Group_Size'])
        self.max_group_size = min(self.max_group_size, member_row['Max_Group_Size'])
        self.language_preference = match_Language_Preference(self.language_preference, member_row['Language_Preference'])[0]
        self.min_budget, self.max_budget, _ = budget_range

    def get_members(self):
        return self.rows
    
    def get_idx_members(self):
        return self.idx_members


def cluster_requests(df, new_cluster_threshold=-0.25):
    if tuple(df.columns) != CSV_COLUMNS:
        raise ValueError(
            "Requests must use the exact user_requests_filtered_activities_formatted.csv schema"
        )

    clusters = []
    ground_truth_clusters = {}
    unassigned_rows = []
    existing_cluster_ids = {
        str(value).strip()
        for value in df["Reference_Group_ID"]
        if pd.notna(value) and str(value).strip()
    }

    for index, row in df.iterrows():
        reference_group_id = row["Reference_Group_ID"]
        if pd.isna(reference_group_id) or not str(reference_group_id).strip():
            unassigned_rows.append((index, row))
            continue

        cluster_id = str(reference_group_id).strip()
        group_clusters = ground_truth_clusters.setdefault(cluster_id, [])
        current_cluster = next(
            (
                candidate
                for candidate in group_clusters
                if check_hard_constraints(row, candidate)
            ),
            None,
        )
        if current_cluster is not None:
            current_cluster.add_member(row, index)
        else:
            if not group_clusters:
                split_cluster_id = cluster_id
            else:
                split_number = 2
                split_cluster_id = f"{cluster_id}-SPLIT-{split_number:03d}"
                while split_cluster_id in existing_cluster_ids:
                    split_number += 1
                    split_cluster_id = f"{cluster_id}-SPLIT-{split_number:03d}"
                existing_cluster_ids.add(split_cluster_id)
            current_cluster = cluster(row, index, split_cluster_id)
            group_clusters.append(current_cluster)
            clusters.append(current_cluster)

    generated_cluster_number = 1
    for index, row in unassigned_rows:
        best_cluster = None
        best_score = new_cluster_threshold
        for current_cluster in clusters:
            score = calculate_overall_score(row, current_cluster)
            if score > best_score:
                best_cluster = current_cluster
                best_score = score

        if best_cluster is None:
            cluster_id = f"NEW-{generated_cluster_number:03d}"
            while cluster_id in existing_cluster_ids:
                generated_cluster_number += 1
                cluster_id = f"NEW-{generated_cluster_number:03d}"
            generated_cluster_number += 1
            existing_cluster_ids.add(cluster_id)
            best_cluster = cluster(row, index, cluster_id)
            clusters.append(best_cluster)
        else:
            best_cluster.add_member(row, index)

    return clusters


def assign_cluster_ids_to_reference_groups(df, clusters):
    updated_requests = df.copy()
    for current_cluster in clusters:
        for row_index in current_cluster.get_idx_members():
            updated_requests.at[row_index, "Reference_Group_ID"] = current_cluster.cluster_id
    return updated_requests


def cluster_members_frame(df, clusters):
    rows = []
    for current_cluster in clusters:
        constraint_values = (
            current_cluster.activity_type,
            current_cluster.availability,
            current_cluster.location_zone,
            current_cluster.language_preference,
        )
        if not current_cluster.hard_constraints_valid or "NOT A MATCH" in constraint_values:
            raise ValueError(
                f"Cluster {current_cluster.cluster_id!r} contains incompatible hard constraints"
            )

        for row_index, member_row in zip(
            current_cluster.get_idx_members(),
            current_cluster.get_members(),
        ):
            cluster_fields = {
                "cluster_id": current_cluster.cluster_id,
                "cluster_activity_type": current_cluster.activity_type,
                "row_index": row_index,
                "member_count": len(current_cluster.get_members()),
                "cluster_availability": current_cluster.availability,
                "cluster_location_zone": current_cluster.location_zone,
                "cluster_min_group_size": current_cluster.min_group_size,
                "cluster_max_group_size": current_cluster.max_group_size,
                "cluster_min_budget_usd": current_cluster.min_budget,
                "cluster_max_budget_usd": current_cluster.max_budget,
                "cluster_language_preference": current_cluster.language_preference,
            }
            member_fields = {
                column: member_row.get(column, "")
                for column in df.columns
            }
            member_fields["Reference_Group_ID"] = current_cluster.cluster_id
            rows.append({**cluster_fields, **member_fields})

    cluster_columns = (
        "cluster_id",
        "cluster_activity_type",
        "row_index",
        "member_count",
        "cluster_availability",
        "cluster_location_zone",
        "cluster_min_group_size",
        "cluster_max_group_size",
        "cluster_min_budget_usd",
        "cluster_max_budget_usd",
        "cluster_language_preference",
    )
    return pd.DataFrame(rows, columns=cluster_columns + CSV_COLUMNS)