import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pennside_backend import matching_func


def make_request(**overrides):
    row = {
        "email": "student@example.edu",
        "Activity_Type": "Study session",
        "Availability": "2026/11/24 18:00-20:00",
        "Location_Zone": "Penn Libraries",
        "Budget_Min_USD": 0,
        "Budget_Max_USD": 20,
        "Min_Group_Size": 2,
        "Max_Group_Size": 4,
        "Language_Preference": "Both",
        "Social_Energy": "Quiet",
        "Soft_Preferences": "Study together",
        "Reference_Group_ID": "G001",
    }
    row.update(overrides)
    return row


class MatchingFunctionTests(unittest.TestCase):
    def test_availability_intersects_using_csv_date_format(self):
        self.assertEqual(
            matching_func.match_Availability(
                "2026/11/24 18:00-20:00",
                "2026/11/24 19:00-21:00",
            ),
            ("2026/11/24 19:00-20:00", 1),
        )
        self.assertEqual(
            matching_func.match_Availability(
                "2026/11/24 18:00-20:00",
                "2026/11/25 19:00-21:00",
            ),
            ("NOT A MATCH", 0),
        )
        self.assertEqual(
            matching_func.match_Availability(
                "not a date",
                "2026/11/24 19:00-21:00",
            ),
            ("NOT A MATCH", 0),
        )

    def test_budget_ranges_must_overlap_and_keep_their_intersection(self):
        self.assertEqual(
            matching_func.match_Budget_Range(0, 15, 10, 20),
            (10, 15, 1),
        )
        self.assertEqual(
            matching_func.match_Budget_Range(0, 0, 0, 0),
            (0, 0, 1),
        )
        self.assertEqual(
            matching_func.match_Budget_Range(0, 10, 15, 20),
            ("NOT A MATCH", 0),
        )

    def test_language_preferences_use_the_csv_options(self):
        self.assertEqual(
            matching_func.match_Language_Preference("Both", "English"),
            ("English", 1),
        )
        self.assertEqual(
            matching_func.match_Language_Preference("Both", "Chinese"),
            ("Chinese", 1),
        )
        self.assertEqual(
            matching_func.match_Language_Preference("English", "Chinese"),
            ("NOT A MATCH", 0),
        )
        self.assertEqual(
            matching_func.match_Language_Preference("English", "English"),
            ("English", 1),
        )

    def test_cluster_updates_intersected_availability_budget_and_language(self):
        current_cluster = matching_func.cluster(make_request(), 0)
        next_request = make_request(
            email="second@example.edu",
            Availability="2026/11/24 19:00-21:00",
            Budget_Min_USD=5,
            Budget_Max_USD=15,
            Language_Preference="English",
        )

        current_cluster.add_member(next_request, 1)

        self.assertEqual(current_cluster.availability, "2026/11/24 19:00-20:00")
        self.assertEqual((current_cluster.min_budget, current_cluster.max_budget), (5, 15))
        self.assertEqual(current_cluster.language_preference, "English")
        self.assertEqual(current_cluster.get_members()[1]["email"], "second@example.edu")

    def test_nli_compatibility_combines_with_group_size_and_budget_is_hard(self):
        current_cluster = matching_func.cluster(make_request(), 0)
        matching_request = make_request(
            email="new@example.edu",
            Budget_Min_USD=10,
            Budget_Max_USD=30,
            Language_Preference="English",
        )
        with patch.object(matching_func, "calculate_compatibility", return_value=0.65):
            self.assertAlmostEqual(
                matching_func.calculate_overall_score(matching_request, current_cluster),
                0.455,
            )

        over_budget_request = make_request(Budget_Min_USD=21, Budget_Max_USD=30)
        with patch.object(matching_func, "calculate_compatibility") as compatibility:
            self.assertEqual(
                matching_func.calculate_overall_score(over_budget_request, current_cluster),
                float("-inf"),
            )
            compatibility.assert_not_called()

    def test_duration_preferences_use_numeric_overlap(self):
        self.assertAlmostEqual(
            matching_func.calculate_compatibility(
                "Less than 30 minutes.",
                "Less than 20 minutes.",
            ),
            2 / 3,
        )
        self.assertAlmostEqual(
            matching_func.calculate_compatibility(
                "Less than 30 minutes.",
                "Over 1 hour.",
            ),
            -0.5,
        )
        self.assertAlmostEqual(
            matching_func.calculate_compatibility("少于30分钟", "少于20分钟"),
            2 / 3,
        )

    def test_nested_duration_preferences_can_fill_minimum_group_size(self):
        requests = pd.DataFrame(
            [
                make_request(
                    email="a@b.com",
                    Soft_Preferences="Less than 30 minutes.",
                    Reference_Group_ID="",
                ),
                make_request(
                    email="iii@uuuy.com",
                    Soft_Preferences="Less than 20 minutes.",
                    Reference_Group_ID="",
                ),
            ],
            columns=matching_func.CSV_COLUMNS,
        )

        def compatibility(text1, text2):
            duration_score = matching_func._duration_compatibility(text1, text2)
            if duration_score is not None:
                return duration_score
            return 1.0 if text1 == text2 else 0.0

        with patch.object(
            matching_func,
            "calculate_compatibility",
            side_effect=compatibility,
        ):
            clusters = matching_func.cluster_requests(requests)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].get_members()), 2)

    def test_conflicting_duration_does_not_join_a_minimum_size_cluster(self):
        first = make_request(
            email="a@b.com",
            Soft_Preferences="Less than 30 minutes.",
        )
        second = make_request(
            email="iii@uuuy.com",
            Soft_Preferences="Less than 20 minutes.",
        )
        conflicting = make_request(
            email="c@d.com",
            Soft_Preferences="Over 1 hour.",
        )
        current_cluster = matching_func.cluster(first, 0)
        current_cluster.add_member(second, 1)

        def compatibility(text1, text2):
            duration_score = matching_func._duration_compatibility(text1, text2)
            if duration_score is not None:
                return duration_score
            return 1.0 if text1 == text2 else 0.0

        with patch.object(
            matching_func,
            "calculate_compatibility",
            side_effect=compatibility,
        ):
            score = matching_func.calculate_overall_score(conflicting, current_cluster)

        self.assertLess(score, -0.25)

    def test_nli_probabilities_map_contradiction_neutral_and_entailment_to_signed_scores(self):
        cases = (
            (
                "我更喜欢安静、低社交能量的聚会。",
                "我想参加热闹、高社交能量的派对。",
                {"entailment": 0.000188, "neutral": 0.000166, "contradiction": 0.999646},
                {"entailment": 0.000125, "neutral": 0.000174, "contradiction": 0.999701},
                -0.9996,
            ),
            (
                "我更喜欢安静、低社交能量的聚会。",
                "我希望认识不同专业的同学。",
                {"entailment": 0.000360, "neutral": 0.999404, "contradiction": 0.000236},
                {"entailment": 0.000107, "neutral": 0.000244, "contradiction": 0.999649},
                0.0,
            ),
            (
                "我喜欢平静、安静的活动。",
                "我偏好安静、低社交能量的活动。",
                {"entailment": 0.998384, "neutral": 0.001537, "contradiction": 0.000079},
                {"entailment": 0.999057, "neutral": 0.000807, "contradiction": 0.000136},
                0.9984,
            ),
        )
        for text1, text2, forward, reverse, expected_score in cases:
            with patch.object(
                matching_func,
                "_predict_nli_probabilities",
                side_effect=[[forward, reverse], [reverse, forward]],
            ):
                score = matching_func.calculate_compatibility(text1, text2)
                reversed_score = matching_func.calculate_compatibility(text2, text1)

            self.assertAlmostEqual(score, expected_score, places=4)
            self.assertAlmostEqual(score, reversed_score, places=7)

    def test_one_way_entailment_is_compatible_but_entailment_and_contradiction_conflict(self):
        neutral = {"entailment": 0.01, "neutral": 0.98, "contradiction": 0.01}
        entailment = {"entailment": 0.97, "neutral": 0.02, "contradiction": 0.01}
        contradiction = {
            "entailment": 0.01,
            "neutral": 0.01,
            "contradiction": 0.98,
        }

        self.assertAlmostEqual(
            matching_func._score_bidirectional_nli(neutral, entailment),
            0.97,
        )
        self.assertAlmostEqual(
            matching_func._score_bidirectional_nli(entailment, contradiction),
            -0.98,
        )
        self.assertEqual(
            matching_func._score_bidirectional_nli(neutral, contradiction),
            0.0,
        )

    def test_negative_soft_score_is_combined_instead_of_directly_rejected(self):
        current_cluster = matching_func.cluster(make_request(), 0)
        request = make_request(
            email="new@example.edu",
            Reference_Group_ID="",
        )

        with patch.object(matching_func, "calculate_compatibility", return_value=-0.85):
            self.assertEqual(
                matching_func.calculate_overall_score(request, current_cluster),
                -0.595,
            )

    def test_negative_score_can_join_an_underfilled_cluster(self):
        requests = pd.DataFrame(
            [
                make_request(),
                make_request(
                    email="new@example.edu",
                    Reference_Group_ID="",
                ),
            ],
            columns=matching_func.CSV_COLUMNS,
        )

        with patch.object(matching_func, "calculate_compatibility", return_value=-0.2):
            clusters = matching_func.cluster_requests(requests)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].get_members()), 2)

    def test_reaching_minimum_group_size_lowers_the_composite_score(self):
        current_cluster = matching_func.cluster(make_request(), 0)
        request = make_request(email="third@example.edu")
        with patch.object(matching_func, "calculate_compatibility", return_value=-0.2):
            underfilled_score = matching_func.calculate_overall_score(
                request,
                current_cluster,
            )
            current_cluster.add_member(
                make_request(email="second@example.edu"),
                1,
            )
            complete_score = matching_func.calculate_overall_score(
                request,
                current_cluster,
            )

        self.assertAlmostEqual(underfilled_score, -0.14)
        self.assertAlmostEqual(complete_score, -0.44)

    def test_csv_loader_accepts_the_exact_input_schema_and_keeps_emails(self):
        csv_path = Path(__file__).with_name("user_requests_filtered_activities_formatted.csv")
        requests = matching_func.open_file(csv_path)

        self.assertEqual(tuple(requests.columns), matching_func.CSV_COLUMNS)
        self.assertGreater(len(requests), 0)
        self.assertTrue(requests["email"].str.contains("@").all())

    def test_reference_groups_seed_clusters_and_unlabeled_requests_join_by_score(self):
        seed = make_request(Reference_Group_ID="G011")
        new_request = make_request(
            email="new@example.edu",
            Language_Preference="English",
            Reference_Group_ID="",
        )
        requests = pd.DataFrame([seed, new_request], columns=matching_func.CSV_COLUMNS)

        with patch.object(matching_func, "calculate_compatibility", return_value=0.8):
            clusters = matching_func.cluster_requests(requests)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].cluster_id, "G011")
        self.assertEqual(len(clusters[0].get_members()), 2)
        snapshot = matching_func.cluster_members_frame(requests, clusters)
        self.assertEqual(snapshot["cluster_id"].tolist(), ["G011", "G011"])
        self.assertEqual(snapshot["email"].tolist(), ["student@example.edu", "new@example.edu"])

    def test_ground_truth_members_split_when_hard_constraints_conflict(self):
        requests = pd.DataFrame(
            [
                make_request(
                    email="chinese@example.edu",
                    Language_Preference="Chinese",
                    Reference_Group_ID="G055",
                ),
                make_request(
                    email="english@example.edu",
                    Language_Preference="English",
                    Reference_Group_ID="G055",
                ),
                make_request(
                    email="both@example.edu",
                    Language_Preference="Both",
                    Reference_Group_ID="G055",
                ),
            ],
            columns=matching_func.CSV_COLUMNS,
        )

        clusters = matching_func.cluster_requests(requests)
        by_id = {current_cluster.cluster_id: current_cluster for current_cluster in clusters}

        self.assertEqual(set(by_id), {"G055", "G055-SPLIT-002"})
        self.assertEqual(
            {row["email"] for row in by_id["G055"].get_members()},
            {"chinese@example.edu", "both@example.edu"},
        )
        self.assertEqual(by_id["G055"].language_preference, "Chinese")
        self.assertEqual(by_id["G055-SPLIT-002"].language_preference, "English")
        self.assertTrue(all(current_cluster.hard_constraints_valid for current_cluster in clusters))
        snapshot = matching_func.cluster_members_frame(requests, clusters)
        self.assertEqual(
            set(snapshot["Reference_Group_ID"]),
            {"G055", "G055-SPLIT-002"},
        )
        synced_requests = matching_func.assign_cluster_ids_to_reference_groups(
            requests,
            clusters,
        )
        self.assertEqual(
            set(synced_requests["Reference_Group_ID"]),
            {"G055", "G055-SPLIT-002"},
        )

    def test_synced_reference_groups_remain_hard_constraint_compatible(self):
        csv_path = Path(__file__).with_name("user_requests_filtered_activities_formatted.csv")
        requests = matching_func.open_file(csv_path)

        with patch.object(matching_func, "calculate_compatibility", return_value=0.5):
            clusters = matching_func.cluster_requests(requests)

        by_id = {current_cluster.cluster_id: current_cluster for current_cluster in clusters}
        self.assertEqual(len(by_id["G011"].get_members()), 5)
        g055_clusters = [
            current_cluster
            for current_cluster in clusters
            if current_cluster.cluster_id == "G055"
            or current_cluster.cluster_id.startswith("G055-SPLIT-")
        ]
        self.assertGreaterEqual(len(g055_clusters), 2)
        self.assertEqual(
            sum(len(current_cluster.get_members()) for current_cluster in g055_clusters),
            6,
        )
        for current_cluster in clusters:
            self.assertTrue(current_cluster.hard_constraints_valid)
            self.assertNotIn(
                "NOT A MATCH",
                (
                    current_cluster.activity_type,
                    current_cluster.availability,
                    current_cluster.location_zone,
                    current_cluster.language_preference,
                ),
            )
        snapshot = matching_func.cluster_members_frame(requests, clusters)
        aggregate_columns = [
            column
            for column in snapshot.columns
            if column.startswith("cluster_") and column != "cluster_id"
        ]
        self.assertFalse(
            snapshot[aggregate_columns].astype(str).eq("NOT A MATCH").any().any()
        )
        self.assertEqual(
            sum(len(current_cluster.get_members()) for current_cluster in clusters),
            len(requests),
        )


if __name__ == "__main__":
    unittest.main()
