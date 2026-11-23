import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from pennside_backend import interact_demo, matching_func


class IntakeApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.data_path = root / "requests.csv"
        self.clusters_path = root / "clusters.csv"
        self.store_path = root / "intake_store"
        self.store_path.mkdir()
        self._write_seed_csv()

        for target, name, replacement in (
            (interact_demo, "DATA_FILE", self.data_path),
            (interact_demo, "CLUSTERS_FILE", self.clusters_path),
            (interact_demo, "STORE", self.store_path),
            (matching_func, "calculate_compatibility", lambda _left, _right: 0.8),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.client = TestClient(interact_demo.app)

    def _write_seed_csv(self):
        seed = {
            "email": "seed@example.edu",
            "Activity_Type": "Study session",
            "Availability": "2026/11/24 18:00-20:00",
            "Location_Zone": "Penn Libraries",
            "Budget_Min_USD": 0,
            "Budget_Max_USD": 20,
            "Min_Group_Size": 2,
            "Max_Group_Size": 4,
            "Language_Preference": "Both",
            "Social_Energy": "Conversational",
            "Soft_Preferences": "Study together",
            "Reference_Group_ID": "G011",
        }
        with self.data_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=matching_func.CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(seed)

    @staticmethod
    def _record(email):
        return {
            "email": email,
            "activity": "Study session",
            "hardConstraints": (
                "Available: 2026-11-24 at 19:00 – 2026-11-24 at 21:00 "
                "(Eastern Time); location: Penn Libraries; budget: $5–$15/person; "
                "total party: 2–4, including organizer"
            ),
            "location": "Penn Libraries",
            "budgetMin": 5,
            "budgetMax": 15,
            "minParty": 2,
            "maxParty": 4,
            "language": "english",
            "socialEnergy": "conversational",
            "softPreferences": "Study together",
        }

    def test_existing_user_loads_ground_truth_and_new_plans_refresh_cluster(self):
        existing = self.client.get(
            "/api/plans",
            params={"email": "seed@example.edu"},
        )
        self.assertEqual(existing.status_code, 200)
        self.assertEqual(existing.json()["plans"][0]["cluster"]["id"], "G011")

        first = self.client.post(
            "/api/intake",
            json={"record": self._record("new@example.edu")},
        )
        self.assertEqual(first.status_code, 200, first.text)
        first_plan = first.json()["plan"]
        self.assertEqual(first_plan["cluster"]["id"], "G011")
        self.assertEqual(first_plan["cluster"]["memberCount"], 2)
        self.assertEqual(
            {member["email"] for member in first_plan["cluster"]["members"]},
            {"seed@example.edu", "new@example.edu"},
        )

        second = self.client.post(
            "/api/intake",
            json={"record": self._record("new@example.edu")},
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(len(second.json()["plans"]), 2)

        latest = self.client.get(
            "/api/plans",
            params={"email": "new@example.edu"},
        )
        self.assertEqual(len(latest.json()["plans"]), 2)
        self.assertTrue(self.clusters_path.exists())
        with self.clusters_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["Reference_Group_ID"] for row in rows if row["email"] == "new@example.edu"},
            {"G011", "NEW-001"},
        )
        with self.data_path.open(encoding="utf-8-sig", newline="") as file:
            requests = list(csv.DictReader(file))
        self.assertEqual(
            {row["Reference_Group_ID"] for row in requests if row["email"] == "new@example.edu"},
            {"G011", "NEW-001"},
        )

    def test_cross_day_availability_is_rejected_without_saving(self):
        record = self._record("new@example.edu")
        record["hardConstraints"] = (
            "Available: 2026-11-24 at 23:00 – 2026-11-25 at 01:00 (Eastern Time)"
        )
        response = self.client.post("/api/intake", json={"record": record})

        self.assertEqual(response.status_code, 422)
        with self.data_path.open(encoding="utf-8-sig", newline="") as file:
            self.assertEqual(len(list(csv.DictReader(file))), 1)

    def test_seed_csv_user_can_skip_intake_and_view_ground_truth_cluster(self):
        source_csv = Path(__file__).with_name("user_requests_filtered_activities_formatted.csv")
        with (
            patch.object(interact_demo, "DATA_FILE", source_csv),
            patch.object(matching_func, "calculate_compatibility", return_value=0.5),
        ):
            response = self.client.get(
                "/api/plans",
                params={"email": "user-6e502cbfea5a@example.com"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()["plans"][0]
        self.assertEqual(plan["cluster"]["id"], "G011")
        self.assertEqual(plan["cluster"]["memberCount"], 5)
        self.assertTrue(any(member["isCurrentUser"] for member in plan["cluster"]["members"]))
        self.assertTrue(all(member["email"] for member in plan["cluster"]["members"]))
        self.assertIn(
            "user-6e502cbfea5a@example.com",
            {member["email"] for member in plan["cluster"]["members"]},
        )

    def test_legacy_invalid_cluster_snapshot_is_rebuilt(self):
        requests = matching_func.open_file(self.data_path)
        seeded_clusters = matching_func.cluster_requests(requests)
        legacy_snapshot = matching_func.cluster_members_frame(requests, seeded_clusters)
        legacy_snapshot["cluster_language_preference"] = "NOT A MATCH"
        legacy_snapshot.drop(columns=["cluster_activity_type"]).to_csv(
            self.clusters_path,
            index=False,
            encoding="utf-8-sig",
        )

        response = self.client.get(
            "/api/plans",
            params={"email": "seed@example.edu"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["plans"][0]["cluster"]["language"], "Both")
        with self.clusters_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertIn("cluster_activity_type", rows[0])
        self.assertTrue(
            all(row["cluster_language_preference"] != "NOT A MATCH" for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
