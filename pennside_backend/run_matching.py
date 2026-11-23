import csv
from pathlib import Path

import matching_func


if __name__ == "__main__":
    backend_dir = Path(__file__).resolve().parent
    input_path = backend_dir / "user_requests_filtered_activities_formatted.csv"
    file_df = matching_func.open_file(input_path)
    clusters = matching_func.cluster_requests(file_df)
    file_df = matching_func.assign_cluster_ids_to_reference_groups(file_df, clusters)
    cluster_rows = matching_func.cluster_members_frame(file_df, clusters)

    clusters_path = backend_dir / "clusters.csv"
    cluster_rows.to_csv(clusters_path, index=False, encoding="utf-8-sig")

    summary_path = backend_dir / "clusters_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "cluster_id", "num_members",
            "activity_type", "availability", "location_zone",
            "min_group_size", "max_group_size",
            "min_budget_usd", "max_budget_usd",
            "language_preference", "members_soft_preference",
        ])
        for current_cluster in clusters:
            soft_preferences = [
                str(row.get("Soft_Preferences", ""))
                for row in current_cluster.get_members()
            ]
            writer.writerow([
                current_cluster.cluster_id,
                len(current_cluster.get_members()),
                current_cluster.activity_type,
                current_cluster.availability,
                current_cluster.location_zone,
                current_cluster.min_group_size,
                current_cluster.max_group_size,
                current_cluster.min_budget,
                current_cluster.max_budget,
                current_cluster.language_preference,
                " | ".join(soft_preferences),
            ])

    member_path = backend_dir / "cluster_members.csv"
    member_columns = ["cluster_id", "row_index"] + list(file_df.columns)
    cluster_rows[member_columns].to_csv(
        member_path,
        index=False,
        encoding="utf-8-sig",
    )
    input_temp = input_path.with_name(input_path.name + ".tmp")
    file_df.to_csv(input_temp, index=False, encoding="utf-8-sig")
    input_temp.replace(input_path)

    sizes = [len(current_cluster.get_members()) for current_cluster in clusters]
    print(f"共 {len(clusters)} 个 cluster，总人数 {sum(sizes)}")
    print(f"cluster 大小分布: {sorted(sizes, reverse=True)}")
    print(f"已输出: {clusters_path.name}, {summary_path.name}, {member_path.name}")
