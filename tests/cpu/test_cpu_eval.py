from experiments import cpu_eval


def test_cpu_eval_writes_outputs(tmp_path):
    results = cpu_eval.run(n=2, out=str(tmp_path))
    names = {p.name for p in tmp_path.iterdir()}
    assert {"cpu_runs.csv", "cpu_summary.csv", "cpu_paired.csv", "cpu_selection.csv", "cpu_waiting.png"} <= names
    runs = results["runs"]
    assert set(runs.scheduler) == {"AI-selected", "FCFS", "SJF", "Priority", "Round Robin"}
    assert len(runs) == 2 * 4 * 5
    # the AI-selected run is one of the fixed schedulers, so it must match it exactly
    ai = runs[runs.scheduler == "AI-selected"].set_index(["workload_type", "seed"])
    for key, row in ai.iterrows():
        twin = runs[(runs.scheduler == row["effective"]) & (runs.workload_type == key[0]) & (runs.seed == key[1])]
        assert abs(twin.iloc[0]["avg_waiting_time"] - row["avg_waiting_time"]) < 1e-9
