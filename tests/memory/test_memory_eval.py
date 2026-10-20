from experiments import memory_eval


def test_memory_eval_writes_outputs(tmp_path):
    results = memory_eval.run(str(tmp_path), n_strings=2, n_per_type=3)
    names = {p.name for p in tmp_path.iterdir()}
    assert {
        "memory_replacement_summary.csv",
        "memory_replacement_paired.csv",
        "memory_prediction_accuracy.csv",
        "memory_leak_metrics.csv",
        "memory_pressure_metrics.csv",
        "memory_page_faults.png",
    } <= names
    runs = results["runs"]
    # Optimal is a lower bound on demand faults for the classical policies
    for (_, _, _), g in runs.groupby(["pattern", "frames", "seed"]):
        faults = g.set_index("policy")["page_faults"]
        assert faults["Optimal"] <= faults["LRU"]
        assert faults["Optimal"] <= faults["FIFO"]
    seq = runs[(runs.pattern == "sequential") & (runs.policy == "Predictive-prefetch")]
    assert (seq.page_faults <= 2).all()
