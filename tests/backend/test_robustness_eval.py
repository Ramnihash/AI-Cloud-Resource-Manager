from experiments import robustness_eval


def test_robustness_eval_runs_and_finds_no_server_errors(tmp_path):
    results = robustness_eval.run(str(tmp_path), repeat_steps=20, scale_repeats=2)
    failures = results["failures"]
    assert failures["graceful"].all()
    malformed = results["malformed"]
    assert not malformed["is_5xx"].any()
    assert malformed["error_body_ok"].all()
    assert (results["stress"]["status_5xx"] == 0).all()
    assert {"robustness_forced_failures.csv", "stress_latency.csv", "stress_malformed.csv"} <= {
        p.name for p in tmp_path.iterdir()
    }
