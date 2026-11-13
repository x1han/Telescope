use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{synth_record, write_synthetic_bam};

fn fixture_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record("a0", 0, Some(900), 0, 60),
            synth_record("a1", 0, Some(900), 0, 60),
            synth_record("b0", 1, Some(900), 0, 60),
            synth_record("c0", 2, Some(900), 0, 60),
            synth_record("amb0", 0, Some(900), 0, 60),
            synth_record("amb0", 1, Some(850), 1, 60),
        ],
    )
}

fn scratch(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("telescope_cli_{tag}_{}.{ext}", std::process::id()))
}

fn run(args: &[&str]) -> (bool, String) {
    let st = Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args(args)
        .output()
        .expect("failed to spawn the telescope binary");
    (
        st.status.success(),
        String::from_utf8_lossy(&st.stderr).into_owned(),
    )
}

fn quant_stderr(tag: &str) -> String {
    let bam = fixture_bam(tag);
    let pool = scratch(&format!("{tag}_pool"), "parquet");
    let emission = scratch(&format!("{tag}_emis"), "parquet");
    let profile = scratch(&format!("{tag}_prof"), "parquet");

    let (ok, err) = run(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        pool.to_str().unwrap(),
    ]);
    assert!(ok, "build-anchors failed:\n{err}");

    let (ok, err) = run(&[
        "build-emission",
        "--anchor-pool",
        pool.to_str().unwrap(),
        "--output",
        emission.to_str().unwrap(),
        "--tx-min-reads",
        "1",
        "--gene-min-reads",
        "1",
    ]);
    assert!(ok, "build-emission failed:\n{err}");

    let (ok, err) = run(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--emission",
        emission.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        profile.to_str().unwrap(),
    ]);
    cleanup(&[&bam, &pool, &emission, &profile]);
    assert!(
        ok,
        "quant must succeed for this test to mean anything -- if it bailed early \
         its stderr is empty and the negative assertions below would pass for \
         the wrong reason.\nstderr:\n{err}"
    );
    err
}

fn cleanup(paths: &[&Path]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn quant_reports_no_poly_a_validity_number() {
    let stderr = quant_stderr("nopya_validity");

    assert!(
        stderr.contains("[telescope]    emission model coverage"),
        "quant did not reach the telemetry block, so the absence assertions \
         below would prove nothing. stderr:\n{stderr}"
    );

    for (what, needle) in [
        ("the zero-power counters", "zero-power"),
        ("the banned framing itself", "discriminative power"),
    ] {
        assert!(
            !stderr.contains(needle),
            "quant must not report {what} to the user; the counter only ever \
             measured M-step personalisation, and under a 'discriminative \
             power' header it reads as a validity claim it cannot support. \
             stderr:\n{stderr}"
        );
    }
}
