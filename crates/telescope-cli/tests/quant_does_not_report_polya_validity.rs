//! End-to-end: `telescope quant` must NOT report a poly(A) validity number.
//!
//! `TelEmStats` carries two poly(A) zero-power counters, which measure how
//! often a read's candidates share one identical `(mu, sigma2)`. A LOW value
//! means the term is degenerate, not discriminative, so printing it under a
//! "poly(A) discriminative power" header invites a validity claim it cannot
//! support. The counters stay in `TelEmStats` as internal solver monitoring;
//! only the OUTPUT is asserted here.
//!
//! A unit test of the predicate cannot see this: delete the `eprintln!`
//! block in `main.rs` and the `em.rs` tests stay green, so this drives the
//! real CLI chain `build-anchors` -> `build-emission` -> `quant`.
//!
//! The negative assertions are only meaningful because a positive one proves
//! the capture is live: `emission model coverage` prints only after the solve returns `Ok`.

use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{synth_record, write_synthetic_bam};

/// tx0/tx1/tx2 plus one read aligned to BOTH tx0 and tx1, so the EM branch is
/// entered rather than every read being single-candidate.
fn fixture_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record("a0", 0, Some(900), 0, 60),
            synth_record("a1", 0, Some(900), 0, 60),
            synth_record("b0", 1, Some(900), 0, 60),
            synth_record("c0", 2, Some(900), 0, 60),
            // The contested read: two records, one name.
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

/// Run the real chain and return quant's stderr.
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

    // Defaults are 10/30/30; a 4-anchor fixture cannot clear them, so lower
    // them rather than inflate the fixture.
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

/// Pins that `quant` reports no poly(A) validity number to the user.
#[test]
fn quant_does_not_report_a_polya_validity_number() {
    let stderr = quant_stderr("nopya_validity");

    // POSITIVE: proves the capture is live and reached the print zone. This is
    // the string that makes the two negative assertions below non-tautological.
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
