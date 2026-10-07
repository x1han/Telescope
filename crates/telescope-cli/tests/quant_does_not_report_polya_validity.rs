//! End-to-end: `telescope quant` must NOT report a poly(A) validity number.
//!
//! WHY THIS TEST EXISTS. `TelEmStats` carried two counters —
//! `n_polya_zero_power` and `n_polya_zero_power_resolved` — which `quant`
//! printed under the header `poly(A) discriminative power:`. Both counters were
//! measured to be uninformative about deconvolution accuracy. The counter
//! FIRES when a read's candidates share one identical `(mu, sigma2)`, so a low
//! value means non-degenerate and NOT discriminative; on HC2 it read
//! `zero-power_all: 886 of 4,380,726 ambiguous (0.02%)`. The trap was its
//! complement — an earlier round of notes read 0.02% as "poly(A) discriminates
//! on 99.98% of contested reads", which is false, since the poly(A) term's
//! per-read SNR is 0.255. The expert ruling was to remove the counters from user-facing
//! output while KEEPING them in `TelEmStats` as internal solver monitoring, so
//! this asserts on the OUTPUT, never on the fields.
//!
//! WHY AN END-TOEND TEST. The defect is not in the counter, it is in the fact
//! that the counter is printed. A unit test of the predicate cannot see that:
//! delete the `eprintln!` block in `main.rs` and `em.rs`'s three tests stay
//! green, because they assert on `TelEmStats` fields. Only running the binary
//! and reading its stderr can see this regression.
//!
//! WHY THE CHAIN. `quant` needs `--emission <parquet>`, and the helper that
//! writes an emission table (`write_test_emission`) lives inside
//! `#[cfg(test)] mod tests` in `em.rs` and is therefore invisible to an
//! integration test. So this drives the real three-step CLI chain:
//! `build-anchors` -> `build-emission` -> `quant`. This is also the first test
//! in the repo to run `quant` at all.
//!
//! WHY THE POSITIVE PIN. A negative assertion on stderr is a tautology unless a
//! sibling assertion proves the capture is live. `emission model coverage` was
//! the first telemetry line of the block the removed lines sat in, and it
//! prints only after `solve_and_write_profile` returned `Ok`, so it proves the
//! run reached the print zone rather than merely having started. Without it, a
//! fixture that bailed early would leave stderr empty and both negative
//! assertions below would pass for the wrong reason.
//!
//! Before this test existed, `quant`'s stderr contained `zero-power_all`, so it
//! failed on first run. That red is the evidence the fixture reaches the print
//! block: had the fixture died early, it would have been green and proved
//! nothing.

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

    // Defaults are 10/30/30; a 4-anchor fixture cannot clear them, and the
    // thresholds are the point of the Universal-Mode fallback, so lower them
    // rather than inflate the fixture.
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
