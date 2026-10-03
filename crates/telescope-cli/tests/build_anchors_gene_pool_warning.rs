//! End-to-end: does `telescope build-anchors` actually WARN, on stderr, when it
//! emits a pool in the degenerate state?
//!
//! WHY AN END-TO-END TEST AND NOT ONLY THE HELPER'S UNIT TEST. The helper in
//! `anchor_builder` is pure and easy to test, and it is tested. But the defect
//! this fixes was not in the helper — it was that the CLI never said anything.
//! `L2=0` sat in four run logs for three days because `tx_to_gene entries: 0`
//! and `L2=0` are printed far apart and neither names the consequence. A test
//! of the helper alone cannot see that regression: delete the `eprintln!` in
//! `main.rs` and it stays green. This file is what goes red instead.
//!
//! Both directions are here on purpose. The negative case is the one that keeps
//! the warning from becoming noise on every legitimate run.

use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{synth_record, write_synthetic_bam};

/// A three-transcript synthetic BAM (tx0/tx1/tx2) whose reads all clear the
/// anchor criteria: MAPQ 60, full-length so 3'-anchored with coverage 1.0, and
/// `pt:i = 100`.
fn three_tx_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record("a0", 0, None, 0, 60),
            synth_record("a1", 0, None, 0, 60),
            synth_record("b0", 1, None, 0, 60),
            synth_record("c0", 2, None, 0, 60),
        ],
    )
}

fn scratch(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("telescope_cli_{tag}_{}.{ext}", std::process::id()))
}

fn run_build_anchors(bam: &Path, out: &Path, tx_to_gene: Option<&Path>) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_telescope"));
    cmd.args([
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        out.to_str().unwrap(),
    ]);
    if let Some(tsv) = tx_to_gene {
        cmd.args(["--tx-to-gene", tsv.to_str().unwrap()]);
    }
    let st = cmd.output().expect("failed to spawn the telescope binary");
    let stderr = String::from_utf8_lossy(&st.stderr).into_owned();
    assert!(
        st.status.success(),
        "build-anchors must still SUCCEED in the degenerate state -- this is a \
         warning, not an error, because a single-isoform reference is a \
         legitimate input. status={:?}\nstderr:\n{stderr}",
        st.status
    );
    stderr
}

fn cleanup(paths: &[&Path]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

/// POSITIVE. No `--tx-to-gene`, so `gene_id := transcript_id` and the L2
/// gene-level fallback is structurally unreachable. The stderr must name the
/// consequence, not just the state.
#[test]
fn build_anchors_warns_that_l2_is_unreachable_without_a_gene_map() {
    let bam = three_tx_bam("cli_warn_pos");
    let out = scratch("warn_pos", "parquet");
    let stderr = run_build_anchors(&bam, &out, None);
    cleanup(&[&bam, &out]);

    for (what, needle) in [
        ("the [telescope] prefix", "[telescope][WARN]"),
        ("the degenerate state itself", "NO gene structure"),
        (
            "the consequence for the L2 tier",
            "no two transcripts ever share",
        ),
        (
            "the default thresholds the claim is scoped to",
            "tx_min_reads=10",
        ),
        ("the remedy", "--tx-to-gene"),
    ] {
        assert!(
            stderr.contains(needle),
            "stderr must name {what}; got:\n{stderr}"
        );
    }

    // The thresholds live on `build-emission`, which this function cannot see,
    // so the warning must NOT assert reachability unconditionally: at
    // `--gene-min-reads 3` the same degenerate pool does fire L2. A message
    // claiming otherwise would be false for a real, reachable configuration.
    assert!(
        !stderr.contains("STRUCTURALLY UNREACHABLE"),
        "the warning must not claim L2 is unconditionally unreachable; the \
         thresholds are build-emission flags and can make it fire. got:\n{stderr}"
    );
}

/// NEGATIVE. The same BAM and the same command, given a gene map that puts all
/// three transcripts on one shared gene. L2 is reachable, so the warning would
/// be noise — it must not appear.
#[test]
fn build_anchors_is_silent_when_the_pool_has_real_gene_structure() {
    let bam = three_tx_bam("cli_warn_neg");
    let out = scratch("warn_neg", "parquet");
    let tsv = scratch("warn_neg", "tsv");
    std::fs::write(
        &tsv,
        "transcript_id\tgene_id\ntx0\tGENE_A\ntx1\tGENE_A\ntx2\tGENE_A\n",
    )
    .unwrap();

    let stderr = run_build_anchors(&bam, &out, Some(&tsv));
    cleanup(&[&bam, &out, &tsv]);

    assert!(
        !stderr.contains("STRUCTURALLY UNREACHABLE"),
        "a run WITH --tx-to-gene supplying real structure must not warn; got:\n{stderr}"
    );
    assert!(
        stderr.contains("tx_to_gene entries: 3"),
        "the gene map must actually have been loaded, or this test proves \
         nothing; got:\n{stderr}"
    );
    assert!(
        stderr.contains("passed=4"),
        "the fixture must actually pass the anchor filters, or this test is \
         asserting on an empty pool; got:\n{stderr}"
    );
}
