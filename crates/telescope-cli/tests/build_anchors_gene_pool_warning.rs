use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{synth_record, write_synthetic_bam};

fn three_tx_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record("a0", 0, Some(100), 0, 60),
            synth_record("a1", 0, Some(100), 0, 60),
            synth_record("b0", 1, Some(100), 0, 60),
            synth_record("c0", 2, Some(100), 0, 60),
        ],
    )
}

fn scratch(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("telescope_cli_{tag}_{}.{ext}", std::process::id()))
}

fn run_build_anchors(bam: &Path, out: &Path, gtf: Option<&Path>) -> String {
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
    if let Some(gtf) = gtf {
        cmd.args(["--gtf", gtf.to_str().unwrap()]);
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
        ("the remedy", "--gtf"),
    ] {
        assert!(
            stderr.contains(needle),
            "stderr must name {what}; got:\n{stderr}"
        );
    }

    assert!(
        !stderr.contains("STRUCTURALLY UNREACHABLE"),
        "the warning must not claim L2 is unconditionally unreachable; the \
         thresholds are build-emission flags and can make it fire. got:\n{stderr}"
    );
}

#[test]
fn build_anchors_is_silent_when_the_pool_has_real_gene_structure() {
    let bam = three_tx_bam("cli_warn_neg");
    let out = scratch("warn_neg", "parquet");
    let gtf = scratch("warn_neg", "gtf");
    std::fs::write(
        &gtf,
        "chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"GENE_A\"; transcript_id \"tx0\"; gene_name \"GENE_A\";\n\
         chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"GENE_A\"; transcript_id \"tx1\"; gene_name \"GENE_A\";\n\
         chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"GENE_A\"; transcript_id \"tx2\"; gene_name \"GENE_A\";\n",
    )
    .unwrap();

    let stderr = run_build_anchors(&bam, &out, Some(&gtf));
    cleanup(&[&bam, &out, &gtf]);

    assert!(
        !stderr.contains("NO gene structure"),
        "a run WITH --gtf supplying real structure must not warn; got:\n{stderr}"
    );
    assert!(
        !stderr.contains("[telescope][WARN]"),
        "no warning of any kind may be emitted for a pool with real gene \
         structure; got:\n{stderr}"
    );
    assert!(
        stderr.contains("gtf transcripts:    3"),
        "the annotation must actually have been loaded, or this test proves \
         nothing; got:\n{stderr}"
    );
    assert!(
        stderr.contains("passed=4"),
        "the fixture must actually pass the anchor filters, or this test is \
         asserting on an empty pool; got:\n{stderr}"
    );
}
