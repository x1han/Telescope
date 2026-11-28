use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{synth_record, write_synthetic_bam_named};

fn anchor_bam(tag: &str) -> PathBuf {
    write_synthetic_bam_named(
        &format!("{tag}_pool"),
        &["tx0", "tx1"],
        vec![
            synth_record("a0", 0, Some(100), 0, 60),
            synth_record("a1", 0, Some(100), 0, 60),
            synth_record("b0", 1, Some(100), 0, 60),
        ],
    )
}

fn duplicated_universe_bam(tag: &str) -> PathBuf {
    write_synthetic_bam_named(&format!("{tag}_dup"), &["tx0", "tx1", "tx1"], Vec::new())
}

fn scratch(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("telescope_cli_{tag}_{}.{ext}", std::process::id()))
}

fn build_anchor_pool(bam: &Path, out: &Path) {
    let st = Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args([
            "build-anchors",
            "--bam",
            bam.to_str().unwrap(),
            "--sample-id",
            "S1",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn the telescope binary");
    assert!(
        st.status.success(),
        "build-anchors failed: status={:?}\nstderr:\n{}",
        st.status,
        String::from_utf8_lossy(&st.stderr)
    );
}

fn build_emission(pool: &Path, universe_bam: &Path, out: &Path) -> (bool, String) {
    let st = Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args([
            "build-emission",
            "--anchor-pool",
            pool.to_str().unwrap(),
            "--reference-bam",
            universe_bam.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn the telescope binary");
    (
        st.status.success(),
        String::from_utf8_lossy(&st.stderr).into_owned(),
    )
}

fn cleanup(paths: &[&Path]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn a_duplicated_universe_entry_is_collapsed_by_the_header_reader() {
    let dup_bam = duplicated_universe_bam("consv");
    let names = telescope_core::emission_model::read_reference_universe(&dup_bam)
        .expect("the duplicated-SQ header must still parse");
    cleanup(&[&dup_bam]);

    assert_eq!(
        names,
        vec!["tx0".to_string(), "tx1".to_string()],
        "read_reference_universe returned {names:?} for a header that declares \
         `tx0`, `tx1`, `tx1`. If the duplicate now survives, the CLI \
         conservation guard has become REACHABLE -- it is the only input that \
         breaks that law -- and it needs a negative control of its own before \
         it can be called load-bearing."
    );
}

#[test]
fn a_well_formed_universe_succeeds_and_prints_the_w2_split() {
    let pool_bam = anchor_bam("consv_ok");
    let ok_bam = write_synthetic_bam_named("consv_ok_ref", &["tx0", "tx1"], Vec::new());
    let pool = scratch("consv_ok_pool", "parquet");
    let emission = scratch("consv_ok_emission", "parquet");
    let _ = std::fs::remove_file(&pool);
    let _ = std::fs::remove_file(&emission);
    build_anchor_pool(&pool_bam, &pool);

    let (ok, stderr) = build_emission(&pool, &ok_bam, &emission);

    cleanup(&[
        &pool_bam,
        &ok_bam,
        &pool,
        &emission,
        &emission.with_extension("provenance.parquet"),
    ]);

    assert!(
        ok,
        "CONTROL: the same fixture with a well-formed universe must succeed. \
         If it fails, nothing in this file is measuring the conservation guard. \
         stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("emission conservation violated"),
        "a well-formed universe must not trip the guard. stderr:\n{stderr}"
    );

    assert!(
        stderr.contains("n_imputed_rows:"),
        "the completeness block must still print. stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("(padded: 0, suppressed: 0)"),
        "with no transcript evidence and nothing to pad, both detail counts \
         are 0 and both must be PRINTED -- the split is the point of the two \
         detail fields, and a \
         line that only appeared when the counts were non-zero would hide it. \
         stderr:\n{stderr}"
    );
}

#[test]
fn a_well_formed_universe_still_succeeds() {
    let pool_bam = anchor_bam("consv_ok2");
    let ok_bam = write_synthetic_bam_named("consv_ok2_ref", &["tx0", "tx1"], Vec::new());
    let pool = scratch("consv_ok2_pool", "parquet");
    let emission = scratch("consv_ok2_emission", "parquet");
    let _ = std::fs::remove_file(&pool);
    let _ = std::fs::remove_file(&emission);
    build_anchor_pool(&pool_bam, &pool);

    let (ok, stderr) = build_emission(&pool, &ok_bam, &emission);

    cleanup(&[
        &pool_bam,
        &ok_bam,
        &pool,
        &emission,
        &emission.with_extension("provenance.parquet"),
    ]);

    assert!(
        ok,
        "CONTROL: the same fixture with NO duplicated @SQ must succeed. If it \
         fails, the negative test above is not testing the guard -- it is \
         testing whatever else broke. stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("emission conservation violated"),
        "a well-formed universe must not trip the guard. stderr:\n{stderr}"
    );

    assert!(
        stderr.contains("n_imputed_rows:"),
        "the completeness block must still print. stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("(padded: 0, suppressed: 0)"),
        "with no transcript evidence and nothing to pad, both detail counts \
         are 0 and both must be PRINTED -- the split is the point of the two \
         detail fields, and a \
         line that only appears when the counts are non-zero would hide it. \
         stderr:\n{stderr}"
    );
}
