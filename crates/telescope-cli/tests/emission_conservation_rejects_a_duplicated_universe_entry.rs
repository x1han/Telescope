//! The emission-conservation `ensure!` in `main.rs`, and the measurement that
//! says it CANNOT fire from here.
//!
//! WHY A PROCESS-LEVEL TEST. The guard is `anyhow::ensure!` inside `fn main()`:
//! it returns `Err`, not a panic, so the process exits non-zero with the
//! message on **stderr** — which a `#[should_panic]` unit test cannot see.
//!
//! THE LAW IS AN IDENTITY: both sides count `anchored UNION universe`, so for
//! any two SETS it holds; only a universe naming a transcript twice breaks it.
//!
//! MEASURED OUTCOME: the guard CANNOT fire from the CLI, because the header
//! reader returns a MAP that collapses duplicate `@SQ` names. This file pins
//! that precondition: if the reader stops collapsing duplicates the guard
//! becomes reachable and this file goes red; the reachable half of that
//! failure mode is pinned in `emission_model.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{synth_record, write_synthetic_bam_named};

/// Anchor-pool BAM: tx0 and tx1, both with reads that clear the anchor
/// criteria (MAPQ 60, full-length, `pt:i = 100`).
///
/// The `AS:i = 100` is load-bearing: the gap is `AS_1 - (AS_2 or 0)`, so a
/// record with no `AS` tag measures a delta of 0 against the shipped floor of
/// 12 and is rejected into `rejected_gap`. An empty pool would make this
/// file's conservation tests vacuous — the subject here is a universe, and a
/// universe over zero anchored transcripts is a different scenario.
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

/// The MALFORMED reference: `tx1` is named twice, so the universe reads
/// `[tx0, tx1, tx1]`. `tx0` is anchored; `tx1` is not, so it is padded twice.
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

/// THE PRECONDITION OF THE CONSERVATION GUARD, pinned: the header reader
/// collapses duplicate `@SQ` names, so the guard is currently unreachable.
///
/// The reachable half of the failure mode is pinned in `emission_model.rs`,
/// which drives the library with a duplicated universe directly and shows it
/// is refused there, before any row is written.
///
/// IF THIS TEST GOES RED, a header-reader upgrade has changed the duplicate
/// handling. The CLI conservation guard has become REACHABLE and now needs its
/// own negative control before anyone relies on it.
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

/// THE CLI RUNS, PRINTS THE COMPLETENESS SPLIT, AND EXITS ZERO.
///
/// This is the positive half and it is not optional. Without it, "the
/// conservation guard did not fire" would also be satisfied by a run that fails
/// for any other reason.
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
    // The completeness block's print shape, asserted here because this is the
    // only test that runs the real CLI far enough to see it.
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

/// POSITIVE CONTROL, and it is not optional.
///
/// Without this the test above would also pass if `build-emission` failed for
/// ANY reason on any input — a missing flag, an unparseable header, a bad
/// parquet — so the failure would not be attributable to the duplication.
#[test]
fn a_well_formed_universe_still_succeeds() {
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
        "CONTROL: the same fixture with NO duplicated @SQ must succeed. If it \
         fails, the negative test above is not testing the guard -- it is \
         testing whatever else broke. stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("emission conservation violated"),
        "a well-formed universe must not trip the guard. stderr:\n{stderr}"
    );
    // The completeness block's print shape, asserted here because this is the
    // only test that runs the real CLI far enough to see it.
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
