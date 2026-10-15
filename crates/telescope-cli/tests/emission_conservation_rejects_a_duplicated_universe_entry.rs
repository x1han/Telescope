//! W1 — the emission-conservation `ensure!` in `main.rs`, and the measurement
//! that says it CANNOT fire from here.
//!
//! WHY A PROCESS-LEVEL TEST. The guard is `anyhow::ensure!` inside `fn main()`.
//! It does not panic; it returns `Err` from `main`, so the process exits
//! non-zero with the message on **stderr**. A `#[should_panic]` unit test
//! cannot see that at all, and a library-level test cannot see it either: the
//! assert lives in the binary, and nothing in `telescope-core` owns it. So this
//! test spawns the real binary via `CARGO_BIN_EXE_telescope` and asserts on the
//! exit code and on stderr.
//!
//! WHY A DUPLICATED `@SQ` IS THE ONLY INPUT THAT REACHES IT. Both sides of
//! the law are the cardinality of `anchored UNION universe`:
//!
//! ```text
//!   LHS = n_after                   = |anchored| + |universe \ anchored|
//!   RHS = n_reference + n_anch_not_ref = |universe| + |anchored \ universe|
//! ```
//!
//! and those are equal for any two SETS. So on well-formed input the guard is
//! an identity and cannot fail — which is a property worth stating plainly,
//! because "the guard passed" would otherwise be read as evidence the emitted
//! table adds up, and it is not: it is arithmetic over the CLI's own counters,
//! never against the parquet that was actually written.
//!
//! `read_reference_universe` explicitly does NOT de-duplicate `@SQ` names, on
//! the grounds that SAM requires `SN` to be unique and that silently collapsing
//! a duplicate would "turn a corrupt reference into a plausible-looking
//! universe". That is the right call, and it is exactly what makes the guard
//! reachable: a duplicated name enters `universe` twice, the padding loop pushes
//! the key into `tx_keys` twice (so the row is emitted TWICE), while
//! `n_imputed` is a `HashSet` and counts it ONCE. `n_after` therefore falls one
//! short of `n_reference`, and the run fails loudly instead of writing a
//! duplicated row and reporting a completeness line that does not add up.
//!
//! MEASURED OUTCOME, and it is not the one this file was written to produce:
//! the guard CANNOT fire from the CLI, because `read_reference_universe`
//! returns noodles' header MAP and that map collapses duplicate `@SQ` names.
//! Writing `@SQ SN:tx0, SN:tx1, SN:tx1` and running the real binary prints
//! `@SQ transcripts: 2`, and the run completes successfully. The negative
//! control that would have driven the `ensure!` is therefore not constructible
//! from outside the process, and `main.rs`'s `ensure!` has no test that can
//! make it fail.
//!
//! What this file therefore pins is the guard's PRECONDITION, which is the part
//! that can actually change: if a `noodles` upgrade stops collapsing
//! duplicates, the guard becomes reachable and `a_duplicated_universe_entry_is_collapsed_by_the_header_reader`
//! goes red. The reachable half of the failure mode is pinned in
//! `emission_model.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{synth_record, write_synthetic_bam_named};

/// Anchor-pool BAM: tx0 and tx1, both with reads that clear the anchor
/// criteria (MAPQ 60, full-length, `pt:i = 100`).
fn anchor_bam(tag: &str) -> PathBuf {
    write_synthetic_bam_named(
        &format!("{tag}_pool"),
        &["tx0", "tx1"],
        vec![
            synth_record("a0", 0, None, 0, 60),
            synth_record("a1", 0, None, 0, 60),
            synth_record("b0", 1, None, 0, 60),
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

/// THE PRECONDITION OF THE CONSERVATION GUARD, pinned.
///
/// The guard in `main.rs` is
/// ```text
///   n_after == n_reference + n_anchored_not_in_reference
/// ```
/// and both sides are the cardinality of `anchored UNION universe`, so for any
/// two SETS the law is an identity. The single input that breaks it is a
/// universe naming a transcript twice — the key enters the emitted set twice
/// while `n_imputed`, a `HashSet`, counts it once — and `universe` has exactly
/// one source: `read_reference_universe`, which returns noodles' header MAP.
///
/// **This test documents that the map collapses duplicate `@SQ` names, and so
/// the guard is currently unreachable.** That is not a footnote. It means the
/// `ensure!` protecting the P0-2 fix has no test that can make it fire, and its
/// soundness rests on a third-party crate's header semantics rather than on
/// anything this repository asserts.
///
/// The reachable half of the failure mode is pinned in `emission_model.rs`:
/// `test_a_duplicated_universe_key_reaches_the_emitted_row_set_twice` drives the
/// library with a duplicated universe directly and shows the row really is
/// emitted twice.
///
/// IF THIS TEST GOES RED, a `noodles` upgrade has changed the header map's
/// duplicate handling. The consequence is not cosmetic: the CLI conservation
/// guard has become REACHABLE, and now needs its own negative control before
/// anyone relies on it.
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

/// THE CLI RUNS, PRINTS THE W2 SPLIT, AND EXITS ZERO.
///
/// This is the positive half of W1 and it is not optional. Without it, "the
/// conservation guard did not fire" would also be satisfied by a run that fails
/// for any other reason. Here the same fixture with a WELL-FORMED universe is
/// required to succeed, which is what makes the unreachable-ness above an
/// attribute of the duplication rather than of the harness.
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
    // W2's print shape, asserted here because this is the only test that runs
    // the real CLI far enough to see it.
    assert!(
        stderr.contains("n_imputed_rows:"),
        "the completeness block must still print. stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("(padded: 0, suppressed: 0)"),
        "W2: with no transcript evidence and nothing to pad, both detail counts \
         are 0 and both must be PRINTED -- the split is the point of W2, and a \
         line that only appeared when the counts were non-zero would hide it. \
         stderr:\n{stderr}"
    );
}

/// POSITIVE CONTROL, and it is not optional. Without this, the test above
/// would also pass if `build-emission` failed for ANY reason on any input —
/// a missing flag, an unparseable header, a bad parquet. It asserts that a
/// well-formed universe runs to completion, so the failure in the other test
/// is attributable to the duplication and to nothing else.
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
    // W2's print shape, asserted here because this is the only test that runs
    // the real CLI far enough to see it.
    assert!(
        stderr.contains("n_imputed_rows:"),
        "the completeness block must still print. stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("(padded: 0, suppressed: 0)"),
        "W2: with no transcript evidence and nothing to pad, both detail counts \
         are 0 and both must be PRINTED -- the split is the point of W2, and a \
         line that only appears when the counts are non-zero would hide it. \
         stderr:\n{stderr}"
    );
}
