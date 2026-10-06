//! End-to-end: real BAM on disk → `extract_evidence` → `solve_and_write_profile`
//! → read the profile back. Nothing here injects `candidate_5p_gap`.
//!
//! # Why this file exists
//!
//! Every 5'-term test that lived in `em.rs` built its graph with
//! `synth_graph_with_gaps`, which writes the 5' gap pool **directly**. That
//! makes all of them blind to the half of the feature that could actually be
//! wrong: the BAM → `g_k` path in `evidence.rs`. The gap a test hands the E-step
//! and the gap a real BAM produces are the same number only if
//! `CandidateAlign::gap5p` is computed correctly — and if it were not, not one
//! of the 37 unit tests would go red.
//!
//! So the acceptance criterion for this file is a mutation, not a number:
//! **forcing `gap5p` to a constant must turn the assertion in this file red.**
//! `test_the_5p_gap_survives_the_bam_to_em_step_seam` is that tripwire; it is
//! written so that a neutered `gap5p` makes the two candidates identical, and
//! identical candidates are 50/50 under *any* implementation of the likelihood,
//! including one where the 5' term does not exist.
//!
//! # The fixture
//!
//! One ambiguous read, aligned to `tx0` and `tx1` with **equal `AS:i`**, so the
//! alignment term `ln A_rk` is exactly 0 in both directions. The two
//! candidates differ in exactly one observable: `tx0`'s alignment starts at
//! base 1 (`g_k = 0`, inside the 20 bp window) and `tx1`'s starts 500 bp in
//! (`g_k = 500`, outside it). The emission gives the two transcripts the *same*
//! `mu_logPAL`, so the poly(A) term is symmetric too and cannot break the tie.
//!
//! That leaves the 5' term as the only thing that can move the posterior, which
//! is what makes "the posterior moved" evidence about the seam rather than
//! about the emission model.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{BooleanArray, Float64Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use telescope_core::em::solve_and_write_profile;
use telescope_core::evidence::{extract_evidence, test_support};

/// λ, β, window and prune are the shipped frozen values. See
/// `workspace/notes/EXPERIMENT.md` for provenance.
const POLY_A_ON: f64 = 1.0;
const LAMBDA: f64 = 20.0;
const BETA_ON: f64 = 1.8;
const BETA_OFF: f64 = 0.0;
const WINDOW: u32 = 20;
const PRUNE: i32 = 50;
/// Truncation OFF, so the two arms are scored on the same support. At the 0.01
/// default the β-on arm emits more rows than the β-off arm (measured on GNG11:
/// 2726 vs 1816) and the comparison would not be like-for-like.
const MIN_WEIGHT: f64 = 0.0;

/// Read `(transcript_id, posterior_weight)` back out of a profile parquet.
fn read_profile(path: &std::path::Path) -> Vec<(String, f64)> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let file = std::fs::File::open(path).expect("open profile");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("profile reader")
        .with_batch_size(1024)
        .build()
        .expect("build profile reader");
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch.expect("read profile batch");
        let tx = batch
            .column_by_name("transcript_id")
            .expect("transcript_id")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("transcript_id is StringArray");
        let w = batch
            .column_by_name("posterior_weight")
            .expect("posterior_weight")
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("posterior_weight is Float64Array");
        for i in 0..batch.num_rows() {
            out.push((tx.value(i).to_string(), w.value(i)));
        }
    }
    out
}

fn scratch(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "telescope_e2e_{}_{}.parquet",
        name,
        std::process::id()
    ));
    let _ = std::fs::remove_file(&p);
    p
}

/// Write a minimal emission parquet.
///
/// Deliberately NOT shared with `em.rs`'s test writer: that one is a
/// `#[cfg(test)]` item in a different crate. The drift risk is bounded and
/// fails LOUDLY — `load_emission_parameters` `context()`s on any column it
/// needs and cannot find, so adding a sixth column to the reader breaks this
/// test at the first read instead of quietly reading a default.
///
/// That prediction was tested for real on 2026-10-06: the sixth column
/// (`n_cohort_flnc`, the in-schema marker of an IMPUTED row) went in, and
/// `cargo test --lib` stayed green because it only builds the lib's own
/// fixtures. The breakage appeared in `cargo test --release` on this file
/// only. A required column is a change to the READ CONTRACT, and a lib-only
/// test run cannot see one.
fn write_emission(path: &PathBuf, rows: &[(&str, &str, f64, f64)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("is_polyA_competent", DataType::Boolean, false),
        Field::new("mu_logPAL", DataType::Float64, false),
        Field::new("sigma2_logPAL", DataType::Float64, false),
        // Non-zero on every row: these are hand-fitted values, so no row here
        // is imputed. The loader REQUIRES this column -- see the doc comment
        // above and `TranscriptParam::is_imputed`.
        Field::new("n_cohort_flnc", DataType::UInt32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(vec![true; rows.len()])),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.2).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.3).collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(vec![10u32; rows.len()])),
        ],
    )
    .expect("emission RecordBatch");
    let file = std::fs::File::create(path).expect("create emission");
    let mut w = ArrowWriter::try_new(
        file,
        batch.schema(),
        Some(WriterProperties::builder().build()),
    )
    .expect("emission writer");
    w.write(&batch).expect("write emission batch");
    w.close().expect("close emission");
}

/// The real pipeline, run end to end. Returns `(pi_tx0, rows)` as percentages
/// and counts, with the 5' term set to `beta`.
fn run_pipeline(beta: f64, tag: &str) -> (f64, usize) {
    // `pt:i = 100` in every record, so the molecule's log-pal is log2(100). Both
    // transcripts get that same value as their emission μ, making the poly(A)
    // term symmetric across the pair.
    let mu = (100.0_f64).log2();
    let records = vec![
        test_support::synth_record_with_gap("r1", 0, 0, false, Some(100), 0, 60),
        test_support::synth_record_with_gap("r1", 1, 500, false, Some(100), 0, 60),
    ];
    let bam = test_support::write_synthetic_bam(tag, records);
    let emission = scratch(&format!("{tag}_emission"));
    let profile = scratch(&format!("{tag}_profile"));
    write_emission(
        &emission,
        &[("tx0", "gene0", mu, 0.04), ("tx1", "gene1", mu, 0.04)],
    );

    // Module 2. No id mapping: this fixture's BAM and emission already share a
    // key space, which is the case `solve_and_write_profile` now refuses if it
    // is not.
    let graph = extract_evidence(&bam, 35).expect("extract_evidence");
    assert_eq!(
        graph.molecules.len(),
        1,
        "the two records must pool into ONE molecule, not two"
    );
    assert_eq!(
        graph.molecules[0].candidate_count, 2,
        "the read must be ambiguous (candidate_count > 1) or it takes the \
         deterministic w=1 branch and never reaches the E-step at all"
    );

    // Module 3.
    solve_and_write_profile(
        graph, &emission, "TEST", &profile, 1e-6, 1, LAMBDA, beta, POLY_A_ON, WINDOW, PRUNE,
        MIN_WEIGHT, None, None, None,
    )
    .expect("solve_and_write_profile");

    let d = read_profile(&profile);
    let _ = std::fs::remove_file(&bam);
    let _ = std::fs::remove_file(&emission);
    let _ = std::fs::remove_file(&profile);

    let total: f64 = d.iter().map(|&(_, w)| w).sum();
    assert!(
        total > 0.0,
        "profile {tag} is empty -- every read was dropped, which is the \
         id-space / is_competent failure mode, not a modelling result"
    );
    let tx0: f64 = d.iter().filter(|(t, _)| t == "tx0").map(|&(_, w)| w).sum();
    (100.0 * tx0 / total, d.len())
}

/// **The tripwire.** A 5'-gap asymmetry created in a real BAM must change the
/// posterior, which can only happen if `evidence.rs` actually read the read's
/// start position off the alignment.
///
/// If `gap5p` is neutered — forced to a constant, or to `u32::MAX`, or derived
/// from something other than the alignment start — both candidates get the same
/// gap, the atom fires for both or for neither, the two likelihoods become
/// identical, and the posterior is 50/50 by construction. `pi_off` is 50.0 and
/// `pi_on` collapses onto it, so the `> 90.0` assertion fails.
///
/// The 50/50 reading is not a bug in the assertion. It is the point: 50/50 is
/// what you get when the tie is genuinely un-broken, including under an
/// implementation in which the 5' term was deleted outright.
#[test]
fn test_the_5p_gap_survives_the_bam_to_em_step_seam() {
    let (pi_on, _) = run_pipeline(BETA_ON, "on");
    let (pi_off, _) = run_pipeline(BETA_OFF, "off");

    assert_eq!(
        pi_off, 50.0,
        "with the 5' term off, identical AS and identical emission μ must leave \
         the pair at exactly 50/50; anything else means the fixture is not as \
         symmetric as it claims and the comparison below is not controlled"
    );

    // The closed form, not a threshold. With `ln_a ≡ 0` (equal AS), identical μ
    // and π_0 = π_1 = 0.5, the E-step reduces to a two-way softmax in which only
    // tx0's atom fires, so
    //
    //     π_0 = e^β / (e^β + 1) = 6.0496 / 7.0496 = 85.81%   at β = 1.8
    //
    // Pinning THAT number to 0.1 pp is much stronger than "the posterior moved":
    // it says the gap the E-step used is the gap the BAM encodes, because any
    // other value of `gap5p` — constant, `u32::MAX`, off-by-one from the 1-based
    // start, or derived from the wrong end — either fails to break the tie (50%)
    // or breaks it by some other amount.
    let expected = 100.0 * BETA_ON.exp() / (BETA_ON.exp() + 1.0);
    assert!(
        (pi_on - expected).abs() < 0.1,
        "posterior on tx0 was {pi_on:.4}%, expected the closed form e^β/(e^β+1) \
         = {expected:.4}% to within 0.1 pp. A value near 50% means the 5' gap \
         never reached the E-step — check `gap5p` in evidence.rs. A value \
         anywhere else means it arrived with the wrong magnitude."
    );
}

/// The two arms must differ ONLY because of the 5' term, so they have to be
/// scored on the same reads and the same emitted support.
#[test]
fn test_the_ablation_arms_differ_only_by_the_5p_term() {
    let (pi_on, rows_on) = run_pipeline(BETA_ON, "rows_on");
    let (pi_off, rows_off) = run_pipeline(BETA_OFF, "rows_off");

    assert_eq!(
        rows_on, 2,
        "with `min_weight = 0` and two candidates per read, both rows must be \
         emitted; got {rows_on}"
    );
    assert_eq!(rows_on, rows_off, "arms must be scored on the same support");
    assert!(
        (pi_on - pi_off).abs() > 20.0,
        "gate Δ = {pi_on:.2} - {pi_off:.2} = {:.2} pp, below the 20 pp bar. \
         A Δ near 0 is the neutered-`gap5p` signature: both candidates get the \
         same gap, the atom fires for both or neither, and the two arms agree.",
        pi_on - pi_off
    );
}
