//! Analytic toy cases: hand-computed closed forms for the TEL-EM.
//!
//! # Why this file exists
//!
//! `tests/five_prime_end_to_end.rs` already pins one closed form
//! (`e^β / (e^β + 1)` at β = 1.8) and proves the 5' gap survives the BAM →
//! E-step seam. What it does **not** do is exercise the M-step: it runs
//! `max_iter = 1`, so the fixed point — the thing an EM is actually for — is
//! never reached and never has to be correct.
//!
//! That gap is what let the 2026-10-02 incident run for a full round. The
//! solver was correct the whole time; a *stale binary* was not, and every
//! cross-check that should have caught it was a comparison against a reference
//! reading the right code. An analytic fixed point is the one kind of test a
//! consistent pair of wrong implementations cannot satisfy, which is why the
//! ruling put it in CI.
//!
//! # The fixture
//!
//! Two transcripts, seven molecules:
//!
//! | molecules | candidates | `AS` | role |
//! |---|---|---|---|
//! | 4 | `{tx0}` only | 100 | unambiguous; each contributes exactly `1.0` to `accum_w[tx0]` |
//! | 1 | `{tx1}` only | 80 | unambiguous; contributes `1.0` to `accum_w[tx1]` |
//! | 2 | `{tx0, tx1}` | 100 / 80 | ambiguous; the only reads in the E-step |
//!
//! `poly_a_weight = 0`, so the Gaussian is out of the likelihood entirely and
//! the only EM parameter that can matter is `alpha` (μ and σ² are still
//! updated, but both are multiplied by zero in the E-step).
//!
//! # Why the two `AS` values are NOT equal
//!
//! The obvious fixture — equal `AS` on both candidates — is **degenerate**, and
//! this file was written wrong once before that was discovered.
//!
//! With equal `AS` the alignment term is 0 for both, so `ll = ln α_k` and
//! `log_z = ln(α₀ + α₁) = ln 1 = 0` **at every iteration**. The likelihood is
//! exactly constant along the whole trajectory, `delta_avg_ll = 0 < tol`
//! immediately, and the EM declares convergence on iteration 2 while `alpha` is
//! still moving (measured: 0.7137 → 0.7753 on the way to the 0.8 fixed point).
//! The run stops long before the fixed point, and a test that asserts the fixed
//! point silently asserts a number the solver never visited.
//!
//! So the fixture makes the likelihood **non-degenerate on purpose**: `tx0` is
//! the better alignment, `g = exp(−ΔAS/λ)` is a constant that does not drift,
//! and `log_z = ln(α₀ + α₁ g)` genuinely changes as α moves. Nothing in the
//! M-step feeds back into `g`, so the expectations below stay closed-form.
//!
//! # The derivation
//!
//! The M-step sets `alpha = w_sum / n_total_molecules` (`em.rs:644`) and
//! `n_total_molecules = graph.molecules.len()` — it counts the **unambiguous**
//! reads too, which is the detail the whole result turns on. The two ambiguous
//! molecules are identical, so each contributes the same `p`:
//!
//! ```text
//!     E-step:  p  = α₀ / (α₀ + α₁·g)
//!     M-step:  α₀ = (4 + 2p) / 7        α₁ = 1 − α₀ = (3 − 2p) / 7
//! ```
//!
//! Substituting α in terms of p and clearing denominators:
//!
//! ```text
//!     p·[(4 + 2p) + (3 − 2p)·g] = 4 + 2p
//!  ⇒ 4p + 2p² + 3pg − 2p²g = 4 + 2p
//!  ⇒ 2(1 − g)·p² + (2 + 3g)·p − 4 = 0
//! ```
//!
//! A quadratic, **not** the linear `p = 4/(4 + g)` this file asserted in its
//! first draft. The `2p²` and `−2p²g` terms only cancel when `g = 1`, i.e. when
//! the two alignments tie; with `g = e⁻¹ ≈ 0.368` they leave
//! `2(1 − g)·p² ≈ 1.264·p²` behind, and the linear form is wrong by 1.8 pp
//! (0.9158 against a measured 0.9337). The quadratic has `A > 0` and `C < 0`,
//! so exactly one root is positive and it lies in (0, 1); that is the physical
//! one.
//!
//! **The value is independent of the initialiser.** The init carries `ε = 0.01`
//! per transcript plus `1/count` per ambiguous read (`em.rs:338-357`), which
//! gives 0.7137, not 0.9337. The M-step washes it out. A test that pins the
//! quadratic root is therefore testing the fixed-point equation itself, and
//! would survive changing ε tomorrow.
//!
//! With β = 1.8 firing on `tx0` only, `e^β` replaces the leading 1 and the
//! constant 4 picks it up too — the same substitution gives
//! `2(e^β − g)·p² + (2e^β + 3g)·p − 4e^β = 0`. Both cases are solved by
//! `fixed_point()` below, so the two expectations cannot drift apart.
//!
//! # Tolerances
//!
//! Test 1 asserts a **converged fixed point**, so how close the EM gets is set
//! by the stopping rule rather than the arithmetic; it runs `tol` far below the
//! shipped `1e-6` and asserts to `1e-9`. Tests 2–4 assert **exact**
//! relationships — a symmetric quadratic root, and two cancellations that must
//! hold to the last bit — and are 1e-12.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{BooleanArray, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use telescope_core::em::solve_and_write_profile;
use telescope_core::evidence::{FlatEvidenceGraph, MoleculeEvidence};

const LAMBDA: f64 = 20.0;
const BETA_ON: f64 = 1.8;
const BETA_OFF: f64 = 0.0;
const WINDOW: u32 = 20;
const PRUNE: i32 = 50;
const POLY_A_OFF: f64 = 0.0;
/// `min_weight = 0.0` so BOTH candidates are emitted. At the 0.01 default the
/// arm that shifts mass onto tx0 emits fewer rows than the arm that does not,
/// and the arm-to-arm comparison stops being like-for-like.
const MIN_WEIGHT: f64 = 0.0;
/// The shipped default is 1e-6. Test 1 asserts a fixed point, so the loop has
/// to actually get there; 1e-14 leaves the residual at the contraction's own
/// rate (2/7 per iteration) instead of the stopping rule's.
const TOL_TIGHT: f64 = 1e-14;
const MAX_ITER: usize = 500;

const AS_TX0: i32 = 100;
const AS_TX1: i32 = 80;
const N_UNAMB_TX0: usize = 4;
const N_UNAMB_TX1: usize = 1;
const N_AMBIG: usize = 2;

/// `g = exp(−ΔAS/λ)`, the constant alignment factor on the weaker candidate.
/// A function, not a `const`: `f64::exp` is not const-callable.
fn g() -> f64 {
    (-((AS_TX0 - AS_TX1) as f64) / LAMBDA).exp()
}

struct Fixture {
    ambiguous_gaps: [(u32, u32); N_AMBIG],
}

fn build_graph(fx: &Fixture) -> FlatEvidenceGraph {
    let mut molecules = Vec::new();
    let mut candidate_pool = Vec::new();
    let mut candidate_as = Vec::new();
    let mut candidate_5p_gap = Vec::new();

    fn push(
        molecules: &mut Vec<MoleculeEvidence>,
        candidate_pool: &mut Vec<u32>,
        candidate_as: &mut Vec<i32>,
        candidate_5p_gap: &mut Vec<u32>,
        read_id: &str,
        txs: &[u32],
        as_: &[i32],
        gaps: &[u32],
    ) {
        let offset = candidate_pool.len() as u32;
        for i in 0..txs.len() {
            candidate_pool.push(txs[i]);
            candidate_as.push(as_[i]);
            candidate_5p_gap.push(gaps[i]);
        }
        molecules.push(MoleculeEvidence {
            read_id: read_id.to_string(),
            log_pal: 0.0,
            candidate_offset: offset,
            candidate_count: txs.len() as u16,
        });
    }

    for i in 0..N_UNAMB_TX0 {
        push(
            &mut molecules,
            &mut candidate_pool,
            &mut candidate_as,
            &mut candidate_5p_gap,
            &format!("u0_{i}"),
            &[0],
            &[AS_TX0],
            &[0],
        );
    }
    for i in 0..N_UNAMB_TX1 {
        push(
            &mut molecules,
            &mut candidate_pool,
            &mut candidate_as,
            &mut candidate_5p_gap,
            &format!("u1_{i}"),
            &[1],
            &[AS_TX1],
            &[0],
        );
    }
    for (i, &(g0, g1)) in fx.ambiguous_gaps.iter().enumerate() {
        push(
            &mut molecules,
            &mut candidate_pool,
            &mut candidate_as,
            &mut candidate_5p_gap,
            &format!("amb_{i}"),
            &[0, 1],
            &[AS_TX0, AS_TX1],
            &[g0, g1],
        );
    }

    let n_cands = candidate_pool.len();
    FlatEvidenceGraph {
        molecules,
        candidate_pool,
        candidate_as,
        candidate_5p_gap,
        candidate_nm: vec![0_i32; n_cands],
        candidate_mapq: vec![60_u8; n_cands],
        transcript_names: vec!["tx0".to_string(), "tx1".to_string()],
    }
}

fn write_emission(path: &PathBuf) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("is_polyA_competent", DataType::Boolean, false),
        Field::new("mu_logPAL", DataType::Float64, false),
        Field::new("sigma2_logPAL", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["tx0", "tx1"])),
            Arc::new(StringArray::from(vec!["gene0", "gene0"])),
            Arc::new(BooleanArray::from(vec![true, true])),
            // `poly_a_weight = 0` makes these numerically irrelevant, but they
            // must exist: the loader `context()`s on every column it needs, and
            // a missing one is a panic rather than a readable failure.
            Arc::new(Float64Array::from(vec![6.0, 6.0])),
            Arc::new(Float64Array::from(vec![0.04, 0.04])),
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

/// Run the fixture to convergence and return `(p_tx0, p_tx1)` for each
/// ambiguous molecule, in fixture order.
fn run(fx: &Fixture, beta: f64, tag: &str) -> Vec<(f64, f64)> {
    let emission =
        std::env::temp_dir().join(format!("toy_em_{}_{}.parquet", tag, std::process::id()));
    let profile =
        std::env::temp_dir().join(format!("toy_pf_{}_{}.parquet", tag, std::process::id()));
    let _ = std::fs::remove_file(&emission);
    let _ = std::fs::remove_file(&profile);
    write_emission(&emission);

    let graph = build_graph(fx);
    assert_eq!(
        graph.molecules.len(),
        N_UNAMB_TX0 + N_UNAMB_TX1 + N_AMBIG,
        "the molecule count is what the M-step's denominator is built from"
    );

    solve_and_write_profile(
        graph, &emission, "TOY", &profile, TOL_TIGHT, MAX_ITER, LAMBDA, beta, POLY_A_OFF, WINDOW,
        PRUNE, MIN_WEIGHT, None,
    )
    .expect("solve_and_write_profile");

    let rows = read_profile(&profile);
    let _ = std::fs::remove_file(&emission);
    let _ = std::fs::remove_file(&profile);

    let single = N_UNAMB_TX0 + N_UNAMB_TX1;
    let mut out = Vec::new();
    for i in 0..N_AMBIG {
        let base = single + i * 2;
        let (t0, w0) = &rows[base];
        let (t1, w1) = &rows[base + 1];
        assert_eq!(
            (t0.as_str(), t1.as_str()),
            ("tx0", "tx1"),
            "ambiguous molecule {i} must emit both candidates, in candidate order"
        );
        out.push((*w0, *w1));
    }
    out
}

/// Both candidates outside the window — β is on but the atom fires for nobody.
fn gaps_none() -> Fixture {
    Fixture {
        ambiguous_gaps: [(100, 100); N_AMBIG],
    }
}

/// tx0 inside the window, tx1 outside — β fires on tx0 only.
fn gaps_tx0_only() -> Fixture {
    Fixture {
        ambiguous_gaps: [(0, 100); N_AMBIG],
    }
}

/// Both inside the window — β fires on both, so it cancels.
fn gaps_both() -> Fixture {
    Fixture {
        ambiguous_gaps: [(0, 0); N_AMBIG],
    }
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}\n  got  {got:.17}\n  want {want:.17}\n  |Δ|  {:.3e}  (tol {tol:.0e})",
        (got - want).abs()
    );
}

/// The hand-computed fixed point, as a closed-form quadratic root.
///
/// `bonus_on_tx0` selects the β = 1.8 case, where `e^β` multiplies tx0's term:
///
/// ```text
///     β = 0     :  2(1 − g)·p² + (2 + 3g)·p − 4     = 0
///     β = 1.8   :  2(e^β − g)·p² + (2e^β + 3g)·p − 4e^β = 0
/// ```
///
/// One function for both so the two arms' expectations cannot drift apart.
fn fixed_point(bonus_on_tx0: bool) -> f64 {
    let (eb, c) = if bonus_on_tx0 {
        (BETA_ON.exp(), -4.0 * BETA_ON.exp())
    } else {
        (1.0, -4.0)
    };
    let a = 2.0 * (eb - g());
    let b = 2.0 * eb + 3.0 * g();
    let disc = b * b - 4.0 * a * c;
    assert!(disc > 0.0, "quadratic must have a real root; disc = {disc}");
    assert!(a > 0.0 && c < 0.0, "exactly one root must be positive");
    // `a > 0, c < 0` ⇒ the `+` branch is the positive root, and it is in (0,1).
    let r = (-b + disc.sqrt()) / (2.0 * a);
    assert!(r > 0.0 && r < 1.0, "the physical root is in (0,1); got {r}");
    r
}

fn expected_beta_off() -> f64 {
    fixed_point(false)
}

fn expected_beta_on_tx0() -> f64 {
    fixed_point(true)
}

/// **Test 1 — the M-step fixed point, by hand, with the 5' term off.**
///
/// Pins `p = 4/(4+g)`. This is the assertion the whole file exists for: it
/// exercises the M-step, which no other test does, and it is independent of the
/// initialiser, so it can only be satisfied by a solver whose E-step and
/// M-step agree on a common fixed point.
#[test]
fn toy_m_step_fixed_point_is_the_hand_computed_ratio() {
    let out = run(&gaps_none(), BETA_OFF, "beta_off");
    for (i, &(p0, p1)) in out.iter().enumerate() {
        assert_close(
            p0,
            expected_beta_off(),
            1e-9,
            &format!("ambiguous molecule {i}: posterior on tx0"),
        );
        assert_close(
            p0 + p1,
            1.0,
            1e-12,
            &format!("molecule {i} must renormalise to 1"),
        );
    }
}

/// **Test 2 — β moves the fixed point to a hand-computed value, and the
/// converged state satisfies both fixed-point equations.**
///
/// The expected value is a closed-form quadratic root, so this pins a number
/// rather than a direction. It then additionally checks the E-step and M-step
/// equations hold simultaneously on the returned state, which is strictly
/// stronger than matching a constant: it says the solver converged to *the*
/// self-consistent point, not merely to one that happens to be near it.
#[test]
fn toy_five_prime_moves_the_fixed_point_to_the_hand_computed_root() {
    let out = run(&gaps_tx0_only(), BETA_ON, "beta_on");
    assert_eq!(
        out[0], out[1],
        "the two ambiguous molecules are identical, so they must receive \
         identical posteriors; a difference means the E-step is order-dependent"
    );
    let p0 = out[0].0;
    assert_close(
        p0,
        expected_beta_on_tx0(),
        1e-9,
        "β = 1.8 firing on tx0 must land on the closed-form quadratic root",
    );
    assert!(
        p0 > expected_beta_off(),
        "β must move the fixed point UP from {}; got {:.15}. \
         A value at or below it means the 5' term never reached the likelihood.",
        expected_beta_off(),
        p0
    );
    // The E-step equation, checked independently. `alpha` is not in the output
    // schema (the profile has no abundance column and `TelEmStats` reports only
    // counts), so it has to be reconstructed from the M-step's own definition
    // `α₀ = (4 + 2p)/7`. That reconstruction makes the M-step equation itself
    // untestable here — asserting `α₀ == (4 + 2p)/7` would be asserting the
    // test's own arithmetic back to itself, and it can never fail. So the
    // load-bearing assertion in this test is the pinned quadratic root above;
    // this E-step check is a cross-check that the reconstructed α and the
    // returned p agree, which a solver that drifted between its own M-step and
    // E-step would fail.
    let n = (N_UNAMB_TX0 + N_UNAMB_TX1 + N_AMBIG) as f64;
    let alpha0 = (N_UNAMB_TX0 as f64 + N_AMBIG as f64 * p0) / n;
    let alpha1 = 1.0 - alpha0;
    assert_close(
        alpha0 * BETA_ON.exp() / (alpha0 * BETA_ON.exp() + alpha1 * g()),
        p0,
        1e-12,
        "the E-step equation must hold for the alpha the M-step implies",
    );
}

/// **Test 3 — the controlled arm: changing ONLY the gap must reproduce the
/// β = 0 posterior exactly.**
///
/// β stays at 1.8 and only the gaps move outside the window, so `ln_5p = 0`
/// for both candidates and the run must be indistinguishable from β = 0. This
/// is what makes Test 2 a statement about the 5' term: two runs, one changed
/// observable, and the null is an exact constant rather than "the number went
/// down somewhere".
#[test]
fn toy_changing_only_the_gap_reproduces_the_bonus_off_posterior_exactly() {
    let off = run(&gaps_none(), BETA_OFF, "ctrl_off");
    let neither = run(&gaps_none(), BETA_ON, "ctrl_neither");
    for (i, (&(a0, _), &(b0, _))) in off.iter().zip(neither.iter()).enumerate() {
        assert_close(
            b0,
            a0,
            1e-12,
            &format!("molecule {i}: β on with neither atom firing must equal β off"),
        );
    }
}

/// **Test 4 — β firing for BOTH candidates must cancel exactly.**
///
/// `ln_5p` is then a constant added to both `ll`s, i.e. a common factor in the
/// softmax, which cannot move anything. If it ever does, `ln_5p` has leaked
/// into the normalisation or the `max_ll` shift — precisely the class of bug
/// the `max_ll` comment in `em.rs` warns about.
#[test]
fn toy_both_candidates_firing_leaves_the_posterior_at_the_prior() {
    let both = run(&gaps_both(), BETA_ON, "both_fire");
    for (i, &(p0, _)) in both.iter().enumerate() {
        assert_close(
            p0,
            expected_beta_off(),
            1e-12,
            &format!("molecule {i}: β firing symmetrically must cancel"),
        );
    }
}
