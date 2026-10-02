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
//! The M-step sets `alpha = w_sum / n_total_molecules` (`em.rs:648`), and
//! `n_total_molecules = graph.molecules.len()` — it counts the **unambiguous**
//! reads too. The two ambiguous molecules are identical, so each contributes
//! the same `p`:
//!
//! ```text
//!     alpha_0 = (4 + 2p) / N        alpha_1 = (3 - 2p) / N      N = 7
//! ```
//!
//! **The DENOMINATOR cancels, and the NUMERATOR is the load-bearing part.**
//! `p.alpha` is read at exactly two places (`em.rs:495` and `em.rs:507`), both
//! inside the E-step softmax, so alpha enters only through
//! `exp(ln alpha_k + ...) / Σ_j exp(ln alpha_j + ...)` — invariant under a
//! common rescaling. Substituting gives
//!
//! ```text
//!     p = (4 + 2p) / [(4 + 2p) + (3 - 2p)·g]        <- no N anywhere
//! ```
//!
//! and indeed mutating `/7` to `/2` turns **0 of 4** tests red. What the fixed
//! point does depend on is the numerator `4 + 2p`, i.e. the `+= 1.0` that each
//! unambiguous read contributes at `em.rs:605`: changing that to `+= 0.5` turns
//! **3 of 4** red. So the unambiguous reads matter through the *numerator*, and
//! this file's expectations would be unchanged if `N` were anything at all.
//!
//! The E-step is `p = α₀ / (α₀ + α₁·g)` — alpha as the mixture prior, not as
//! part of the poly(A) emission. Substituting alpha in terms of p and clearing
//! denominators:
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
//! per transcript plus `1/count` per ambiguous read (`em.rs:339-359`), which
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
//! Tests 1, 2 and 3 assert a **converged fixed point**, so how close the EM
//! gets is set by the stopping rule rather than the arithmetic; they run `tol`
//! far below the shipped `1e-6` and assert to `1e-9`. Test 4 asserts an
//! **exact cancellation** — a constant added to both `ll`s must vanish in the
//! softmax to the last bit — and is 1e-12.
//!
//! # What each test can and cannot detect
//!
//! The only honest answer to "is this a gate or decoration" is to mutate the
//! solver and re-run, so that is what was done, in isolated git worktrees with
//! their own `CARGO_TARGET_DIR` so the primary tree is never written to.
//! Measured:
//!
//! | mutation | turns red |
//! |---|---|
//! | `ln_5p` neutered (`em.rs:488` → always 0) | tests 2 and 3 |
//! | window gate `<=` → `<` | test 3, and `em.rs:2294` |
//! | window gate always fires (`em.rs:488` → unconditional) | tests 2 and 3 |
//! | M-step numerator `+= 1.0` → `+= 0.5` (`em.rs:605`) | **all four** |
//! | M-step denominator `/7` → `/2` (`em.rs:648`) | **nothing** — see the scale-invariance note above |
//! | output-stage renormalisation removed (`em.rs:789`) | **nothing** — at `min_weight = 0` the retained sum is already 1 to ~1e-16; covered by the `em.rs` unit tests instead |
//!
//! Re-run with `bash ci/gates.sh full`, or
//! `bash workspace/validation/controls/mutation_battery.sh`. The last row of
//! that table is a **prediction**: M5 turning anything red would mean the
//! scale-invariance argument above is wrong, and that is worth knowing.
//!
//! So: each test detects a real class, no test detects everything, and two
//! whole classes are invisible to this file by construction. Both are stated
//! rather than left for a reader to assume.
//!
//! | test | detects |
//! |---|---|
//! | 1 | a change to the M-step's numerator; initialiser drift |
//! | 2 | a neutered or wrong-magnitude `ln_5p` |
//! | 3 | the window gate being wrong at its own edge (`<` vs `<=`, wrong candidate's gap) |
//! | 4 | an **asymmetric** leak of `ln_5p` into the normalisation or the `max_ll` shift — a leak that is uniform across candidates still cancels in the softmax, which is why this test is green under both a neuter and a gate-always-on |

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

/// `(read_id, transcript_id, posterior_weight)`, one entry per profile row.
fn read_profile(path: &std::path::Path) -> Vec<(String, String, f64)> {
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
        let rid = batch
            .column_by_name("read_id")
            .expect("read_id")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("read_id is StringArray");
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
            out.push((
                rid.value(i).to_string(),
                tx.value(i).to_string(),
                w.value(i),
            ));
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
        PRUNE, MIN_WEIGHT, None, None,
    )
    .expect("solve_and_write_profile");

    let rows = read_profile(&profile);
    let _ = std::fs::remove_file(&emission);
    let _ = std::fs::remove_file(&profile);

    // Rows are looked up BY READ ID, not by position. The profile happens to
    // write the unambiguous reads first (`em.rs:673` then the ambiguous block
    // at 715), but that ordering is an implementation detail and it is
    // conditional: an unambiguous read whose transcript is missing from the
    // emission, or `is_polyA_competent == false`, is `continue`d without
    // emitting a row. A positional slice would then be off by one and quietly
    // compare the wrong numbers. Keying on `read_id` makes the test immune to
    // both the ordering and the skipped row, and it also lets the test ASSERT
    // that the ambiguous reads really did emit two rows each.
    let mut out = Vec::new();
    for i in 0..N_AMBIG {
        let rid = format!("amb_{i}");
        let got: Vec<(&str, f64)> = rows
            .iter()
            .filter(|(r, _, _)| r == &rid)
            .map(|(_, t, w)| (t.as_str(), *w))
            .collect();
        assert_eq!(
            got.len(),
            2,
            "ambiguous molecule {rid} must emit exactly 2 rows at \
             min_weight = 0, one per candidate; got {}: {got:?}",
            got.len()
        );
        assert_eq!(
            (got[0].0, got[1].0),
            ("tx0", "tx1"),
            "both candidates of {rid} must be present"
        );
        out.push((got[0].1, got[1].1));
    }
    // The unambiguous reads must be present too, at weight exactly 1.0 — they
    // are what the M-step's numerator is built from, so a silently dropped one
    // would move the fixed point and the closed forms would stop matching.
    for (i, tx) in (0..N_UNAMB_TX0)
        .map(|i| (i, "tx0"))
        .chain((0..N_UNAMB_TX1).map(|i| (i, "tx1")))
    {
        let rid = if tx == "tx0" {
            format!("u0_{i}")
        } else {
            format!("u1_{i}")
        };
        let got: Vec<(&str, f64)> = rows
            .iter()
            .filter(|(r, _, _)| r == &rid)
            .map(|(_, t, w)| (t.as_str(), *w))
            .collect();
        assert_eq!(
            got,
            vec![(tx, 1.0)],
            "unambiguous molecule {rid} must emit exactly one row at weight 1.0"
        );
    }
    out
}

/// tx0's gap exactly on the window edge — the atom fires (`<=`, `em.rs:488`).
fn gaps_at_window() -> Fixture {
    Fixture {
        ambiguous_gaps: [(WINDOW, 100); N_AMBIG],
    }
}

/// tx0's gap one nucleotide past the edge — the atom must NOT fire.
fn gaps_one_past_window() -> Fixture {
    Fixture {
        ambiguous_gaps: [(WINDOW + 1, 100); N_AMBIG],
    }
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
/// Pins `p` to the positive root of `2(1−g)·p² + (2+3g)·p − 4 = 0` — the
/// quadratic, NOT the linear `4/(4+g)` that the first draft of this file used
/// and that is off by 1.8 pp. See the module docstring: the `2p²` and
/// `−2p²g` terms only cancel at `g = 1`, so the equation is genuinely
/// quadratic. If you are reading this and "simplifying" it back to the linear
/// form, that is the exact regression this file was written to prevent.
///
/// This is the assertion the whole file exists for: it exercises the M-step,
/// which no other test does, and it is independent of the initialiser, so it
/// can only be satisfied by a solver whose E-step and M-step agree on a common
/// fixed point.
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
            &format!(
                "molecule {i}: the emitted weights must sum to 1. NOTE this does NOT \
                 test the output-stage renormalisation -- at min_weight = 0 the \
                 retained sum is already 1 to ~1e-16, so deleting `w / retained_sum` \
                 leaves this green. It is a sanity check on the emit path, not \
                 coverage of that code; see the module docstring."
            ),
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

/// **Test 3 — the gate boundary: β is held ON and only `gap5p` moves, by
/// one nucleotide, across the window edge.**
///
/// `em.rs:488` reads `if graph.candidate_5p_gap[idx] <= five_prime_window` —
/// inclusive, and it must be the *candidate's own* gap, not the read's or the
/// other candidate's. Nothing else in this file pins either. Test 2 shows the
/// term fires somewhere; this shows it fires in exactly the right place:
///
/// ```text
///   gap5p = (WINDOW, 100)  →  atom fires on tx0  →  p = expected_beta_on_tx0()
///   gap5p = (WINDOW+1, 100) →  atom fires on nobody →  p = expected_beta_off()
/// ```
///
/// Both expectations are the same closed-form roots Tests 1 and 2 use, so this
/// is a statement about the gate and not a new constant. The `AS` values are
/// module constants, so they do not move with the gap and `g` is identical in
/// both runs — the gate is the only thing that changed.
///
/// **Overlap with an existing test, stated rather than hidden.**
/// `em.rs:2294 test_five_prime_window_is_inclusive_at_20` already pins this
/// edge, and both were kept — settled by mutation rather than by taste. The
/// `<=` → `<` mutation turns **both** red, so neither is redundant. Three
/// concrete differences:
///
/// 1. It runs **after convergence**. The existing pair uses `max_iter = 1`
///    (`em.rs:2143`), so it pins the gate in the initialiser's neighbourhood —
///    re-testing a `max_iter = 1` property inside a file whose founding
///    observation is that `max_iter = 1` never reaches the fixed point would add
///    nothing; re-testing it AT the fixed point does.
/// 2. Its "must not fire" arm pins **0.5**, the degenerate flat answer, which an
///    always-on gate, an always-off gate, a correct gate, and a solver that
///    never reads `five_prime_weight` all satisfy. This test's corresponding arm
///    pins 0.933695 where firing would give 0.989727 — a 5.6e-2 separation, so
///    a wrong gate cannot land on the right number.
/// 3. It is an integration test of the public API; `two_candidate_posteriors` is
///    private to `src/em.rs` and cannot run through the parquet round-trip.
///
/// **What it catches:** `<` for `<=` (off-by-one at the edge), a window read
/// as exclusive or off-by-one, the wrong candidate's gap, or a gate applied to
/// the read instead of the candidate. Each moves one of the two arms onto the
/// other's answer, and the pair is asymmetric by construction, so the
/// assertion cannot pass by coincidence.
/// **What it does not catch:** a neutered term — that puts both arms on
/// `expected_beta_off()` and only the first half goes red. It is not coverage
/// of the 5' channel's magnitude; Test 2 is. See the module docstring for the
/// mutation result.
///
/// 1e-9, not 1e-12: both halves assert a *converged fixed point*, so they are
/// subject to the stopping rule exactly like Test 1.
#[test]
fn toy_changing_only_the_gap_reproduces_the_bonus_off_posterior_exactly() {
    let at_edge = run(&gaps_at_window(), BETA_ON, "gate_at");
    let past_edge = run(&gaps_one_past_window(), BETA_ON, "gate_past");
    for (i, (&(a0, _), &(b0, _))) in at_edge.iter().zip(past_edge.iter()).enumerate() {
        assert_close(
            a0,
            expected_beta_on_tx0(),
            1e-9,
            &format!("molecule {i}: gap5p.tx0 == WINDOW must fire the atom"),
        );
        assert_close(
            b0,
            expected_beta_off(),
            1e-9,
            &format!("molecule {i}: gap5p.tx0 == WINDOW+1 must NOT fire the atom"),
        );
        // The two arms must straddle, not merely each hit its own target: a
        // solver that ignored the gate entirely would satisfy neither, but
        // this pins that the edge is where the model says it is.
        assert!(
            a0 > b0,
            "molecule {i}: the atom firing must RAISE the posterior on tx0 \
             (at edge {a0}, past edge {b0})"
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
