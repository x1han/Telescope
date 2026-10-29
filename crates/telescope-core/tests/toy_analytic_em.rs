//! Analytic toy cases: hand-computed closed forms for the TEL-EM.
//!
//! `tests/five_prime_end_to_end.rs` runs `max_iter = 1`, so the fixed point —
//! the thing an EM is actually for — is never reached and never has to be
//! correct: an analytic fixed point is the one kind of test a consistent
//! pair of wrong implementations cannot satisfy.
//!
//! # The fixture — two transcripts, seven molecules: 4 unambiguous on `tx0`
//! (AS 100, each contributing exactly `1.0` to `accum_w[tx0]`), 1
//! unambiguous on `tx1` (AS 80), and 2 ambiguous `{tx0, tx1}` reads
//! (AS 100 / 80) — the only reads in the E-step.
//! `poly_a_weight = 0`, so the only EM parameter that can matter is `alpha`.
//! The fixed point is derived in `fixed_point()`. The two `AS` values are
//! deliberately unequal: equal `AS` makes the likelihood constant and the EM
//! declares convergence while `alpha` is still moving.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{BooleanArray, Float64Array, StringArray, UInt32Array};
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
/// rate instead of the stopping rule's.
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
        // synthetic graph: `log_pal` is a real measurement
        poly_a_is_placeholder: false,
        stats: telescope_core::evidence::FilterCounts::default(),
    }
}

fn write_emission(path: &PathBuf) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("is_polyA_competent", DataType::Boolean, false),
        Field::new("mu_logPAL", DataType::Float64, false),
        Field::new("sigma2_logPAL", DataType::Float64, false),
        // Non-zero on both rows: these are hand-written fixtures, so no row is
        // imputed. The loader REQUIRES this column (it is the only in-schema
        // marker of an imputed row), so a fixture without it fails at the first
        // read rather than quietly reporting every row as anchored.
        Field::new("n_cohort_flnc", DataType::UInt32, false),
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
            Arc::new(UInt32Array::from(vec![10u32, 10u32])),
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
        PRUNE, MIN_WEIGHT, None, None, None,
    )
    .expect("solve_and_write_profile");

    let rows = read_profile(&profile);
    let _ = std::fs::remove_file(&emission);
    let _ = std::fs::remove_file(&profile);

    // Rows are looked up BY READ ID, not by position. The profile happens to
    // write the unambiguous reads first, but that ordering is an implementation
    // detail and it is conditional: a read whose transcript is missing from
    // the emission, or that is not poly(A)-competent, emits no row. A
    // positional slice would then be off by one and quietly compare the wrong
    // numbers. Keying on `read_id` also lets the test ASSERT that the
    // ambiguous reads really did emit two rows each.
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
    // would move the fixed point.
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
/// Derived from the M-step (`alpha = w_sum / n_total`, so
/// `alpha_0 = (4 + 2p)/N`, `alpha_1 = (3 − 2p)/N`) with `N` cancelling under
/// the E-step softmax's scale invariance, leaving
/// `2(1 − g)·p² + (2 + 3g)·p − 4 = 0`. The `2p²` terms only cancel at `g = 1`,
/// so the linear `4/(4+g)` is wrong by 1.8 pp.
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
/// quadratic, NOT the linear `4/(4+g)`, which is off by 1.8 pp. If you are
/// reading this and "simplifying" it back to the linear form, that is the exact
/// regression this file was written to prevent.
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
    // schema, so it has to be reconstructed from the M-step's own definition
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
/// **What it catches:** `<` for `<=` (off-by-one at the edge), a window read
/// as exclusive or off-by-one, the wrong candidate's gap, or a gate applied to
/// the read instead of the candidate. Each moves one of the two arms onto the
/// other's answer, and the pair is asymmetric by construction, so the
/// assertion cannot pass by coincidence.
/// **What it does not catch:** a neutered term — that puts both arms on
/// `expected_beta_off()` and only the first half goes red. It is not coverage
/// of the 5' channel's magnitude; Test 2 is.
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
        // The two arms must straddle, not merely each hit its own target: a solver
        // that ignored the gate entirely would satisfy neither.
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
///
/// 1e-12, not 1e-9: this asserts an EXACT cancellation — a constant added to
/// both `ll`s must vanish in the softmax to the last bit.
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
