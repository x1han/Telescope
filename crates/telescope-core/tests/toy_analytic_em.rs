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

const MIN_WEIGHT: f64 = 0.0;

const TOL_TIGHT: f64 = 1e-14;
const MAX_ITER: usize = 500;

const AS_TX0: i32 = 100;
const AS_TX1: i32 = 80;
const N_UNAMB_TX0: usize = 4;
const N_UNAMB_TX1: usize = 1;
const N_AMBIG: usize = 2;

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
    let mut candidate_3p_gap = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn push(
        molecules: &mut Vec<MoleculeEvidence>,
        candidate_pool: &mut Vec<u32>,
        candidate_as: &mut Vec<i32>,
        candidate_5p_gap: &mut Vec<u32>,
        candidate_3p_gap: &mut Vec<u32>,
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

            candidate_3p_gap.push(0);
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
            &mut candidate_3p_gap,
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
            &mut candidate_3p_gap,
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
            &mut candidate_3p_gap,
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
        candidate_3p_gap,
        candidate_nm: vec![0_i32; n_cands],
        candidate_mapq: vec![60_u8; n_cands],
        transcript_names: vec!["tx0".to_string(), "tx1".to_string()],

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
        Field::new("n_cohort_flnc", DataType::UInt32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["tx0", "tx1"])),
            Arc::new(StringArray::from(vec!["gene0", "gene0"])),
            Arc::new(BooleanArray::from(vec![true, true])),
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

fn gaps_at_window() -> Fixture {
    Fixture {
        ambiguous_gaps: [(WINDOW, 100); N_AMBIG],
    }
}

fn gaps_one_past_window() -> Fixture {
    Fixture {
        ambiguous_gaps: [(WINDOW + 1, 100); N_AMBIG],
    }
}

fn gaps_none() -> Fixture {
    Fixture {
        ambiguous_gaps: [(100, 100); N_AMBIG],
    }
}

fn gaps_tx0_only() -> Fixture {
    Fixture {
        ambiguous_gaps: [(0, 100); N_AMBIG],
    }
}

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

        assert!(
            a0 > b0,
            "molecule {i}: the atom firing must RAISE the posterior on tx0 \
             (at edge {a0}, past edge {b0})"
        );
    }
}

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
