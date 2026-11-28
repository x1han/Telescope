use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{BooleanArray, Float64Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use telescope_core::em::solve_and_write_profile;
use telescope_core::evidence::{extract_evidence, test_support};

const POLY_A_ON: f64 = 1.0;
const LAMBDA: f64 = 20.0;
const BETA_ON: f64 = 1.8;
const BETA_OFF: f64 = 0.0;
const WINDOW: u32 = 20;
const PRUNE: i32 = 50;

const MIN_WEIGHT: f64 = 0.0;

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

fn write_emission(path: &PathBuf, rows: &[(&str, &str, f64, f64)]) {
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

fn run_pipeline(beta: f64, tag: &str) -> (f64, usize) {
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

    let expected = 100.0 * BETA_ON.exp() / (BETA_ON.exp() + 1.0);
    assert!(
        (pi_on - expected).abs() < 0.1,
        "posterior on tx0 was {pi_on:.4}%, expected the closed form e^β/(e^β+1) \
         = {expected:.4}% to within 0.1 pp. A value near 50% means the 5' gap \
         never reached the E-step — check `gap5p` in evidence.rs. A value \
         anywhere else means it arrived with the wrong magnitude."
    );
}

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
