mod common;

use common::{run_manual_chain_with_universe, scratch, stderr_of, telescope};
use telescope_core::evidence::test_support::{
    synth_record, synth_record_without_pt, write_synthetic_bam_named,
};

fn mixed_bam(tag: &str) -> std::path::PathBuf {
    let mut recs = Vec::new();
    for i in 0..65 {
        recs.push(synth_record(&format!("m{i}"), 0, Some(100), 0, 60));
    }
    for i in 0..35 {
        recs.push(synth_record_without_pt(
            &format!("n{i}"),
            0,
            Some(100),
            0,
            60,
        ));
    }
    for i in 0..3 {
        recs.push(synth_record(&format!("a{i}"), 1, Some(100), 0, 60));
    }
    for i in 0..4 {
        recs.push(synth_record(&format!("b{i}"), 2, Some(100), 0, 60));
    }
    write_synthetic_bam_named(tag, &["tx0", "tx1", "tx2"], recs)
}

fn pure_no_tail_bam(tag: &str) -> std::path::PathBuf {
    let mut recs = Vec::new();
    for i in 0..100 {
        recs.push(synth_record_without_pt(
            &format!("h{i}"),
            0,
            Some(100),
            0,
            60,
        ));
    }
    for i in 0..3 {
        recs.push(synth_record(&format!("a{i}"), 1, Some(100), 0, 60));
    }
    for i in 0..4 {
        recs.push(synth_record(&format!("b{i}"), 2, Some(100), 0, 60));
    }
    write_synthetic_bam_named(tag, &["tx0", "tx1", "tx2"], recs)
}

fn profile_rows(path: &std::path::Path) -> Vec<(String, Option<i32>, f64)> {
    use arrow::array::{Array, Float64Array, Int32Array, StringArray};
    let file = std::fs::File::open(path).expect("open the profile");
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("the profile is a parquet table")
        .build()
        .expect("build the reader");
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch.expect("read a batch");
        let tx = batch
            .column_by_name("transcript_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .expect("the transcript_id column");
        let pal = batch
            .column_by_name("polyA_length")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
            .expect("the polyA_length column");
        let w = batch
            .column_by_name("posterior_weight")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
            .expect("the posterior_weight column");
        for i in 0..batch.num_rows() {
            out.push((
                tx.value(i).to_string(),
                if pal.is_null(i) {
                    None
                } else {
                    Some(pal.value(i))
                },
                w.value(i),
            ));
        }
    }
    out
}

#[test]
fn the_mixed_transcript_is_counted_by_its_abundance_and_measured_by_its_tail() {
    let bam = mixed_bam("mixed_tail");
    let gtf = common::tiny_gtf();
    let dir = scratch("mixed_tail", "dir");
    let _ = std::fs::remove_dir_all(&dir);

    let chain = run_manual_chain_with_universe(&dir, &bam, &gtf, "S1");
    let profile = &chain.profiles[0];

    let rows = profile_rows(profile);
    let tx0: Vec<_> = rows.iter().filter(|(tx, _, _)| tx == "tx0").collect();
    assert_eq!(
        tx0.len(),
        100,
        "every read of the mixed transcript must be quantified: {} rows",
        tx0.len()
    );
    let counted: f64 = tx0.iter().map(|(_, _, w)| *w).sum();
    assert!(
        (counted - 100.0).abs() < 1e-6,
        "the expected count must be 100: got {counted}"
    );
    let with_tail = tx0.iter().filter(|(_, pal, _)| pal.is_some()).count();
    assert_eq!(
        with_tail, 65,
        "exactly the 65 tail-bearing reads carry a tail length"
    );
    assert!(
        tx0.iter().filter(|(_, pal, _)| pal.is_none()).count() == 35,
        "the 35 tail-less reads carry a NULL tail length"
    );

    let agg = telescope::diff::aggregate_profile(profile).expect("aggregate the profile");
    let cell = agg
        .iter()
        .find(|r| r.transcript_id == "tx0")
        .expect("the aggregate must carry the mixed transcript");
    assert!(
        (cell.weight_sum - 100.0).abs() < 1e-6,
        "the abundance column must count every read: got {}",
        cell.weight_sum
    );
    assert!(
        (cell.weight_sum_pal - 65.0).abs() < 1e-6,
        "the tail column must count only the tail-bearing reads: got {}",
        cell.weight_sum_pal
    );
    assert!(
        (cell.weight_sq_sum_pal - 65.0).abs() < 1e-6,
        "the squared tail mass must be the tail-bearing reads': got {}",
        cell.weight_sq_sum_pal
    );

    let neff = cell.weight_sum_pal * cell.weight_sum_pal / cell.weight_sq_sum_pal;
    assert!(
        (neff - 65.0).abs() < 1e-9,
        "the tail track's n_eff must be computed from the 65, not the 100: got {neff}"
    );

    let neff_all = cell.weight_sum * cell.weight_sum / cell.weight_sq_sum;
    assert!(
        (neff_all - 100.0).abs() < 1e-9,
        "the abundance track's n_eff must be computed from the 100: got {neff_all}"
    );

    println!(
        "RECEIPT bc6-mixed (CLI): profile_rows(tx0)={}, expected_count={counted}, \
         rows_with_a_tail={with_tail}, rows_with_a_null_tail={}, \
         weight_sum={}, weight_sum_pal={}, weight_sq_sum_pal={}, \
         n_eff(tail track)={neff}, n_eff(abundance track)={neff_all}",
        tx0.len(),
        tx0.iter().filter(|(_, pal, _)| pal.is_none()).count(),
        cell.weight_sum,
        cell.weight_sum_pal,
        cell.weight_sq_sum_pal,
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn the_pure_no_tail_transcript_is_quantified_and_has_no_tail() {
    let bam = pure_no_tail_bam("pure_no_tail");
    let gtf = common::tiny_gtf();
    let dir = scratch("pure_no_tail", "dir");
    let _ = std::fs::remove_dir_all(&dir);

    let chain = run_manual_chain_with_universe(&dir, &bam, &gtf, "S1");
    let profile = &chain.profiles[0];

    let rows = profile_rows(profile);
    let tx0: Vec<_> = rows.iter().filter(|(tx, _, _)| tx == "tx0").collect();
    assert_eq!(
        tx0.len(),
        100,
        "a transcript whose reads carry no tail must still be quantified: {} rows",
        tx0.len()
    );
    let counted: f64 = tx0.iter().map(|(_, _, w)| *w).sum();
    assert!(
        (counted - 100.0).abs() < 1e-6,
        "the expected count must be the true read count: got {counted}"
    );
    assert!(
        tx0.iter().all(|(_, pal, _)| pal.is_none()),
        "a tail-less read has NO tail length: the profile must carry a null"
    );

    let agg = telescope::diff::aggregate_profile(profile).expect("aggregate the profile");
    let cell = agg
        .iter()
        .find(|r| r.transcript_id == "tx0")
        .expect("the aggregate must carry the no-tail transcript");
    assert!(
        (cell.weight_sum - 100.0).abs() < 1e-6,
        "the abundance column must count every read: got {}",
        cell.weight_sum
    );
    assert_eq!(
        cell.weight_sum_pal, 0.0,
        "the tail column must be ZERO for a transcript with no tail"
    );
    assert_eq!(cell.weight_sq_sum_pal, 0.0);
    assert_eq!(cell.weighted_log2pal_sum, 0.0);
    assert!(
        cell.weighted_log2pal_sum.is_finite(),
        "the tail sum must be finite, never NaN"
    );

    let ledger = dir.join("ledger.json");
    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--emission",
        chain.emission.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        dir.join("again.profile.parquet").to_str().unwrap(),
        "--ledger",
        ledger.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "quant failed: {}", stderr_of(&out));
    let text = std::fs::read_to_string(&ledger).expect("the ledger is written");
    let json = common::read_json(&ledger);
    let no_tail = json
        .get("record_level")
        .and_then(|r| r.get("no_tail_records"))
        .and_then(|r| r.get("no_poly_a_tag"))
        .and_then(|v| v.as_num())
        .expect("the ledger must report the retained no-tail records");
    assert!(
        no_tail >= 100.0,
        "the ledger must report the {no_tail} retained no-tail records: {text}"
    );

    let deficit = json
        .get("record_level")
        .and_then(|r| r.get("deficit"))
        .and_then(|v| v.as_num())
        .expect("the ledger must carry the deficit");
    assert_eq!(deficit, 0.0, "the ledger must close: {text}");

    println!(
        "RECEIPT pure-no-tail (CLI): profile_rows(tx0)={}, expected_count={counted}, \
         rows_with_a_null_tail={}, weight_sum={}, weight_sum_pal={}, \
         weight_sq_sum_pal={}, weighted_log2pal_sum={}, \
         ledger no_poly_a_tag retained={no_tail}, ledger deficit={deficit}",
        tx0.len(),
        tx0.iter().filter(|(_, pal, _)| pal.is_none()).count(),
        cell.weight_sum,
        cell.weight_sum_pal,
        cell.weight_sq_sum_pal,
        cell.weighted_log2pal_sum,
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam);
}
