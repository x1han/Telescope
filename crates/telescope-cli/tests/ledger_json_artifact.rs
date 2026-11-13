use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{
    synth_record, synth_record_no_position, synth_record_no_read_id, synth_record_no_refid,
    synth_record_with_flags, synth_record_with_gap, write_synthetic_bam, SamFlags as Flags,
};

const SCHEMA: &str = "telescope.filter_ledger";

const ANCHOR_BUCKETS: [&str; 12] = [
    "unique",
    "no_refid",
    "no_position",
    "no_read_id",
    "no_reflen",
    "3p_anchor",
    "coverage",
    "polya_invalid",
    "cap",
    "unmapped",
    "supplementary",
    "eligible_not_anchor",
];

fn three_filter_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record("ok0", 0, Some(100), 0, 60),
            synth_record("ok1", 1, Some(100), 0, 60),
            synth_record_no_refid("norid", Some(2), 0, 60),
            synth_record_no_position(2, Some(3), 0, 60),
            synth_record_no_position(1, Some(4), 0, 60),
            synth_record_no_read_id(0, Some(5), 0, 60),
            synth_record_no_read_id(0, Some(6), 0, 60),
            synth_record_no_read_id(0, Some(7), 0, 60),
        ],
    )
}

fn all_good_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record("a0", 0, Some(100), 0, 60),
            synth_record("b0", 1, Some(100), 0, 60),
        ],
    )
}

fn three_new_buckets_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record_with_flags("u", 0, Flags::UNMAPPED, Some(10)),
            synth_record_with_flags("s1", 0, Flags::SUPPLEMENTARY, Some(10)),
            synth_record_with_flags("s2", 0, Flags::SUPPLEMENTARY, Some(10)),
            synth_record("t0", 0, Some(100), 0, 60),
            synth_record("t0", 1, Some(100), 0, 60),
            synth_record("t0", 2, Some(100), 0, 60),
        ],
    )
}

fn usurpation_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record_with_gap("u0", 0, 100, true, Some(100), 0, 60),
            synth_record("u0", 1, Some(50), 0, 60),
        ],
    )
}

fn retired_mapq_gate_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record("lowq", 0, Some(100), 0, 3),
            synth_record("lowq2", 1, Some(100), 0, 11),
        ],
    )
}

fn scratch(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "telescope_cli_ledger_{tag}_{}.{ext}",
        std::process::id()
    ))
}

fn int_field(json: &str, key: &str) -> i64 {
    let needle = format!("\"{key}\": ");
    let at = json
        .find(&needle)
        .unwrap_or_else(|| panic!("artifact has no field {key:?}:\n{json}"))
        + needle.len();
    let digits: String = json[at..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    assert!(
        !digits.is_empty(),
        "field {key:?} is not an integer:\n{json}"
    );
    digits.parse().expect("captured digits are an i64")
}

fn run(args: &[&str]) -> std::process::Output {
    let st = Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args(args)
        .output()
        .expect("failed to spawn the telescope binary");
    assert!(
        st.status.success(),
        "command failed: {args:?}\nstatus={:?}\nstderr:\n{}",
        st.status,
        String::from_utf8_lossy(&st.stderr)
    );
    st
}

fn cleanup(paths: &[&Path]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("--ledger did not produce {}: {e}", path.display()))
}

#[test]
fn build_anchors_writes_an_anchor_ledger_that_satisfies_the_identity() {
    let bam = three_filter_bam("pos");
    let out = scratch("pos", "parquet");
    let led = scratch("pos", "json");
    run(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        out.to_str().unwrap(),
        "--ledger",
        led.to_str().unwrap(),
    ]);
    let json = read(&led);
    cleanup(&[&bam, &out, &led]);

    assert!(json.contains(SCHEMA), "schema tag missing:\n{json}");
    assert!(
        json.contains("\"anchor_level\""),
        "no anchor_level:\n{json}"
    );

    for (bucket, want) in [("no_refid", 1), ("no_position", 2), ("no_read_id", 3)] {
        assert_eq!(
            int_field(&json, bucket),
            want,
            "bucket {bucket} must be serialized as {want} on this fixture:\n{json}"
        );
    }

    let buckets: i64 = ANCHOR_BUCKETS.iter().map(|k| int_field(&json, k)).sum();
    let total = int_field(&json, "total_records");
    let passed = int_field(&json, "passed");
    let deficit = int_field(&json, "deficit");
    assert_eq!(
        total - buckets - passed,
        deficit,
        "artifact does not satisfy total == sum(filtered) + passed:\n{json}"
    );
    assert_eq!(deficit, 0, "a balanced fixture must report zero:\n{json}");
    assert_eq!(
        total, 8,
        "fixture has 8 records; if this changed, so did the test"
    );
    assert_eq!(
        passed, 2,
        "two records clear every filter; if this changed, so did the test"
    );

    let parse_errors = int_field(&json, "parse_errors");
    assert_eq!(
        parse_errors, 0,
        "synthetic BAMs do not produce parse errors"
    );

    assert!(
        json.contains("\"scope\""),
        "the unit must be in the artifact, not only in a doc comment:\n{json}"
    );
}

#[test]
fn the_anchor_ledger_carries_no_record_level() {
    let bam = all_good_bam("lvl");
    let out = scratch("lvl", "parquet");
    let led = scratch("lvl", "json");
    run(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        out.to_str().unwrap(),
        "--ledger",
        led.to_str().unwrap(),
    ]);
    let json = read(&led);
    cleanup(&[&bam, &out, &led]);

    assert!(
        !json.contains("\"record_level\""),
        "a build-anchors ledger must not carry the record level:\n{json}"
    );
}

#[test]
fn extract_evidence_writes_a_record_ledger_that_satisfies_the_identity() {
    let bam = all_good_bam("ev");
    let led = scratch("ev", "json");
    run(&[
        "extract-evidence",
        "--bam",
        bam.to_str().unwrap(),
        "--ledger",
        led.to_str().unwrap(),
    ]);
    let json = read(&led);
    cleanup(&[&bam, &led]);

    assert!(json.contains(SCHEMA), "schema tag missing:\n{json}");
    assert!(
        json.contains("\"record_level\""),
        "no record_level:\n{json}"
    );
    assert!(
        !json.contains("\"anchor_level\""),
        "an extract-evidence ledger must not carry the anchor level:\n{json}"
    );

    let parts: i64 = [
        "unmapped",
        "supplementary",
        "no_poly_a_tag",
        "invalid_poly_a",
        "no_span",
        "empty_span",
        "tes_window",
    ]
    .iter()
    .map(|k| int_field(&json, k))
    .sum();
    let dropped_total = int_field(&json, "dropped_total");
    assert_eq!(
        parts, dropped_total,
        "dropped_total disagrees with its own parts:\n{json}"
    );

    let deficit = int_field(&json, "deficit");
    assert_eq!(
        int_field(&json, "records_entered")
            - dropped_total
            - int_field(&json, "records_with_candidates"),
        deficit,
        "artifact does not satisfy entered == dropped + reached:\n{json}"
    );
    assert_eq!(deficit, 0, "a balanced fixture must report zero:\n{json}");
}

#[test]
fn no_ledger_flag_writes_no_ledger() {
    let bam = all_good_bam("optin");
    let out = scratch("optin", "parquet");
    let led = scratch("optin", "json");
    run(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        out.to_str().unwrap(),
    ]);
    cleanup(&[&bam, &out]);

    assert!(
        !led.exists(),
        "--ledger is opt-in; a run without it must not create {}",
        led.display()
    );
}

#[test]
fn build_anchors_serializes_the_three_new_record_buckets() {
    let bam = three_new_buckets_bam("newbuckets");
    let out = scratch("newbuckets", "parquet");
    let led = scratch("newbuckets", "json");
    run(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        out.to_str().unwrap(),
        "--ledger",
        led.to_str().unwrap(),
    ]);
    let json = read(&led);
    cleanup(&[&bam, &out, &led]);

    for (bucket, want) in [
        ("unmapped", 1),
        ("supplementary", 2),
        ("eligible_not_anchor", 3),
    ] {
        assert_eq!(
            int_field(&json, bucket),
            want,
            "bucket {bucket} must be serialized as {want} on this fixture:\n{json}"
        );
    }

    let buckets: i64 = ANCHOR_BUCKETS.iter().map(|k| int_field(&json, k)).sum();
    assert_eq!(
        int_field(&json, "total_records") - buckets - int_field(&json, "passed"),
        int_field(&json, "deficit"),
        "the twelve-bucket identity must still hold:\n{json}"
    );
    assert_eq!(
        int_field(&json, "deficit"),
        0,
        "a balanced fixture must report zero:\n{json}"
    );
    assert_eq!(
        int_field(&json, "passed"),
        0,
        "the tie means no read produces an anchor; if this changed, so did the \
         fixture"
    );

    assert_eq!(
        int_field(&json, "reads_seen"),
        1,
        "only the tie read is a read"
    );
    assert_eq!(int_field(&json, "rejected_tie"), 1);
    assert_eq!(int_field(&json, "anchors_emitted"), 0);

    assert_eq!(
        int_field(&json, "reads_over_cap"),
        0,
        "one read on this fixture and the cap is 200, so nothing is over it"
    );
    assert!(
        json.contains("\"reads\""),
        "the read-denominated block must be present and separately named:\n{json}"
    );
}

#[test]
fn build_anchors_reports_a_retired_unique_bucket_of_zero() {
    let bam = retired_mapq_gate_bam("retiredmapq");
    let out = scratch("retiredmapq", "parquet");
    let led = scratch("retiredmapq", "json");
    run(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        out.to_str().unwrap(),
        "--ledger",
        led.to_str().unwrap(),
    ]);
    let json = read(&led);
    cleanup(&[&bam, &out, &led]);

    assert_eq!(
        int_field(&json, "unique"),
        0,
        "`unique` is a compatibility ledger field and must remain 0.\n{json}"
    );
    assert_eq!(
        int_field(&json, "passed"),
        2,
        "BOTH records carry MAPQ < 20 and both must now reach the pool — this \
         is what makes the assertion above meaningful rather than vacuous.\n{json}"
    );
    assert_eq!(int_field(&json, "total_records"), 2);

    let buckets: i64 = ANCHOR_BUCKETS.iter().map(|k| int_field(&json, k)).sum();
    assert_eq!(
        int_field(&json, "total_records") - buckets - int_field(&json, "passed"),
        int_field(&json, "deficit"),
        "the twelve-bucket identity must still hold:\n{json}"
    );
    assert_eq!(
        int_field(&json, "deficit"),
        0,
        "a balanced fixture:\n{json}"
    );

    assert_eq!(int_field(&json, "reads_seen"), 2);
    assert_eq!(int_field(&json, "anchors_emitted"), 2);
}

#[test]
fn build_anchors_serializes_rejected_usurpation_and_refuses_the_runner_up() {
    let bam = usurpation_bam("usurp");
    let out = scratch("usurp", "parquet");
    let led = scratch("usurp", "json");
    run(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        out.to_str().unwrap(),
        "--ledger",
        led.to_str().unwrap(),
    ]);
    let json = read(&led);
    cleanup(&[&bam, &out, &led]);

    assert_eq!(
        int_field(&json, "rejected_usurpation"),
        1,
        "the read's RAW argmax (tx0, AS 100) is 3'-truncated and ineligible; \
         only the lower-AS tx1 survives, and that is a refusal, not an anchor.\n{json}"
    );
    assert_eq!(
        int_field(&json, "rejected_tie"),
        0,
        "100 vs 50 is not a tie — if this is 1 the branches were swapped.\n{json}"
    );
    assert_eq!(
        int_field(&json, "anchors_emitted"),
        0,
        "the runner-up must not inherit the read.\n{json}"
    );
    assert_eq!(
        int_field(&json, "passed"),
        0,
        "no record became an anchor.\n{json}"
    );
    assert_eq!(
        int_field(&json, "eligible_not_anchor"),
        1,
        "tx1 WAS physically eligible and is still not emitted, so its record \
         must be absorbed here or the record ledger leaks.\n{json}"
    );
    assert_eq!(
        int_field(&json, "3p_anchor"),
        1,
        "the disqualified tx0 record went to its own bucket.\n{json}"
    );

    let buckets: i64 = ANCHOR_BUCKETS.iter().map(|k| int_field(&json, k)).sum();
    assert_eq!(
        int_field(&json, "total_records") - buckets - int_field(&json, "passed"),
        int_field(&json, "deficit"),
        "the twelve-bucket identity must still hold:\n{json}"
    );
    assert_eq!(
        int_field(&json, "deficit"),
        0,
        "a balanced fixture:\n{json}"
    );

    assert_eq!(
        int_field(&json, "reads_seen"),
        int_field(&json, "anchors_emitted")
            + int_field(&json, "rejected_tie")
            + int_field(&json, "rejected_usurpation")
            + int_field(&json, "rejected_gap")
            + int_field(&json, "rejected_same_gene")
            + int_field(&json, "rejected_cross_gene")
            + int_field(&json, "reads_no_eligible_candidate")
            + int_field(&json, "reads_over_cap"),
        "the READS identity does not close on the rendered artifact:\n{json}"
    );
    assert_eq!(
        int_field(&json, "reads_seen"),
        1,
        "the fixture holds ONE read"
    );
    assert!(
        json.contains("rejected_usurpation"),
        "the key must be present in the artifact itself:\n{json}"
    );
}

#[test]
fn a_build_anchors_run_without_gap_abs_uses_the_default_of_twelve() {
    let bam = write_synthetic_bam(
        "gapdefault",
        vec![
            synth_record("wide", 0, Some(100), 0, 60),
            synth_record("wide", 1, Some(50), 0, 60),
            synth_record("narrow", 0, Some(100), 0, 60),
            synth_record("narrow", 1, Some(95), 0, 60),
        ],
    );
    let out = scratch("gapdefault", "parquet");
    let led = scratch("gapdefault", "json");
    let st = Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args([
            "build-anchors",
            "--bam",
            bam.to_str().unwrap(),
            "--sample-id",
            "S1",
            "--output",
            out.to_str().unwrap(),
            "--ledger",
            led.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn the telescope binary");
    assert!(
        st.status.success(),
        "build-anchors failed: {}\nstderr:\n{}",
        st.status,
        String::from_utf8_lossy(&st.stderr)
    );
    let stderr = String::from_utf8_lossy(&st.stderr).into_owned();
    let json = read(&led);
    cleanup(&[&bam, &out, &led]);

    assert!(
        stderr.contains("gap_abs:    12 (active)"),
        "the SHIPPED default must be 12.0 and the banner must say `active`. \
         A banner reading `gap_abs: 0 (OFF ...)` is the disabled default and \
         every gap-based rejection on this data would silently disappear.\n\
         stderr:\n{stderr}"
    );

    assert_eq!(int_field(&json, "reads_seen"), 2);
    assert_eq!(
        int_field(&json, "anchors_emitted"),
        1,
        "the wide read (gap 50) emits; the narrow one (gap 5) is refused by the \
         12-point default.\n{json}"
    );
    assert_eq!(
        int_field(&json, "rejected_gap"),
        1,
        "the narrow read fails the 12-point default. No --gtf was \
         passed, so the gene relation is Unknown and the gap failure is \
         classified into `rejected_gap` rather than same/cross.\n{json}"
    );
    assert_eq!(
        int_field(&json, "rejected_same_gene"),
        0,
        "no map was supplied, so `same_gene` must never be invented.\n{json}"
    );
    assert_eq!(
        int_field(&json, "rejected_usurpation"),
        0,
        "both candidates of both reads are physically eligible.\n{json}"
    );
}
