//! End-to-end: does `--ledger` actually put a `telescope.filter_ledger/v1`
//! document on disk, with the numbers in it?
//!
//! WHY THIS EXISTS ALONGSIDE `ledger_json.rs`'S UNIT TESTS. The unit tests
//! prove the renderer formats the struct it is handed. This file proves the
//! CLI reaches that call at all. Those are different failures: delete the
//! `maybe_write_filter_ledger(...)` line from `main.rs`, add a `ledger` field
//! to a subcommand's struct, and forget to wire it — every unit test in
//! `ledger_json.rs` stays green and the flag silently does nothing.
//!
//! It also pins the two structural rules that a future refactor would most
//! plausibly break, both from the Round 65 ruling:
//!
//!   * the two levels are DIFFERENT POPULATIONS and go to different files;
//!   * `--ledger` is OPT-IN, so an absent flag writes nothing.
//!
//! The `deficit != 0` direction is NOT here — it cannot be: producing an
//! unbalanced ledger through the real binary requires unbalanced code. It
//! lives in `ledger_json.rs::anchor_ledger_carries_the_deficit_field`, which
//! hands the renderer a deliberately leaking `AnchorStats`.

use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{
    synth_record, synth_record_no_position, synth_record_no_read_id, synth_record_no_refid,
    synth_record_with_flags, write_synthetic_bam, SamFlags as Flags,
};

const SCHEMA: &str = "telescope.filter_ledger/v1";

/// The twelve anchor-level filter buckets. Every one is read back out of the
/// artifact and summed, so dropping a key from the `format!` cannot survive.
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

/// A BAM that trips all three of the filter paths Ticket 2B-Fix proved
/// reachable, alongside two records that pass. Without the three defects the
/// ledger's `deficit` would still be 0, and the test would pass without ever
/// proving those buckets are countable in the artifact — which is exactly the
/// `deficit == 0` ambiguity the whole ticket exists to remove.
///
/// **THE THREE DEFECT COUNTS ARE 1, 2 AND 3, NOT 1, 1 AND 1, AND THAT IS THE
/// POINT.** The first version of this fixture used one record per defect. A
/// mutation that transposes the `no_position` and `no_read_id` arguments in
/// `render_anchor` leaves every SUM unchanged, so it is invisible to the
/// identity check — the only thing that can see it is a per-bucket assertion
/// comparing each bucket against an EXPECTED value. With equal fixtures that
/// assertion is a tautology, and the mutation survived (`mutate_ledger_json.py`
/// L3, 2026-10-13). A per-bucket assertion is worth nothing unless the fixture
/// makes the buckets distinguishable from each other.
fn three_defect_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record("ok0", 0, Some(0), 0, 60),
            synth_record("ok1", 1, Some(1), 0, 60),
            // 1 x no_refid
            synth_record_no_refid("norid", Some(2), 0, 60),
            // 2 x no_position
            synth_record_no_position(2, Some(3), 0, 60),
            synth_record_no_position(1, Some(4), 0, 60),
            // 3 x no_read_id
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
            synth_record("a0", 0, Some(0), 0, 60),
            synth_record("b0", 1, Some(1), 0, 60),
        ],
    )
}

/// A BAM whose three NEW buckets are non-zero and DISTINGUISHABLE: 1 unmapped,
/// 2 supplementary, 3 physically eligible but not selected as the anchor.
///
/// The values are 1 / 2 / 3 for the same reason `three_defect_bam` uses them:
/// equal values would make a transposition in the `format!` invisible, because
/// it leaves the SUM unchanged and only a per-bucket expectation can see it.
fn three_new_buckets_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            // 1 x unmapped
            synth_record_with_flags("u", 0, Flags::UNMAPPED, Some(10)),
            // 2 x supplementary
            synth_record_with_flags("s1", 0, Flags::SUPPLEMENTARY, Some(10)),
            synth_record_with_flags("s2", 0, Flags::SUPPLEMENTARY, Some(10)),
            // 3 records of ONE read that is physically eligible twice over but
            // emits no anchor: an exact AS tie between two transcripts. That
            // is 3 records in, 3 records into `eligible_not_anchor`.
            synth_record("t0", 0, Some(100), 0, 60),
            synth_record("t0", 1, Some(100), 0, 60),
            synth_record("t0", 2, Some(100), 0, 60),
        ],
    )
}

/// A BAM whose every record carries MAPQ BELOW the old gate's threshold of 20.
///
/// This is the input `filtered_unique` used to absorb. It exists so the retired
/// slot can be pinned on the path that actually runs the tool: the renderer unit
/// test can only prove the key is still emitted, not that nothing increments it,
/// because `AnchorStats::default()` is 0 whether the rule is enforced or not.
fn retired_mapq_gate_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            // MAPQ 3. Used to die at `if mapq < 20`.
            synth_record("lowq", 0, Some(0), 0, 3),
            // MAPQ 11, on a different transcript so the two do not tie.
            synth_record("lowq2", 1, Some(1), 0, 11),
        ],
    )
}

fn scratch(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "telescope_cli_ledger_{tag}_{}.{ext}",
        std::process::id()
    ))
}

/// Read `"key": <int>` out of the artifact. A substring scan, not a JSON
/// parser: the artifact is the thing under test, so parsing it against a
/// schema would only prove it against the schema. Every value here is an
/// integer, so the scan is unambiguous.
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

/// `build-anchors --ledger`: the anchor-level document.
///
/// Recomputes the conservation identity from the ARTIFACT's own numbers. This
/// is the assertion that would fail if a bucket were serialized under a name
/// the consumer does not read, or if a `+ 1` were dropped from `total_records`.
#[test]
fn build_anchors_writes_an_anchor_ledger_that_satisfies_the_identity() {
    let bam = three_defect_bam("pos");
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

    // The three buckets must be present, non-zero, AND DISTINGUISHABLE. Equal
    // expected values would make this loop tautological -- see the doc comment
    // on `three_defect_bam`.
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

    // Decision 2: `parse_errors` is disjoint from `total_records`, so folding
    // it into the RHS would make the identity fail by exactly its value. It
    // must be present and the identity must hold without it.
    let parse_errors = int_field(&json, "parse_errors");
    assert_eq!(
        parse_errors, 0,
        "synthetic BAMs do not produce parse errors"
    );

    // Decision 4: the unit travels with the number, IN THE ARTIFACT.
    assert!(
        json.contains("\"scope\""),
        "the unit must be in the artifact, not only in a doc comment:\n{json}"
    );
}

/// `build-anchors` emits `anchor_level` and NOT `record_level`. Round 65 §2.2:
/// the two ledgers belong to different stages and must never share a document,
/// because summing their `records_entered` values produces a number that looks
/// like a population and is not one.
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

/// `extract-evidence --ledger`: the record-level document, same identity.
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

/// OPT-IN CONTRACT. `--ledger` defaults to `None`, so a run without it writes
/// nothing. The negative case is what stops the flag from quietly becoming a
/// default sidecar in a later change — which would put a file next to every
/// profile parquet and make "is there a ledger for this run?" unanswerable.
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

/// THE THREE BUCKETS ADDED 2026-10-18 REACH THE ARTIFACT, WITH THE COUNTS THE
/// TOOL ACTUALLY PRODUCED.
///
/// `three_defect_bam` proves the Ticket-2A buckets are countable; it cannot
/// reach these three, because every fixture it builds has MAPQ 60 and no
/// flag-only record. Without this the three new keys would be asserted only by
/// the renderer unit test, and the CLI path — the one that would drop a key on
/// the floor — would be uncovered.
///
/// The per-bucket expectations are 1 / 2 / 3 on purpose; see the doc comment on
/// `three_new_buckets_bam`.
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

    // The read block rides along in the same document and is labelled as reads.
    assert_eq!(
        int_field(&json, "reads_seen"),
        1,
        "only the tie read is a read"
    );
    assert_eq!(int_field(&json, "rejected_tie"), 1);
    assert_eq!(int_field(&json, "anchors_emitted"), 0);
    // `reads_over_cap` is on the read block and must be serialized even when the
    // cap never bit: an absent counter and a zero counter are different facts,
    // and a consumer that reads it with `int_field` must not have to distinguish
    // them by guessing.
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

/// THE RETIRED `unique` SLOT IS REPORTED AS ZERO ON A REAL RUN, AND THE RECORDS
/// IT USED TO ABSORB ARE IN THE POOL INSTEAD.
///
/// TWO ASSERTIONS, AND EITHER ALONE IS WORTH NOTHING. `unique == 0` on its own
/// is satisfied by a `build-anchors` that deleted the records. `passed == 2` on
/// its own is satisfied by a `build-anchors` that never had the gate. Together
/// they say the gate is gone: the two MAPQ-below-20 records were dropped into
/// `unique` before R139 and now reach the pool, and the slot they would have
/// occupied is 0.
///
/// The fixture uses two DIFFERENT low MAPQs (3 and 11) on two transcripts so a
/// re-added gate could not be mistaken for a fixture that happened to sit on
/// the boundary, and so the two records cannot tie into one read.
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
        "`unique` is a RETIRED compat slot: the MAPQ < 20 gate was removed in \
         R139 because it discarded every secondary on HC2 and left every \
         surviving read with one candidate, which made strict argmax, the gap \
         floor and the same-gene veto dead code. The key stays in the frozen v1 \
         schema; the value must stay 0.\n{json}"
    );
    assert_eq!(
        int_field(&json, "passed"),
        2,
        "BOTH records carry MAPQ < 20 and both must now reach the pool — this \
         is what makes the assertion above meaningful rather than vacuous.\n{json}"
    );
    assert_eq!(int_field(&json, "total_records"), 2);

    // The twelve-bucket identity must still close with the slot in it.
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

    // And both reads reached the per-read stage, as anchors.
    assert_eq!(int_field(&json, "reads_seen"), 2);
    assert_eq!(int_field(&json, "anchors_emitted"), 2);
}
