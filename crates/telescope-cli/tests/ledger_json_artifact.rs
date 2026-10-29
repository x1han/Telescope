//! End-to-end: does `--ledger` actually put a `telescope.filter_ledger`
//! document on disk, with the numbers in it?
//!
//! WHY THIS EXISTS ALONGSIDE `ledger_json.rs`'S UNIT TESTS. The unit tests
//! prove the renderer formats the struct it is handed; this file proves the
//! CLI reaches that call at all — delete the `maybe_write_filter_ledger(...)`
//! line from `main.rs` and every unit test stays green while the flag does
//! nothing.
//!
//! It also pins the two structural rules a future refactor would most
//! plausibly break: the two levels are DIFFERENT POPULATIONS and go to
//! different files; and `--ledger` is OPT-IN, so an absent flag writes
//! nothing. The `deficit != 0` direction is NOT here — it needs unbalanced
//! code, and lives in `ledger_json.rs::anchor_ledger_carries_the_deficit_field`.

use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{
    synth_record, synth_record_no_position, synth_record_no_read_id, synth_record_no_refid,
    synth_record_with_flags, synth_record_with_gap, write_synthetic_bam, SamFlags as Flags,
};

const SCHEMA: &str = "telescope.filter_ledger";

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

/// A BAM that trips all three of the filter paths, alongside two records that
/// pass. The three defect counts are 1, 2 AND 3, NOT 1, 1 AND 1: a mutation
/// that transposes two `render_anchor` arguments leaves every SUM unchanged, so
/// only a per-bucket assertion against an EXPECTED value can see it, and such
/// an assertion is a tautology unless the buckets are distinguishable.
///
/// ⚠ EVERY "record that passes" HERE CARRIES `AS:i = 100`. The gap is
/// `AS_1 - (AS_2 or 0)` against a shipped floor of 12, so a record with no `AS`
/// tag measures 0 and is rejected — these fixtures used to build an empty pool,
/// which the `passed == 2` assertion below would report as a missing record
/// rather than as a floor that had moved.
fn three_defect_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record("ok0", 0, Some(100), 0, 60),
            synth_record("ok1", 1, Some(100), 0, 60),
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
            synth_record("a0", 0, Some(100), 0, 60),
            synth_record("b0", 1, Some(100), 0, 60),
        ],
    )
}

/// A BAM whose three NEW buckets are non-zero and DISTINGUISHABLE: 1 unmapped,
/// 2 supplementary, 3 physically eligible but not selected as the anchor.
///
/// The values are 1 / 2 / 3 for the same reason `three_defect_bam` uses them:
/// equal values would make a transposition in the `format!` invisible.
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

/// A BAM that drives the RUNNER-UP USURPATION branch end to end.
///
/// One read, two records, two transcripts:
///   * `tx0` at AS 100, reverse-strand with a 100 bp 5' gap — its 3' end lands
///     100 bp short of the TES, beyond the slack of 35, so it is the read's RAW
///     argmax and it is physically disqualified.
///   * `tx1` at AS 50, full length, physically eligible — the runner-up.
///
/// The two AS values are 100 and 50 on purpose: DISTINCT is what makes this
/// fixture different from a tie fixture.
fn usurpation_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            synth_record_with_gap("u0", 0, 100, true, Some(100), 0, 60),
            synth_record("u0", 1, Some(50), 0, 60),
        ],
    )
}

/// A BAM whose every record carries MAPQ BELOW the old gate's threshold of 20.
///
/// This is the input `filtered_unique` used to absorb. It exists so the retired
/// slot can be pinned on the path that actually runs the tool: the renderer unit
/// test can only prove the key is still emitted, not that nothing increments it.
fn retired_mapq_gate_bam(tag: &str) -> PathBuf {
    write_synthetic_bam(
        tag,
        vec![
            // MAPQ 3. Used to die at `if mapq < 20`.
            synth_record("lowq", 0, Some(100), 0, 3),
            // MAPQ 11, on a different transcript so the two do not tie.
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
/// Recomputes the conservation identity from the ARTIFACT's own numbers, so a
/// bucket serialized under a name the consumer does not read, or a dropped
/// `+ 1` in `total_records`, fails here.
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

    // The three buckets must be present, non-zero, AND DISTINGUISHABLE — equal
    // expected values would make this loop tautological.
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

    // `parse_errors` is disjoint from `total_records`, so folding it into the
    // RHS would make the identity fail by exactly its value.
    let parse_errors = int_field(&json, "parse_errors");
    assert_eq!(
        parse_errors, 0,
        "synthetic BAMs do not produce parse errors"
    );

    // The unit travels with the number, IN THE ARTIFACT.
    assert!(
        json.contains("\"scope\""),
        "the unit must be in the artifact, not only in a doc comment:\n{json}"
    );
}

/// `build-anchors` emits `anchor_level` and NOT `record_level`: the two ledgers
/// belong to different stages and must never share a document, because summing
/// their `records_entered` values produces a number that looks like a
/// population and is not one.
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
/// nothing — the negative case stops the flag from quietly becoming a default
/// sidecar in a later change, which would put a file next to every profile
/// parquet and make "is there a ledger for this run?" unanswerable.
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

/// THE THREE NEW RECORD BUCKETS REACH THE ARTIFACT, WITH THE COUNTS THE TOOL
/// ACTUALLY PRODUCED.
///
/// `three_defect_bam` proves the older buckets are countable; it cannot reach
/// these three, because every fixture it builds has MAPQ 60 and no flag-only
/// record. Without this the three new keys would be asserted only by the
/// renderer unit test, and the CLI path — the one that would drop a key on the
/// floor — would be uncovered.
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
    // cap never bit: an absent counter and a zero counter are different facts.
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
/// is satisfied by a `build-anchors` that deleted the records; `passed == 2` on
/// its own is satisfied by one that never had the gate. Together they say the
/// gate is gone: the two MAPQ-below-20 records now reach the pool and the slot
/// they would have occupied is 0.
///
/// The fixture uses two DIFFERENT low MAPQs (3 and 11) on two transcripts, so
/// a re-added gate cannot be mistaken for a fixture that happened to sit on
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
        "`unique` is a RETIRED compat slot: the MAPQ < 20 gate was removed \
         because it discarded every secondary candidate and left every \
         surviving read with one candidate, which made strict argmax, the gap \
         floor and the same-gene veto dead code. The key stays in the frozen \
         ledger schema; the value must stay 0.\n{json}"
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

/// `rejected_usurpation` REACHES THE ARTIFACT, WITH THE COUNT THE TOOL
/// ACTUALLY PRODUCED — AND IT IS NOT A TIE.
///
/// The renderer unit test proves the key is formatted; only a real
/// `build-anchors` run proves the CLI reaches the branch at all. Delete
/// `if !scratch[0].2 { ... }` from `anchor_builder.rs` and this goes red at
/// `anchors_emitted == 1` — the runner-up is emitted instead of refused.
///
/// FOUR ASSERTIONS, because any one alone is satisfiable by accident:
///   * `rejected_usurpation == 1`  — the branch fired;
///   * `rejected_tie == 0`         — and it was NOT the tie branch (the two
///                                    candidates are 100 and 50, not equal);
///   * `anchors_emitted == 0`      — the runner-up was not emitted;
///   * `eligible_not_anchor == 1`  — the physically-passing record was
///                                    absorbed, so the record ledger still
///                                    closes at `deficit == 0`.
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

    // The twelve-bucket RECORD identity is untouched by the new read counter.
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

    // And the READ identity closes with usurpation as a destination.
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

/// THE SHIPPED `--gap-abs` DEFAULT IS 12.0, NOT THE 0.0 THIS COMMAND USED TO
/// SHIP.
///
/// The other twenty-odd `anchor_builder` tests deliberately run at 0.0 (see
/// `params()` there), so NOTHING in the library suite can see the CLI default —
/// a change to `default_value_t` would leave every one of them green. This test
/// reads it off the run banner, which prints the resolved value the rule
/// actually receives.
///
/// It also pins the two facts that make the banner trustworthy: the banner says
/// `active` at 12.0, and the effect is real — two reads identical except for a
/// 12 AS gap behave differently under the DEFAULT.
#[test]
fn a_build_anchors_run_without_gap_abs_uses_the_ruled_default_of_twelve() {
    let bam = write_synthetic_bam(
        "gapdefault",
        vec![
            // 100 vs 50 -> raw gap 50, clears a floor of 12.
            synth_record("wide", 0, Some(100), 0, 60),
            synth_record("wide", 1, Some(50), 0, 60),
            // 100 vs 95 -> raw gap 5, BELOW a floor of 12. Under the old 0.0
            // default both reads emitted; exactly one does now.
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

    // The default is not merely printed; it is APPLIED.
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
        "the narrow read fails the 12-point default. No --tx-to-gene was \
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
