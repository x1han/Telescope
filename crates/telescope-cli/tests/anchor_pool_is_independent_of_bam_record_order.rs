//! The per-read anchor decision must not depend on BAM record order.
//!
//! The property is about a whole `BAM → pool` run, and the only way to get
//! one is the binary: `extract_anchors_from_bam` is private, and a hand-built
//! `Vec` of `Staged` records is a construction production never performs.
//!
//! The invariant: the read's verdict is a function of the read's own candidate
//! set; the order the records were emitted in is not an input to it.
//!
//! Two kinds of order-dependence are reachable, both covered here: **argmax
//! order** — the anchor flips if `scratch[0]` is chosen by arrival rather than
//! by AS — and **classification order** — the same read filed under a
//! different rejection bucket; `usurp` and `tie` differ ONLY in one AS point,
//! so anchors alone cannot see a swap, and the ledger is compared counter by
//! counter too.

use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{
    synth_record, synth_record_with_flags, synth_record_with_gap, write_synthetic_bam,
    SamFlags as Flags,
};

/// Write the fixture's read set as a BAM, optionally with every record
/// REVERSED. Same records, same reads, same per-read candidate sets.
///
/// ⚠ EVERY "record that passes" CARRIES `AS:i`. The gap is
/// `AS_1 - (AS_2 or 0)` against the shipped `--gap-abs` of 12, so a tag-less
/// record measures 0 and is rejected into `rejected_gap` — such records build
/// an empty pool, which the control assertions below would then report as
/// "the two pools agree" over nothing.
///
/// The read names are deliberately NOT in sorted order, so "reversed" is not
/// also "sorted".
fn write_bam(tag: &str, reversed: bool) -> PathBuf {
    let mut recs = vec![
        // `solo0` / `solo1`: one raw candidate each. The absent runner-up counts
        // as 0, so both clear the 12-point floor on AS alone.
        synth_record("solo0", 0, Some(100), 0, 60),
        synth_record("solo1", 1, Some(50), 0, 60),
        // `wide`: a clean argmax, 20 AS clear of the floor.
        synth_record("wide", 0, Some(100), 0, 60),
        synth_record("wide", 1, Some(80), 0, 60),
        // `cross`: the same shape on a different transcript.
        synth_record("cross", 0, Some(100), 0, 60),
        synth_record("cross", 2, Some(60), 0, 60),
        // `flipped`: THE ANCHOR ORDERING FIXTURE, and the reason this file is
        // not a tautology. The argmax is tx2 (AS 90) and the RUNNER-UP is tx0
        // (AS 60), so the AS-descending sort of `scratch` has to REORDER the
        // candidates — every other read in this fixture has its AS descending
        // in `ref_id` order too, so none of them can tell "sorted by AS" apart
        // from "left as the dedup left it".
        synth_record("flipped", 0, Some(60), 0, 60),
        synth_record("flipped", 2, Some(90), 0, 60),
        // `usurp`: tx0 is the RAW argmax and fails the 3' TES test (reverse
        // strand, 100 bp 5' gap); tx1 is the eligible runner-up.
        synth_record_with_gap("usurp", 0, 100, true, Some(100), 0, 60),
        synth_record("usurp", 1, Some(50), 0, 60),
        // `tie`: the SAME two records with one AS point changed, so the read is a
        // tie rather than a usurpation — this pair separates the two buckets.
        synth_record_with_gap("tie", 0, 100, true, Some(100), 0, 60),
        synth_record("tie", 1, Some(100), 0, 60),
        // `gapfail`: 8 AS apart, below the 12-point floor -> `rejected_gap`
        // (with no `--tx-to-gene` the gene relation is Unknown).
        synth_record("gapfail", 0, Some(100), 0, 60),
        synth_record("gapfail", 1, Some(92), 0, 60),
        // `dedup`: three records on ONE transcript, so the same-`ref_id` dedup
        // collapses them and the multiplicity is what the ledger must account
        // for.
        synth_record("dedup", 2, Some(100), 0, 60),
        synth_record("dedup", 2, Some(100), 0, 60),
        synth_record("dedup", 2, Some(100), 0, 60),
        // Pass-1 buckets, so the fixture also covers the record-level filters
        // that run before a read is ever formed.
        synth_record_with_flags("supp1", 0, Flags::SUPPLEMENTARY, Some(10)),
        synth_record_with_flags("supp2", 1, Flags::SUPPLEMENTARY, Some(10)),
        synth_record_with_flags("unmap", 0, Flags::UNMAPPED, Some(10)),
    ];
    assert_eq!(
        recs.len(),
        20,
        "the fixture is 20 records; the control asserts it"
    );
    if reversed {
        recs.reverse();
    }
    write_synthetic_bam(tag, recs)
}

fn scratch(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "telescope_cli_order_{tag}_{}.{ext}",
        std::process::id()
    ))
}

/// Run `build-anchors` over one BAM. Returns the scratch paths and the ledger
/// the run wrote, so the caller can compare two of them.
fn run_once(tag: &str, reversed: bool) -> (PathBuf, PathBuf, PathBuf, String) {
    let bam = write_bam(tag, reversed);
    let pool = scratch(tag, "parquet");
    let led = scratch(&format!("{tag}_led"), "json");
    let _ = std::fs::remove_file(&pool);
    let _ = std::fs::remove_file(&led);

    let st = Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args([
            "build-anchors",
            "--bam",
            bam.to_str().unwrap(),
            "--sample-id",
            "S1",
            "--output",
            pool.to_str().unwrap(),
            "--ledger",
            led.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn the telescope binary");
    assert!(
        st.status.success(),
        "build-anchors failed on the {} BAM: {}\nstderr:\n{}",
        if reversed { "REVERSED" } else { "forward" },
        st.status,
        String::from_utf8_lossy(&st.stderr)
    );
    let json = std::fs::read_to_string(&led)
        .unwrap_or_else(|e| panic!("--ledger did not produce {}: {e}", led.display()));
    (bam, pool, led, json)
}

fn cleanup(paths: &[&Path]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

/// Read `"key": <int>` out of the ledger. A substring scan, not a JSON parser:
/// the artifact is the thing under test, so parsing it against a schema would
/// only prove it against the schema. Every value here is an integer, so the
/// scan is unambiguous.
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

/// EVERY COUNTER THE LEDGER RECORDS AT ANCHOR LEVEL, so a swap between two
/// rejection buckets is visible even when the emitted anchors agree. Spelled
/// out rather than scanned for, so `parse_errors` and the pool-level
/// `distinct_*` fields are not swept in.
const COUNTERS: [&str; 23] = [
    "total_records",
    "reads_seen",
    "reads_no_eligible_candidate",
    "rejected_tie",
    "rejected_usurpation",
    "rejected_same_gene",
    "rejected_cross_gene",
    "rejected_gap",
    "reads_over_cap",
    "passed",
    "anchors_emitted",
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

/// SAME READ SET, RECORDS REVERSED, IDENTICAL ANCHORS AND IDENTICAL COUNTERS.
#[test]
fn the_emitted_pool_is_identical_when_the_bam_records_are_reversed() {
    let (bam_a, pool_a, led_a, json_a) = run_once("fwd", false);
    let (bam_b, pool_b, led_b, json_b) = run_once("rev", true);

    let bytes_a = std::fs::read(&pool_a).unwrap_or_else(|e| panic!("forward pool unreadable: {e}"));
    let bytes_b =
        std::fs::read(&pool_b).unwrap_or_else(|e| panic!("reversed pool unreadable: {e}"));
    cleanup(&[&bam_a, &pool_a, &led_a, &bam_b, &pool_b, &led_b]);

    // CONTROLS FIRST, so a failure names the layer that broke rather than
    // reporting "the two pools agree" over nothing.
    assert_eq!(
        int_field(&json_a, "total_records"),
        20,
        "CONTROL: 20 records were written, so pass 1 must have seen 20 \
         regardless of their order:\n{json_a}"
    );
    assert_eq!(
        int_field(&json_a, "reads_seen"),
        9,
        "CONTROL: nine distinct reads — solo0, solo1, wide, cross, flipped, \
         usurp, tie, gapfail, dedup. `supp1`/`supp2`/`unmap` die in pass 1 and \
         never become reads:\n{json_a}"
    );
    assert_eq!(
        int_field(&json_a, "anchors_emitted"),
        6,
        "CONTROL: solo0, solo1, wide, cross, flipped and dedup emit; usurp, \
         tie and gapfail are rejected. If this has changed, the record set and \
         the expectations below have drifted apart:\n{json_a}"
    );
    // The read identity closes, so the three rejection buckets above are three
    // different populations and not one number written three times.
    assert_eq!(
        int_field(&json_a, "reads_seen"),
        int_field(&json_a, "rejected_tie")
            + int_field(&json_a, "rejected_usurpation")
            + int_field(&json_a, "rejected_gap")
            + int_field(&json_a, "anchors_emitted"),
        "the read identity must close on this fixture:\n{json_a}"
    );
    assert!(
        bytes_a.len() > 0,
        "CONTROL: the pool must not be empty:\n{json_a}"
    );

    // THE INVARIANT. Byte equality of the emitted pool: same rows, same order,
    // same values.
    assert_eq!(
        bytes_a,
        bytes_b,
        "the emitted anchors must be IDENTICAL for the same read set in \
         reversed BAM order -- {} bytes forward vs {} bytes reversed",
        bytes_a.len(),
        bytes_b.len()
    );

    // ...and the per-read CLASSIFICATION, which comparing anchors alone cannot see.
    for key in COUNTERS {
        assert_eq!(
            int_field(&json_a, key),
            int_field(&json_b, key),
            "counter {key} differs between the forward and reversed BAMs"
        );
    }

    // The three rejection buckets this fixture separates must stay three
    // DIFFERENT numbers, or the per-counter loop above compares two tautologies.
    assert_eq!(int_field(&json_a, "rejected_usurpation"), 1, "\n{json_a}");
    assert_eq!(int_field(&json_a, "rejected_tie"), 1, "\n{json_a}");
    assert_eq!(int_field(&json_a, "rejected_gap"), 1, "\n{json_a}");
    assert_eq!(
        int_field(&json_a, "3p_anchor"),
        2,
        "CONTROL: the disqualified argmax of `usurp` and of `tie`:\n{json_a}"
    );
}
