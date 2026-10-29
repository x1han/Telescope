//! Schema of the machine-readable filter ledger.
//!
//! Two deliberate deviations from the original schema text are recorded here:
//! `anchor_level.filtered` carries NINE keys, not six, because the code has
//! nine filter buckets; and each command emits only the level it has, so the
//! absent level is OMITTED rather than nulled.
//!
//! JSON is hand-written with `format!`: `telescope-core` has no serde_json
//! dependency and this adds none. Every value is an integer or a fixed
//! literal, so there is no escaping to get wrong.

use std::io::Write;
use std::path::Path;

use anyhow::Context as _;
use telescope_core::anchor_builder::AnchorStats;
use telescope_core::evidence::FilterCounts;

/// The schema tag. Change it only together with the artifact format it names.
const SCHEMA: &str = "telescope.filter_ledger";

/// The unit travels WITH the number, so it survives being quoted out of the log.
const RECORD_SCOPE: &str = "RECORDS, not reads, not molecules";

/// `parse_errors` is disjoint from `total_records` (it is incremented before
/// it), so a consumer that folds it into the sum reads `deficit = -1` and
/// blames a bias; the pool-level caveat for `distinct_genes` /
/// `distinct_transcripts` rides in the same string.
const ANCHOR_SCOPE: &str = "RECORDS read from each anchor BAM; parse_errors is \
a DISJOINT population; distinct_genes / distinct_transcripts are POOL-\
level, assigned by build_anchor_pool and NOT summed over samples";

/// The `reads` sub-block's own unit, carried in the artifact for the same
/// reason [`RECORD_SCOPE`] exists. These counts are READS and are deliberately
/// NOT summed into `deficit`, which is a RECORDS identity.
const READ_SCOPE: &str = "READS, not records; NOT part of `deficit`, which is a \
RECORDS identity. Every read contributes at most one anchor, so \
reads_seen == anchors_emitted + rejected_tie + rejected_usurpation + \
rejected_gap + rejected_same_gene + rejected_cross_gene + \
reads_no_eligible_candidate + reads_over_cap. \
`gene_veto_disabled` is DELIBERATELY ABSENT from that identity: it counts reads \
that CLEARED the gap with the gene relation unavailable (no map supplied) and \
were therefore EMITTED, so counting it as a destination would double-count them \
against `anchors_emitted`. `rejected_same_gene` and `rejected_cross_gene` are \
the same-gene / cross-gene classifications of a read the GAP rejected; there is \
no unconditional same-gene veto, so no read is refused merely for competing with \
a sibling isoform. `rejected_usurpation` is a read whose best RAW alignment \
(measured BEFORE the physical filters) failed one of them, so only a LOWER-AS \
homolog survived; the read is refused rather than attributed to that homolog.";

/// Render the `record_level` half: the evidence-stage ledger.
///
/// Consumes `FilterCounts`, which covers extract-evidence, quant and
/// resolve-evidence. Its identity is
/// `records_entered == dropped_total + records_with_candidates`, and
/// `deficit` is `FilterCounts::ledger_deficit()`.
pub fn render_filter(s: &FilterCounts) -> String {
    format!(
        "{{\n\
         \"schema\": \"{SCHEMA}\",\n\
         \"record_level\": {{\n\
         \"scope\": \"{RECORD_SCOPE}\",\n\
         \"records_entered\": {},\n\
         \"dropped\": {{\n\
         \"unmapped\": {},\n\
         \"supplementary\": {},\n\
         \"no_poly_a_tag\": {},\n\
         \"invalid_poly_a\": {},\n\
         \"no_span\": {},\n\
         \"empty_span\": {},\n\
         \"tes_window\": {}\n\
         }},\n\
         \"dropped_total\": {},\n\
         \"records_with_candidates\": {},\n\
         \"deficit\": {}\n\
         }}\n\
         }}\n",
        s.records_entered,
        s.dropped_unmapped,
        s.dropped_supplementary,
        s.dropped_no_poly_a_tag,
        s.dropped_invalid_poly_a,
        s.dropped_no_span,
        s.dropped_empty_span,
        s.dropped_tes_window,
        s.dropped_total(),
        s.records_with_candidates,
        s.ledger_deficit(),
    )
}

/// Render the `anchor_level` half: the anchor-stage ledger.
///
/// Consumes `AnchorStats`, which only `build-anchors` produces. Its identity
/// is `total_records == sum(filtered_*) + passed`.
pub fn render_anchor(s: &AnchorStats) -> String {
    format!(
        "{{\n\
         \"schema\": \"{SCHEMA}\",\n\
         \"anchor_level\": {{\n\
         \"scope\": \"{ANCHOR_SCOPE}\",\n\
         \"samples\": {},\n\
         \"total_records\": {},\n\
         \"parse_errors\": {},\n\
         \"filtered\": {{\n\
         \"unique\": {},\n\
         \"no_refid\": {},\n\
         \"no_position\": {},\n\
         \"no_read_id\": {},\n\
         \"no_reflen\": {},\n\
         \"3p_anchor\": {},\n\
         \"coverage\": {},\n\
         \"polya_invalid\": {},\n\
         \"cap\": {},\n\
         \"unmapped\": {},\n\
         \"supplementary\": {},\n\
         \"eligible_not_anchor\": {}\n\
         }},\n\
         \"passed\": {},\n\
         \"deficit\": {},\n\
         \"distinct_genes\": {},\n\
         \"distinct_transcripts\": {},\n\
         \"reads\": {{\n\
         \"scope\": \"{READ_SCOPE}\",\n\
         \"reads_seen\": {},\n\
         \"anchors_emitted\": {},\n\
         \"rejected_tie\": {},\n\
         \"rejected_gap\": {},\n\
         \"rejected_same_gene\": {},\n\
         \"rejected_cross_gene\": {},\n\
         \"gene_veto_disabled\": {},\n\
         \"same_gene_unknown\": {},\n\
         \"reads_no_eligible_candidate\": {},\n\
         \"reads_over_cap\": {},\n\
         \"rejected_usurpation\": {}\n\
         }}\n\
         }}\n\
         }}\n",
        s.samples,
        s.total_records,
        s.parse_errors,
        s.filtered_unique,
        s.filtered_no_refid,
        s.filtered_no_position,
        s.filtered_no_read_id,
        s.filtered_no_reflen,
        s.filtered_3p_anchor,
        s.filtered_coverage,
        s.filtered_polya_invalid,
        s.filtered_cap,
        s.filtered_unmapped,
        s.filtered_supplementary,
        s.filtered_eligible_not_anchor,
        s.passed,
        s.unaccounted(),
        s.distinct_genes,
        s.distinct_transcripts,
        s.reads_seen,
        s.anchors_emitted,
        s.rejected_tie,
        s.rejected_gap,
        s.rejected_same_gene,
        s.rejected_cross_gene,
        s.gene_veto_disabled,
        s.same_gene_unknown,
        s.reads_no_eligible_candidate,
        s.reads_over_cap,
        s.rejected_usurpation,
    )
}

/// Write the evidence-stage ledger to `path`.
pub fn write_filter(path: &Path, s: &FilterCounts) -> anyhow::Result<()> {
    write_raw(path, &render_filter(s))
}

/// Write the anchor-stage ledger to `path`.
pub fn write_anchor(path: &Path, s: &AnchorStats) -> anyhow::Result<()> {
    write_raw(path, &render_anchor(s))
}

fn write_raw(path: &Path, body: &str) -> anyhow::Result<()> {
    let mut f = std::fs::File::create(path)
        .with_context(|| format!("failed to create ledger {}", path.display()))?;
    f.write_all(body.as_bytes())
        .with_context(|| format!("failed to write ledger {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pull `"key": <int>` out of the rendered text.
    ///
    /// Deliberately a substring scan and not a JSON parser: parsing it with
    /// serde_json would prove it against a schema, not against the identity the
    /// buckets have to satisfy. Every value in the artifact is an integer or a
    /// fixed literal, so `"key": ` followed by an optional `-` and digits is
    /// unambiguous.
    fn int_field(json: &str, key: &str) -> i64 {
        let needle = format!("\"{key}\": ");
        let at = json
            .find(&needle)
            .unwrap_or_else(|| panic!("no field {key:?} in:\n{json}"))
            + needle.len();
        let digits: String = json[at..]
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        assert!(
            !digits.is_empty(),
            "field {key:?} is not an integer in:\n{json}"
        );
        digits.parse().expect("captured digits are an i64")
    }

    /// The twelve anchor-level filter buckets, in the order `report()` emits
    /// them. Two tests walk this list; if a thirteenth bucket is ever added,
    /// both go red rather than silently comparing a 12-term sum against a
    /// 13-term total.
    ///
    /// **`unique` IS A RETIRED COMPAT SLOT AND MUST STAY 0.** The MAPQ < 20 gate
    /// it counted is gone, so nothing increments it, but the key stays:
    /// `telescope.filter_ledger` is a published schema, a downstream
    /// consumer reads this name, and `bucket_sum` sums all twelve.
    ///
    /// Trap: a retired counter is the one kind of field no test here catches
    /// being re-added — `bucket_sum` is happy with any value that keeps the
    /// identity closed. Its value is pinned by
    /// `retired_unique_bucket_is_reported_as_zero` below and
    /// `build_anchors_reports_a_retired_unique_bucket_of_zero` in
    /// `tests/ledger_json_artifact.rs`; `balanced_stats` pins 40 only to keep
    /// its own twelve-term sum closed and is not production evidence.
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

    /// 40+5+3+2+1+4+6+2+1+10+3+7 == 84, plus `passed` 26, is exactly
    /// `total_records` 110.
    ///
    /// Trap: every bucket carries a DISTINCT value. A bucket left at 0 makes
    /// each assertion over this list a tautology on that term — transposing
    /// `unmapped` and `supplementary` in the `format!` would leave the SUM
    /// unchanged and turn green. `rejected_usurpation` carries 2 for the same
    /// reason: it is a READ counter outside `bucket_sum`, and a zero there
    /// would make its term in the read identity arithmetically inert.
    fn balanced_stats() -> AnchorStats {
        AnchorStats {
            samples: 3,
            total_records: 110,
            parse_errors: 7,
            filtered_unique: 40,
            filtered_no_refid: 5,
            filtered_no_position: 3,
            filtered_no_read_id: 2,
            filtered_no_reflen: 1,
            filtered_3p_anchor: 4,
            filtered_coverage: 6,
            filtered_polya_invalid: 2,
            filtered_cap: 1,
            filtered_unmapped: 10,
            filtered_supplementary: 3,
            filtered_eligible_not_anchor: 7,
            passed: 26,
            // A closed read block, so the fixture is internally coherent in
            // BOTH denominations: 45 reads, 26 anchors, 10 rejected ties, 2
            // runner-up usurpations, 4 with no eligible candidate, 3 over the
            // per-transcript cap.
            reads_seen: 45,
            anchors_emitted: 26,
            rejected_tie: 10,
            rejected_usurpation: 2,
            rejected_gap: 0,
            rejected_same_gene: 0,
            rejected_cross_gene: 0,
            gene_veto_disabled: 0,
            same_gene_unknown: 0,
            reads_no_eligible_candidate: 4,
            reads_over_cap: 3,
            distinct_genes: 12,
            distinct_transcripts: 30,
        }
    }

    fn bucket_sum(json: &str) -> i64 {
        ANCHOR_BUCKETS.iter().map(|k| int_field(json, k)).sum()
    }

    fn filter_counts() -> FilterCounts {
        FilterCounts {
            records_entered: 1000,
            dropped_unmapped: 100,
            dropped_supplementary: 10,
            dropped_no_poly_a_tag: 20,
            dropped_invalid_poly_a: 5,
            dropped_no_span: 3,
            dropped_empty_span: 2,
            dropped_tes_window: 60,
            records_with_candidates: 800,
        }
    }

    /// The artifact must carry the deficit as a FIELD; hardcoding it to 0
    /// turns this red on an unbalanced fixture.
    #[test]
    fn anchor_ledger_carries_the_deficit_field() {
        let s = balanced_stats();
        let json = render_anchor(&s);
        assert_eq!(int_field(&json, "deficit"), 0);

        // An UNBALANCED ledger must report a non-zero deficit, not a clamped 0.
        let mut leaking = balanced_stats();
        leaking.passed -= 2;
        assert_eq!(int_field(&render_anchor(&leaking), "deficit"), 2);
    }

    /// `scope` must be present: the unit used to live only in a comment and
    /// died the moment a number was quoted out.
    #[test]
    fn both_levels_carry_a_scope_string() {
        let a = render_anchor(&balanced_stats());
        assert!(a.contains(ANCHOR_SCOPE), "anchor_level lost its scope");
        let f = render_filter(&filter_counts());
        assert!(f.contains(RECORD_SCOPE), "record_level lost its scope");
    }

    /// The identity must hold when recomputed from the artifact's OWN numbers,
    /// not from the struct that produced it — this is the check that catches a
    /// key being dropped from the `format!`, which no other test would.
    #[test]
    fn anchor_identity_recomputes_from_the_rendered_json() {
        let s = balanced_stats();
        let json = render_anchor(&s);
        let buckets = bucket_sum(&json);
        let total = int_field(&json, "total_records");
        let passed = int_field(&json, "passed");
        let deficit = int_field(&json, "deficit");
        assert_eq!(
            total - buckets - passed,
            deficit,
            "the rendered numbers do not satisfy the identity they claim:\n{json}"
        );
        assert_eq!(deficit, 0);
    }

    /// Same for the evidence stage.
    #[test]
    fn filter_identity_recomputes_from_the_rendered_json() {
        let f = filter_counts();
        let json = render_filter(&f);
        let buckets: i64 = [
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
        let total = int_field(&json, "records_entered");
        let reached = int_field(&json, "records_with_candidates");
        let dropped = int_field(&json, "dropped_total");
        let deficit = int_field(&json, "deficit");
        assert_eq!(buckets, dropped, "dropped_total disagrees with its parts");
        assert_eq!(
            total - dropped - reached,
            deficit,
            "the rendered numbers do not satisfy the identity they claim:\n{json}"
        );
        assert_eq!(deficit, 0);
    }

    /// `parse_errors` is aggregated but must stay OUT of the identity. It is
    /// deliberately non-zero in the fixture, so folding it in turns this red.
    #[test]
    fn parse_errors_is_reported_but_excluded_from_the_identity() {
        let s = balanced_stats();
        let json = render_anchor(&s);
        assert_eq!(int_field(&json, "parse_errors"), 7, "not serialized");
        // Folding parse_errors into the RHS would give sum 117 and deficit -7,
        // which reads like a bias parameter rather than a bookkeeping mistake.
        assert_eq!(bucket_sum(&json) + int_field(&json, "passed"), 110);
        assert_eq!(s.unaccounted(), 0);
    }

    /// The two levels must not share a document, and the absent one is omitted
    /// rather than nulled.
    #[test]
    fn each_command_emits_only_its_own_level() {
        let a = render_anchor(&balanced_stats());
        assert!(a.contains("\"anchor_level\""));
        assert!(!a.contains("\"record_level\""));

        let f = render_filter(&filter_counts());
        assert!(f.contains("\"record_level\""));
        assert!(!f.contains("\"anchor_level\""));
    }

    /// The schema tag is pinned so a consumer can detect the artifact version.
    #[test]
    fn schema_tag_is_pinned() {
        assert!(render_anchor(&balanced_stats()).contains(SCHEMA));
        assert!(render_filter(&filter_counts()).contains(SCHEMA));
    }

    /// The `reads` block is a DIFFERENT DENOMINATION and says so in the
    /// artifact. Every counter must reach the JSON, and folding any of them
    /// into `deficit` must break the records identity.
    ///
    /// Trap: `int_field` scans for the FIRST occurrence of a key, so a reads
    /// counter that was never serialized would leave `deficit` at 0 and make
    /// the first half vacuous — which is why the fixture populates reads
    /// counters with values that are NOT the record ones (reads_seen 43 vs
    /// passed 26).
    #[test]
    fn the_reads_block_is_emitted_carries_its_unit_and_stays_out_of_the_identity() {
        let s = balanced_stats();
        let json = render_anchor(&s);
        assert!(
            json.contains(READ_SCOPE),
            "reads block lost its scope:\n{json}"
        );

        for (k, want) in [
            ("reads_seen", 45),
            ("anchors_emitted", 26),
            ("rejected_tie", 10),
            ("rejected_usurpation", 2),
            ("reads_no_eligible_candidate", 4),
            ("reads_over_cap", 3),
        ] {
            assert_eq!(
                int_field(&json, k),
                want,
                "reads.{k} not serialized:\n{json}"
            );
        }
        // Absent counters read 0 rather than vanishing, so a consumer never
        // has to distinguish "no reads were rejected" from "nobody wrote it".
        for k in [
            "rejected_gap",
            "rejected_same_gene",
            "rejected_cross_gene",
            "gene_veto_disabled",
            "same_gene_unknown",
        ] {
            assert_eq!(int_field(&json, k), 0, "reads.{k} not serialized:\n{json}");
        }

        // The reads block's own identity, recomputed from the ARTIFACT rather
        // than from the struct: without `reads_over_cap` the sum is 42 against
        // a reads_seen of 45, and without `rejected_usurpation` it is 43.
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
        assert!(
            READ_SCOPE.contains("reads_over_cap") && READ_SCOPE.contains("rejected_usurpation"),
            "READ_SCOPE quotes the identity, so a new term has to appear in the \
             artifact's own scope string too: {READ_SCOPE}"
        );

        // The whole read block, added to the records identity, must break it.
        // 45 reads added to a 110-record ledger is not a rounding error.
        assert_eq!(
            bucket_sum(&json) + int_field(&json, "passed"),
            110,
            "the RECORDS identity must not move when reads are present"
        );
        assert_eq!(
            bucket_sum(&json) + int_field(&json, "passed") + int_field(&json, "reads_seen"),
            155,
            "sanity: this is the arithmetic that would be reported if a \
             consumer mixed the two denominations"
        );
        assert_eq!(int_field(&json, "deficit"), 0);
    }

    /// THE RETIRED `unique` SLOT IS EMITTED, AND IT IS 0.
    ///
    /// `AnchorStats::default()` is the cheapest witness that the key survives
    /// `render_anchor`: removing the line would make `int_field` panic rather
    /// than return a wrong number.
    ///
    /// It is NOT sufficient on its own: `default()` is 0 whether the production
    /// code increments the counter or not, so this says the KEY exists and
    /// nothing about the RULE. Read it as "the slot is still in the schema",
    /// not as "the slot is still 0"; the rule is pinned by
    /// `build_anchors_reports_a_retired_unique_bucket_of_zero` in
    /// `tests/ledger_json_artifact.rs`.
    #[test]
    fn retired_unique_bucket_is_reported_as_zero() {
        let json = render_anchor(&AnchorStats::default());
        assert_eq!(
            int_field(&json, "unique"),
            0,
            "`unique` is a retired compat slot: the MAPQ < 20 gate is gone, so \
             nothing may increment it, and the key must stay in the frozen \
             ledger schema.\n{json}"
        );
        // And it is still inside the twelve-term sum rather than dangling:
        // `bucket_sum` would not raise on a zero term, so pin the count.
        assert_eq!(ANCHOR_BUCKETS.len(), 12);
        assert!(ANCHOR_BUCKETS.contains(&"unique"));
        assert_eq!(bucket_sum(&json), 0, "a default ledger counts nothing");
    }
}
