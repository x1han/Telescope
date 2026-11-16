use std::io::Write;
use std::path::Path;

use anyhow::Context as _;
use telescope_core::anchor_builder::AnchorStats;
use telescope_core::evidence::FilterCounts;

const SCHEMA: &str = "telescope.filter_ledger";

const RECORD_SCOPE: &str = "RECORDS, not reads, not molecules";

const ANCHOR_SCOPE: &str = "RECORDS read from each anchor BAM; parse_errors is \
a DISJOINT population; distinct_genes / distinct_transcripts are POOL-\
level, assigned by build_anchor_pool and NOT summed over samples";

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

pub fn write_filter(path: &Path, s: &FilterCounts) -> anyhow::Result<()> {
    write_raw(path, &render_filter(s))
}

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

    #[test]
    fn anchor_ledger_carries_the_deficit_field() {
        let s = balanced_stats();
        let json = render_anchor(&s);
        assert_eq!(int_field(&json, "deficit"), 0);

        let mut leaking = balanced_stats();
        leaking.passed -= 2;
        assert_eq!(int_field(&render_anchor(&leaking), "deficit"), 2);
    }

    #[test]
    fn both_levels_carry_a_scope_string() {
        let a = render_anchor(&balanced_stats());
        assert!(a.contains(ANCHOR_SCOPE), "anchor_level lost its scope");
        let f = render_filter(&filter_counts());
        assert!(f.contains(RECORD_SCOPE), "record_level lost its scope");
    }

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

    #[test]
    fn parse_errors_is_reported_but_excluded_from_the_identity() {
        let s = balanced_stats();
        let json = render_anchor(&s);
        assert_eq!(int_field(&json, "parse_errors"), 7, "not serialized");

        assert_eq!(bucket_sum(&json) + int_field(&json, "passed"), 110);
        assert_eq!(s.unaccounted(), 0);
    }

    #[test]
    fn each_command_emits_only_its_own_level() {
        let a = render_anchor(&balanced_stats());
        assert!(a.contains("\"anchor_level\""));
        assert!(!a.contains("\"record_level\""));

        let f = render_filter(&filter_counts());
        assert!(f.contains("\"record_level\""));
        assert!(!f.contains("\"anchor_level\""));
    }

    #[test]
    fn schema_tag_is_pinned() {
        assert!(render_anchor(&balanced_stats()).contains(SCHEMA));
        assert!(render_filter(&filter_counts()).contains(SCHEMA));
    }

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

        for k in [
            "rejected_gap",
            "rejected_same_gene",
            "rejected_cross_gene",
            "gene_veto_disabled",
            "same_gene_unknown",
        ] {
            assert_eq!(int_field(&json, k), 0, "reads.{k} not serialized:\n{json}");
        }

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

        assert_eq!(ANCHOR_BUCKETS.len(), 12);
        assert!(ANCHOR_BUCKETS.contains(&"unique"));
        assert_eq!(bucket_sum(&json), 0, "a default ledger counts nothing");
    }
}
