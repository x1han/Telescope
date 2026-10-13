//! Schema v1 of the machine-readable filter ledger.
//!
//! Frozen by the Round 65 ruling in
//! `workspace/comms/2026-10-13-round65-ticket2-a4-ledger-plan.md` §2.3. Two
//! deviations from the literal text of that document are deliberate and
//! recorded here rather than made silently:
//!
//! 1. **`anchor_level.filtered` carries NINE keys, not the six the ruling
//!    lists.** The ruling's schema was written against the pre-Ticket-2A code,
//!    which had six filter buckets. Ticket 2A added three
//!    (`no_refid` / `no_position` / `no_read_id`), and decision 1 of the same
//!    ruling requires `deficit` to be machine-readable IN THE ARTIFACT.
//!    Emitting six buckets while the code has nine would make `deficit`
//!    wrong in the one place it is supposed to be authoritative. The nested
//!    shape, the key names, and the `schema` string are unchanged.
//!
//! 2. **Each command emits only the level it has.** Ruling §2.2: "它们分属
//!    不同阶段,不能写进同一个文件 —— 一个命令只写自己有的那段." So
//!    [`render_filter`] emits `record_level` and no `anchor_level`; the
//!    absent level is OMITTED, not nulled. `--ledger` gives the caller an
//!    explicit path, so the two can never land in one file by accident.
//!
//! JSON is hand-written with `format!`, matching `sim_bridge::write_stats`.
//! `telescope-core` has no serde_json dependency and this adds none: a static
//! artifact is not worth a new dependency surface. Every value is an integer
//! or a fixed literal, so there is no escaping to get wrong.

use std::io::Write;
use std::path::Path;

use anyhow::Context as _;
use telescope_core::anchor_builder::AnchorStats;
use telescope_core::evidence::FilterCounts;

/// The schema tag. Bump this, and the filename, together.
const SCHEMA: &str = "telescope.filter_ledger/v1";

/// Decision 4: the unit travels WITH the number. It used to live only in a
/// doc comment on `print_filter_ledger`, where it stopped existing the moment
/// the value was quoted out of the log.
const RECORD_SCOPE: &str = "RECORDS, not reads, not molecules";

/// Decision 2: `parse_errors` is disjoint from `total_records` (it is
/// incremented before it), so a consumer that folds it into the sum reads
/// `deficit = -1` and blames a bias. Say so in the artifact.
///
/// Decision 3's caveat is appended to the same string. The frozen schema's
/// literal (ruling :97) carries the decision-2 caveat and NOT the decision-3
/// one, which is an inconsistency inside the frozen document itself: :122-124
/// requires `distinct_genes` / `distinct_transcripts` to be marked pool-level,
/// and `scope` is where decision ② establishes that such caveats live. Putting
/// it in the same place keeps one convention instead of two.
const ANCHOR_SCOPE: &str = "RECORDS read from each anchor BAM; parse_errors is \
a DISJOINT population; distinct_genes / distinct_transcripts are POOL-\
level, assigned by build_anchor_pool and NOT summed over samples";

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
         \"cap\": {}\n\
         }},\n\
         \"passed\": {},\n\
         \"deficit\": {},\n\
         \"distinct_genes\": {},\n\
         \"distinct_transcripts\": {}\n\
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
        s.passed,
        s.unaccounted(),
        s.distinct_genes,
        s.distinct_transcripts,
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
    /// Deliberately a substring scan and not a JSON parser: the writer is
    /// hand-written, and parsing it with serde_json would only prove it
    /// against a schema, not against the identity the buckets have to satisfy.
    /// Every value in the artifact is an integer or a fixed literal, so
    /// `"key": ` followed by an optional `-` and digits is unambiguous.
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

    /// The nine anchor-level filter buckets, in the order `report()` emits
    /// them. Two tests walk this list; if a tenth bucket is ever added, both
    /// go red rather than silently comparing a 9-term sum against a 10-term
    /// total.
    const ANCHOR_BUCKETS: [&str; 9] = [
        "unique",
        "no_refid",
        "no_position",
        "no_read_id",
        "no_reflen",
        "3p_anchor",
        "coverage",
        "polya_invalid",
        "cap",
    ];

    /// 40+5+3+2+1+4+6+2+1 == 64, plus `passed` 46, is exactly `total_records`.
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
            passed: 46,
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

    /// Ruling §2.5 test 2: the artifact must carry the deficit as a FIELD.
    /// Hardcoding it to 0 turns this red on an unbalanced fixture.
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

    /// Ruling §2.5 test 3: `scope` must be present, because the unit used to
    /// live only in a comment and died the moment a number was quoted out.
    #[test]
    fn both_levels_carry_a_scope_string() {
        let a = render_anchor(&balanced_stats());
        assert!(a.contains(ANCHOR_SCOPE), "anchor_level lost its scope");
        let f = render_filter(&filter_counts());
        assert!(f.contains(RECORD_SCOPE), "record_level lost its scope");
    }

    /// The identity must hold when recomputed from the artifact's OWN numbers
    /// -- not from the struct that produced it. This is the check that catches
    /// a key being dropped from the `format!`, which no other test would.
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

    /// Decision 2: `parse_errors` is aggregated but must stay OUT of the
    /// identity. It is deliberately non-zero in the fixture, so folding it in
    /// turns this red.
    #[test]
    fn parse_errors_is_reported_but_excluded_from_the_identity() {
        let s = balanced_stats();
        let json = render_anchor(&s);
        assert_eq!(int_field(&json, "parse_errors"), 7, "not serialized");
        // Folding parse_errors into the RHS would make the sum 117 and the
        // deficit -7, which reads like a bias parameter rather than the
        // bookkeeping mistake it is. The fixture's parse_errors is
        // deliberately non-zero so that mistake is visible here.
        assert_eq!(bucket_sum(&json) + int_field(&json, "passed"), 110);
        assert_eq!(s.unaccounted(), 0);
    }

    /// Ruling §2.2: the two levels must not share a document, and the absent
    /// one is omitted rather than nulled.
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
}
