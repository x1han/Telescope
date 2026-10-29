//! Transcriptome Evidence Extractor: streams a transcriptome BAM, extracts Dorado `pt:i` poly(A) tail length plus per-candidate alignment evidence, and builds a flat candidate graph.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context, Result};
use noodles_bam as bam;
use noodles_sam::alignment::record::data::field::Tag;
use noodles_sam::alignment::Record as _;

/// Dorado poly(A) tail length SAM tag: `pt:i`. The default for `PolyASource::BamTag`.
const TAG_PT: Tag = Tag::new(b'p', b't');

/// minimap2 alignment score SAM tag: `AS:i` (primary; drives the alignment rank term A_rk).
const TAG_AS: Tag = Tag::new(b'A', b'S');

/// Edit distance to reference SAM tag: `NM:i` (audit only).
const TAG_NM: Tag = Tag::new(b'N', b'M');

/// One single-molecule evidence tuple: read identity, log2 poly(A) length, and this read's slice of `candidate_pool`.
#[derive(Debug, Clone)]
pub struct MoleculeEvidence {
    pub read_id: String,
    pub log_pal: f32,
    pub candidate_offset: u32,
    pub candidate_count: u16,
}

/// Record-level accounting for the filters in `extract_evidence_with`. WHY A LEDGER: records are dropped at SIX distinct sites; `ledger_deficit()` is asserted 0, so a new drop site FAILS LOUDLY.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FilterCounts {
    /// Every record read from the BAM, before any filter. The LHS of the identity below.
    pub records_entered: u64,
    pub dropped_unmapped: u64,
    /// `FLAG & 2048` -- dropped, not merged: merging a split read's segments is unsound from a transcriptome BAM alone.
    pub dropped_supplementary: u64,
    pub dropped_no_poly_a_tag: u64,
    /// The poly(A) tag parsed to a non-positive value or above `i32::MAX`: the extraction SUCCEEDED and the value was rejected.
    pub dropped_invalid_poly_a: u64,
    /// `alignment_span()` returned `None` or `Err` -- the CIGAR could not be resolved to a span.
    pub dropped_no_span: u64,
    pub dropped_empty_span: u64,
    /// 3' TES boundary: the alignment ends more than `tes_slack` before the transcript's 3' end.
    pub dropped_tes_window: u64,
    /// Records that survived every filter and reached candidate construction: the identity's RHS, a RECORD count.
    pub records_with_candidates: u64,
}

impl FilterCounts {
    /// Sum of every drop bucket.
    pub fn dropped_total(&self) -> u64 {
        self.dropped_unmapped
            + self.dropped_supplementary
            + self.dropped_no_poly_a_tag
            + self.dropped_invalid_poly_a
            + self.dropped_no_span
            + self.dropped_empty_span
            + self.dropped_tes_window
    }

    /// `records_entered - (dropped_total + records_with_candidates)`. MUST be 0: a non-zero value means a record left the stream through a path not in the ledger.
    pub fn ledger_deficit(&self) -> i64 {
        self.records_entered as i64 - (self.dropped_total() + self.records_with_candidates) as i64
    }
}

/// Compact memory graph containing all reads and a flattened pool of transcript candidate IDs, plus the per-candidate alignment evidence in lockstep pools. Layout invariant: the five pools are molecule-major / candidate-minor concatenations of identical length; a molecule owns `[candidate_offset, candidate_offset + candidate_count)` in *each*.
#[derive(Debug)]
pub struct FlatEvidenceGraph {
    pub molecules: Vec<MoleculeEvidence>,
    /// Candidate transcript IDs (index into `transcript_names`).
    pub candidate_pool: Vec<u32>,
    /// minimap2 alignment score `AS:i` per candidate. A record without `AS:i` contributes 0, which only leaves A_rk uniform when EVERY candidate of that read is tag-less.
    pub candidate_as: Vec<i32>,
    /// Edit distance `NM:i` per candidate (audit only; not used by the EM solver).
    pub candidate_nm: Vec<i32>,
    /// Mapping quality per candidate (audit only; not used by the EM solver).
    pub candidate_mapq: Vec<u8>,
    /// Observed 5' gap `g_k` per candidate, in the candidate transcript's own cDNA coordinates: the
    /// number of transcript bases 5' of the read's 5' end. NO strand correction: reverse records are REFUSED.
    pub candidate_5p_gap: Vec<u32>,
    pub transcript_names: Vec<String>,
    /// PROVENANCE, not measurement: true iff this graph was built with `PolyASource::None`.
    pub poly_a_is_placeholder: bool,
    /// Record-level filter ledger for the extraction that produced this graph.
    pub stats: FilterCounts,
}

/// Where per-read poly(A) length comes from. `pt:i` is Dorado's native tail-length tag, the default; a BAM without it produced an EMPTY GRAPH and exit code 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolyASource {
    /// Read a two-character SAM tag from each record. `None` on a record is a drop; a global absence is a hard error, not a silent success.
    BamTag(Tag),
    /// Declared absence of poly(A). `log_pal` is a finite placeholder, NOT DATA: `log_pal = 0.0` -- `log2(1)` -- so the CLI FORCES `poly_a_weight = 0`.
    None,
}

/// Parse the `--polyA-source` CLI value into a [`PolyASource`]: `bam_tag:<TAG>` (the default) or `none`.
pub fn parse_poly_a_source(spec: &str) -> Result<PolyASource> {
    let spec = spec.trim();
    if spec == "none" {
        return Ok(PolyASource::None);
    }
    if let Some(tag) = spec.strip_prefix("bam_tag:") {
        let bytes = tag.as_bytes();
        // SAM tags are exactly two alphanumeric characters; checking here keeps a typo from a panic in a long batch run.
        if bytes.len() != 2 || !bytes.iter().all(u8::is_ascii_alphanumeric) {
            anyhow::bail!("--polyA-source bam_tag: needs a 2-character SAM tag, got `{tag}`");
        }
        return Ok(PolyASource::BamTag(Tag::from([bytes[0], bytes[1]])));
    }
    anyhow::bail!("--polyA-source must be one of `bam_tag:<TAG>` or `none`; got `{spec}`");
}

/// Number of records scanned before deciding a poly(A) source is globally absent.
const POLY_A_PROBE_RECORDS: usize = 10_000;

/// Parse the transcriptome BAM and emit the candidate graph, reading poly(A) from Dorado's native `pt:i` tag. **Fails** rather than returning an empty graph when no record carries the tag.
pub fn extract_evidence<P: AsRef<Path>>(
    bam_path: P,
    tes_slack: usize,
) -> Result<FlatEvidenceGraph> {
    extract_evidence_with(bam_path, tes_slack, &PolyASource::BamTag(TAG_PT))
}

pub fn extract_evidence_with<P: AsRef<Path>>(
    bam_path: P,
    tes_slack: usize,
    poly_a: &PolyASource,
) -> Result<FlatEvidenceGraph> {
    let file = File::open(&bam_path)
        .with_context(|| format!("Failed to open BAM: {:?}", bam_path.as_ref()))?;
    let mut reader = bam::io::Reader::new(BufReader::new(file));
    let header = reader.read_header().context("Failed to read BAM header")?;

    let mut transcript_names: Vec<String> = Vec::new();
    let mut transcript_lengths: Vec<usize> = Vec::new();

    for (_ref_id, (name, target)) in header.reference_sequences().iter().enumerate() {
        let name_str = String::from_utf8_lossy(name).to_string();
        transcript_names.push(name_str);
        transcript_lengths.push(target.length().get());
    }

    if transcript_names.is_empty() {
        bail!("Transcriptome BAM header contains no reference transcripts");
    }

    let tag_to_read: Option<Tag> = match poly_a {
        PolyASource::BamTag(t) => Some(*t),
        _ => None,
    };
    if let Some(t) = tag_to_read {
        eprintln!(
            "[telescope] poly(A) source: bam_tag:{}",
            String::from_utf8_lossy(t.as_ref())
        );
    }
    // Records seen and records carrying poly(A), for the global-absence guard, keyed on `hits == 0`.
    let mut poly_a_records_seen: u64 = 0;
    let mut poly_a_hits: u64 = 0;

    // Record-level filter ledger. Asserted closed at the end of this function (`fc.ledger_deficit() == 0`), so a drop site without its counter fails loudly.
    let mut fc = FilterCounts::default();

    // Temporary per-read accumulator. MEMORY: the per-candidate alignment evidence moved peak RSS on a 16.4 GB BAM from 2.99 GB to ~4.1 GB; that growth is transient, not a leak.
    struct TempReadEntry {
        poly_a_len: i32,
        candidates: Vec<CandidateAlign>,
    }

    // One (read, transcript) candidate plus the alignment evidence of the record that won the max-AS merge.
    struct CandidateAlign {
        tx_id: u32,
        as_score: i32,
        nm: i32,
        mapq: u8,
        gap5p: u32,
    }

    let mut read_map: HashMap<String, TempReadEntry> = HashMap::new();

    let mut record = bam::Record::default();
    while reader.read_record(&mut record)? != 0 {
        fc.records_entered += 1;
        let flags = record.flags();
        if flags.is_unmapped() {
            fc.dropped_unmapped += 1;
            continue;
        }
        // REVERSE-STRAND GUARD. No code in this crate reads `FLAG & 0x10`; transcriptome alignments are forward by construction, but that is a property of the DATA, not of the CODE.
        if flags.is_reverse_complemented() {
            let qname = match record.name() {
                Some(name) => String::from_utf8_lossy(name).to_string(),
                None => "<unnamed>".to_string(),
            };
            bail!(
                "reverse-strand alignment in a transcriptome BAM: record {} \
                 (QNAME {}). Telescope computes the 5' gap `g_k` on cDNA \
                 coordinates and applies no strand correction -- none exists in \
                 this crate -- so a reverse record would produce a 5' gap that \
                 means nothing. Supply a transcriptome alignment.",
                fc.records_entered,
                qname
            );
        }
        // SUPPLEMENTARY (`FLAG & 2048`) records are DROPPED, not merged. The CIGAR is written in the
        // molecule's own 5'->3' order, so a LEADING hard clip means this record is the **3'** segment,
        // and merging is unsound because the max-AS merge can report its interior start as the 5' end.
        if flags.is_supplementary() {
            fc.dropped_supplementary += 1;
            continue;
        }

        // Counted for EVERY record that survives the flag filters, BEFORE the poly(A) lookup.
        poly_a_records_seen += 1;

        // GLOBAL-ABSENCE GUARD, in-loop form: sampled over the first POLY_A_PROBE_RECORDS records so a BAM with no poly(A) fails at record 10,000. IT MUST SIT ABOVE THE LOOKUP: below it every arm of the `match` `continue`s.
        if poly_a_records_seen as usize == POLY_A_PROBE_RECORDS && poly_a_hits == 0 {
            let spec = match poly_a {
                PolyASource::BamTag(t) => {
                    format!("bam_tag:{}", String::from_utf8_lossy(t.as_ref()))
                }
                PolyASource::None => "none".to_string(),
            };
            bail!(
                "FATAL: no record among the first {POLY_A_PROBE_RECORDS} carries \
                 poly(A) from the requested source `{spec}`. \
                 If your BAM lacks Dorado poly(A) tags, specify \
                 `--polyA-source none` for a pure-alignment run. \
                 Refusing to continue: returning an empty graph here \
                 is indistinguishable downstream from a sample with no reads."
            );
        }

        let data = record.data();
        let poly_a_len: i32 = match poly_a {
            PolyASource::BamTag(tag) => match data.get(tag) {
                Some(Ok(value)) => match value.as_int() {
                    Some(len) if len > 0 && len <= i32::MAX as i64 => len as i32,
                    _ => {
                        fc.dropped_invalid_poly_a += 1;
                        continue; // skip non-positive or >i32::MAX
                    }
                },
                _ => {
                    fc.dropped_no_poly_a_tag += 1;
                    continue; // skip missing or unparseable tag
                }
            },
            PolyASource::None => {
                // Declared-absent. Nothing is dropped; the placeholder must be 1, not 0: `0i32.log2()` is `-inf`.
                1
            }
        };
        if poly_a_len > 0 {
            poly_a_hits += 1;
        }
        // Reference sequence index + alignment 3' end position; noodles-bam methods return `Option<Result<T, io::Error>>`.
        let ref_id = match record.reference_sequence_id() {
            Some(Ok(id)) => id,
            _ => continue,
        };
        let ref_len = transcript_lengths[ref_id];

        // Compute ref_end = alignment_start + alignment_span - 1 (1-based).
        let start_pos = match record.alignment_start() {
            Some(Ok(p)) => p.get(),
            _ => continue,
        };
        let span = match record.alignment_span() {
            Some(Ok(s)) => s,
            _ => {
                fc.dropped_no_span += 1;
                continue;
            }
        };
        if span == 0 {
            fc.dropped_empty_span += 1;
            continue; // empty alignment
        }
        let ref_end = start_pos + span - 1;

        // 3' TES boundary filter: read must terminate within `tes_slack` of the transcript's 3' end.
        if ref_end + tes_slack < ref_len {
            fc.dropped_tes_window += 1;
            continue;
        }

        // THE RHS TERM LIVES BELOW, NOT HERE: `fc.records_with_candidates += 1` sits immediately above
        // `let entry = read_map.entry(read_id)`. INVARIANT: no early exit between it and the push.
        // Observed 5' gap `g_k`, in this candidate's own cDNA coordinates: the reference of a transcriptome BAM IS the transcript cDNA, and noodles' `alignment_start()` is 1-BASED.
        let gap5p: u32 = u32::try_from(start_pos.saturating_sub(1)).unwrap_or(u32::MAX);

        let read_id = match record.name() {
            Some(name) => String::from_utf8_lossy(name).to_string(),
            None => continue,
        };

        // Alignment evidence. `AS:i` is the primary signal consumed by the EM solver's A_rk term; `NM:i` and MAPQ are retained for audit. A missing tag falls back to 0, neutral ONLY when EVERY candidate of the read is tag-less.
        let as_score = match data
            .get(&TAG_AS)
            .and_then(|r| r.ok())
            .and_then(|v| v.as_int())
        {
            Some(v) => i32::try_from(v).unwrap_or(0),
            None => 0,
        };
        let nm = match data
            .get(&TAG_NM)
            .and_then(|r| r.ok())
            .and_then(|v| v.as_int())
        {
            Some(v) => i32::try_from(v).unwrap_or(0),
            None => 0,
        };
        let mapq = match record.mapping_quality() {
            Some(mq) => mq.get(),
            None => 0,
        };

        // Accumulate candidate transcripts per read. A repeated (read, transcript) pair must MERGE, not be dropped.
        let tx_id = ref_id as u32;
        let align = CandidateAlign {
            tx_id,
            as_score,
            nm,
            mapq,
            gap5p,
        };

        // THE RHS TERM OF THE LEDGER IDENTITY: the LAST point at which a record can still leave, so every exit lands on the LHS alone.
        fc.records_with_candidates += 1;

        let entry = read_map.entry(read_id).or_insert_with(|| TempReadEntry {
            poly_a_len,
            candidates: Vec::new(),
        });
        match entry.candidates.iter_mut().find(|c| c.tx_id == tx_id) {
            Some(existing) => {
                if align.as_score > existing.as_score {
                    *existing = align;
                }
            }
            None => entry.candidates.push(align),
        }
    }

    // END-OF-STREAM version of the same guard: a BAM smaller than POLY_A_PROBE_RECORDS never reaches the in-loop check.
    if poly_a_records_seen > 0 && poly_a_hits == 0 {
        let spec = match poly_a {
            PolyASource::BamTag(t) => {
                format!("bam_tag:{}", String::from_utf8_lossy(t.as_ref()))
            }
            PolyASource::None => "none".to_string(),
        };
        bail!(
            "FATAL: none of the {poly_a_records_seen} scanned records carries \
             poly(A) from the requested source `{spec}`. \
             If your BAM lacks Dorado poly(A) tags, specify \
             `--polyA-source none` for a pure-alignment run. \
             Refusing to continue: returning an empty graph here \
             is indistinguishable downstream from a sample with no reads."
        );
    }

    // Flatten into `FlatEvidenceGraph`. MEMORY: reserve the five pools at their exact final length instead
    // of growing them by `push`; the `is_empty` filter mirrors the `continue` below, so the capacity is an UPPER bound.
    let total_candidates: usize = read_map
        .values()
        .filter(|e| !e.candidates.is_empty())
        .map(|e| e.candidates.len())
        .sum();
    let mut molecules: Vec<MoleculeEvidence> = Vec::with_capacity(read_map.len());
    let mut candidate_pool: Vec<u32> = Vec::with_capacity(total_candidates);
    let mut candidate_as: Vec<i32> = Vec::with_capacity(total_candidates);
    let mut candidate_nm: Vec<i32> = Vec::with_capacity(total_candidates);
    let mut candidate_mapq: Vec<u8> = Vec::with_capacity(total_candidates);
    let mut candidate_5p_gap: Vec<u32> = Vec::with_capacity(total_candidates);

    // DETERMINISM: `read_map` is a `HashMap`, so iterating it yields a per-process random order, which decides the order of the `init_counts` FLOAT accumulation. Sorting by `read_id` fixes it.
    let mut entries: Vec<(String, TempReadEntry)> = read_map.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    for (read_id, entry) in entries {
        if entry.candidates.is_empty() {
            continue;
        }
        let candidate_offset = candidate_pool.len() as u32;
        let candidate_count = entry.candidates.len() as u16;
        for c in &entry.candidates {
            candidate_pool.push(c.tx_id);
            candidate_as.push(c.as_score);
            candidate_nm.push(c.nm);
            candidate_mapq.push(c.mapq);
            candidate_5p_gap.push(c.gap5p);
        }

        molecules.push(MoleculeEvidence {
            read_id,
            // `poly_a_len` is >= 1 on every path that reaches here, so this is finite.
            log_pal: (entry.poly_a_len.max(1) as f32).log2(),
            candidate_offset,
            candidate_count,
        });
    }

    // Lockstep invariant: the alignment-evidence pools are index-aligned with `candidate_pool`. UNCONDITIONAL `assert_eq!`, not `debug_assert_eq!`.
    assert_eq!(candidate_pool.len(), candidate_as.len());
    assert_eq!(candidate_pool.len(), candidate_nm.len());
    assert_eq!(candidate_pool.len(), candidate_mapq.len());
    assert_eq!(candidate_pool.len(), candidate_5p_gap.len());

    // THE LEDGER CLOSES HERE, or the library is wrong.
    assert_eq!(
        fc.ledger_deficit(),
        0,
        "evidence filter ledger does not close: {fc:?} -- a record left the \
         stream through an uncounted path (a new drop site without its counter, \
         or an early return between the filters and the candidate push)"
    );

    Ok(FlatEvidenceGraph {
        molecules,
        candidate_pool,
        candidate_as,
        candidate_nm,
        candidate_mapq,
        candidate_5p_gap,
        transcript_names,
        poly_a_is_placeholder: matches!(poly_a, PolyASource::None),
        stats: fc,
    })
}

impl FlatEvidenceGraph {
    /// Distribution of `candidate_count` across molecules: a sorted vec of `(candidate_count, n_molecules_with_that_count)` pairs.
    pub fn candidate_count_histogram(&self) -> Vec<(u16, u64)> {
        let mut hist: HashMap<u16, u64> = HashMap::new();
        for mol in &self.molecules {
            *hist.entry(mol.candidate_count).or_insert(0) += 1;
        }
        let mut v: Vec<_> = hist.into_iter().collect();
        v.sort_by_key(|(k, _)| *k);
        v
    }
}

#[cfg(any(test, feature = "test-support"))]
/// Synthetic-BAM helpers shared with `resolver`'s tests and the integration tests in `tests/`.
#[doc(hidden)]
pub mod test_support {
    use super::*;

    use std::num::NonZero;

    use noodles_core::Position;

    /// RE-EXPORTED so a caller in another crate can build a flagged record without depending on `noodles_sam`.
    pub use noodles_sam::alignment::record::Flags as SamFlags;

    use noodles_sam::{
        self as sam,
        alignment::{
            io::Write as _,
            record::{
                cigar::{op::Kind, Op},
                Flags, MappingQuality,
            },
            record_buf::{data::field::Value, Cigar},
            RecordBuf,
        },
        header::record::value::{map::ReferenceSequence, Map},
    };

    /// Length of every reference in `write_synthetic_bam`'s header; a short REF_LEN pins `g_k ≡ 0` for every synthetic candidate.
    pub const REF_LEN: usize = 2000;

    /// Build a synthetic BAM record: a full-length forward match against `ref_id` starting at 1, carrying `pt:i`, `NM:i`, MAPQ and (optionally) `AS:i`.
    pub fn synth_record(
        name: &str,
        ref_id: usize,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        synth_record_with_gap(name, ref_id, 0, false, as_score, nm, mapq)
    }

    /// As `synth_record`, but with an explicit observed 5' GAP `g_k` (in bp) and strand. Forward:
    /// the read starts `g_k + 1` bases into the transcript; reverse: the read ends `g_k` bases
    /// SHORT of the transcript's 3' end. Both cases have span `REF_LEN − g_k`.
    pub fn synth_record_with_gap(
        name: &str,
        ref_id: usize,
        gap: usize,
        reverse: bool,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        let span = REF_LEN - gap;
        let start = if reverse { 1 } else { gap + 1 };
        let cigar: Cigar = [Op::new(Kind::Match, span)].into_iter().collect();
        let mut tags = vec![(TAG_PT, Value::from(100i32)), (TAG_NM, Value::from(nm))];
        if let Some(a) = as_score {
            tags.push((TAG_AS, Value::from(a)));
        }
        let flags = if reverse {
            Flags::REVERSE_COMPLEMENTED
        } else {
            Flags::empty()
        };
        RecordBuf::builder()
            .set_name(name)
            .set_flags(flags)
            .set_reference_sequence_id(ref_id)
            .set_alignment_start(
                Position::new(start).expect("start position is 1-based and non-zero"),
            )
            .set_mapping_quality(MappingQuality::new(mapq).unwrap())
            .set_cigar(cigar)
            .set_data(tags.into_iter().collect())
            .build()
    }

    /// As [`synth_record`], but with NO reference sequence id at all: the encoded record carries tid = -1.
    pub fn synth_record_no_refid(
        name: &str,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        let cigar: Cigar = [Op::new(Kind::Match, REF_LEN)].into_iter().collect();
        let mut tags = vec![(TAG_PT, Value::from(100i32)), (TAG_NM, Value::from(nm))];
        if let Some(a) = as_score {
            tags.push((TAG_AS, Value::from(a)));
        }
        RecordBuf::builder()
            .set_name(name)
            .set_flags(Flags::empty())
            .set_alignment_start(Position::new(1).expect("1-based and non-zero"))
            .set_mapping_quality(MappingQuality::new(mapq).unwrap())
            .set_cigar(cigar)
            .set_data(tags.into_iter().collect())
            .build()
    }

    /// As [`synth_record`], but with NO read name: `QNAME=*` is legal SAM. The record keeps FLAG 0 and a valid tid, so it dies at `record.name()`.
    pub fn synth_record_no_read_id(
        ref_id: usize,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        let cigar: Cigar = [Op::new(Kind::Match, REF_LEN)].into_iter().collect();
        let mut tags = vec![(TAG_PT, Value::from(100i32)), (TAG_NM, Value::from(nm))];
        if let Some(a) = as_score {
            tags.push((TAG_AS, Value::from(a)));
        }
        RecordBuf::builder()
            .set_flags(Flags::empty())
            .set_reference_sequence_id(ref_id)
            .set_alignment_start(Position::new(1).expect("1-based and non-zero"))
            .set_mapping_quality(MappingQuality::new(mapq).unwrap())
            .set_cigar(cigar)
            .set_data(tags.into_iter().collect())
            .build()
    }

    /// As [`synth_record`], but with NO alignment start: the encoder writes POS = 0 and the
    /// decoder's `Position::new(0)` fails -- an unset field is 0, NOT a `-1` sentinel.
    pub fn synth_record_no_position(
        ref_id: usize,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        let cigar: Cigar = [Op::new(Kind::Match, REF_LEN)].into_iter().collect();
        let mut tags = vec![(TAG_PT, Value::from(100i32)), (TAG_NM, Value::from(nm))];
        if let Some(a) = as_score {
            tags.push((TAG_AS, Value::from(a)));
        }
        RecordBuf::builder()
            .set_name("nopos")
            .set_flags(Flags::empty())
            .set_reference_sequence_id(ref_id)
            .set_mapping_quality(MappingQuality::new(mapq).unwrap())
            .set_cigar(cigar)
            .set_data(tags.into_iter().collect())
            .build()
    }

    /// As [`synth_record_with_gap`], but with an explicit leading CIGAR: the ops that sit 5' of
    /// the first `Match`, as `(kind, length)` pairs. The trailing `Match` length is **derived**.
    pub fn synth_record_with_leading_cigar(
        name: &str,
        ref_id: usize,
        gap: usize,
        leading: &[(Kind, usize)],
        as_score: Option<i32>,
    ) -> RecordBuf {
        let consumed: usize = leading
            .iter()
            .filter(|(k, _)| *k != Kind::SoftClip && *k != Kind::HardClip && *k != Kind::Insertion)
            .map(|(_, l)| *l)
            .sum();
        let match_len = (REF_LEN - gap)
            .checked_sub(consumed)
            .expect("leading CIGAR ops must not consume more reference than the read spans");
        let mut ops: Vec<Op> = leading.iter().map(|(k, l)| Op::new(*k, *l)).collect();
        ops.push(Op::new(Kind::Match, match_len));
        let mut tags = vec![(TAG_PT, Value::from(100i32)), (TAG_NM, Value::from(1i32))];
        if let Some(a) = as_score {
            tags.push((TAG_AS, Value::from(a)));
        }
        RecordBuf::builder()
            .set_name(name)
            .set_flags(Flags::empty())
            .set_reference_sequence_id(ref_id)
            .set_alignment_start(
                Position::new(gap + 1).expect("start position is 1-based and non-zero"),
            )
            .set_mapping_quality(MappingQuality::new(60).unwrap())
            .set_cigar(ops.into_iter().collect::<Cigar>())
            .set_data(tags.into_iter().collect())
            .build()
    }

    /// the evidence extractor's `gap5p` under real CIGARs. `gap5p = s_k − 1` counts the reference bases between
    /// the transcript's 5' end and where the alignment begins, and a leading op must not move it.
    #[test]
    fn test_gap5p_is_invariant_to_the_leading_cigar_ops() {
        use Kind::*;
        let gap = 50usize;
        // A different `ref_id` per case: molecules are keyed by read name, not by transcript.
        let cases: Vec<(&str, usize, Vec<(Kind, usize)>)> = vec![
            ("m_only", 0, vec![]),
            ("soft", 1, vec![(SoftClip, 40)]),
            ("ins", 2, vec![(Insertion, 12)]),
            ("del", 0, vec![(Deletion, 9)]),
            ("skip", 1, vec![(Skip, 30)]),
            ("hard", 2, vec![(HardClip, 25)]),
            (
                "mixed",
                0,
                vec![
                    (SoftClip, 15),
                    (Insertion, 6),
                    (Deletion, 4),
                    (HardClip, 10),
                ],
            ),
        ];
        let records: Vec<RecordBuf> = cases
            .iter()
            .map(|(name, ref_id, lead)| {
                synth_record_with_leading_cigar(name, *ref_id, gap, lead, Some(1000))
            })
            .collect();
        let path = write_synthetic_bam("cigar_invariance", records);
        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

        for (name, _, _) in &cases {
            let mol = graph
                .molecules
                .iter()
                .find(|m| m.read_id == *name)
                .unwrap_or_else(|| {
                    panic!(
                        "{name} missing from the graph -- the leading CIGAR ops must not \
                         have pushed its 3' end outside the TES filter"
                    )
                });
            assert_eq!(mol.candidate_count, 1, "{name} must be single-candidate");
            let got = graph.candidate_5p_gap[mol.candidate_offset as usize];
            assert_eq!(
                got, gap as u32,
                "{name}: g_k must be s_k - 1 = {gap} regardless of what the CIGAR \
                 puts in front of the first match. Leading ops that move it mean \
                 the 5' term is reading an aligner artefact."
            );
        }
    }

    /// `gap5p` agrees with an independent cDNA-coordinate computation, for every CIGAR family,
    /// through the real `extract_evidence` path.
    #[test]
    fn test_gap5p_matches_an_independent_cdna_backward_computation() {
        use Kind::*;
        // (name, ref_id, gap, leading ops) — gaps span both sides of the frozen window edge at 20.
        let cases: Vec<(&str, usize, usize, Vec<(Kind, usize)>)> = vec![
            ("b0", 0, 0, vec![]),
            ("b19", 1, 19, vec![]),
            ("b20", 2, 20, vec![]),
            ("b21", 0, 21, vec![]),
            ("b200", 1, 200, vec![]),
            ("b7_soft", 2, 7, vec![(SoftClip, 33)]),
            ("b7_ins", 0, 7, vec![(Insertion, 5)]),
            ("b7_del", 1, 7, vec![(Deletion, 11)]),
            ("b7_hard", 2, 7, vec![(HardClip, 60)]),
        ];
        let records: Vec<RecordBuf> = cases
            .iter()
            .map(|(name, ref_id, gap, lead)| {
                synth_record_with_leading_cigar(name, *ref_id, *gap, lead, Some(1000))
            })
            .collect();
        let path = write_synthetic_bam("cigar_backward", records);
        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

        for (name, _, gap, lead) in &cases {
            let mol = graph
                .molecules
                .iter()
                .find(|m| m.read_id == *name)
                .unwrap_or_else(|| panic!("{name} missing from the graph"));
            let got = graph.candidate_5p_gap[mol.candidate_offset as usize];

            // Independent route: WALK THE CIGAR to count the reference bases the alignment consumes,
            // then measure the 5' end from the known 3' terminus.
            let leading_ref: usize = lead
                .iter()
                .filter(|(k, _)| matches!(k, Deletion | Skip))
                .map(|(_, l)| *l)
                .sum();
            let span = leading_ref + (REF_LEN - gap - leading_ref);
            let backward = REF_LEN - span;
            assert_eq!(
                backward, *gap,
                "{name}: the CIGAR walk must terminate the alignment at REF_LEN, \
                 otherwise this fixture is not testing what it claims"
            );
            assert_eq!(
                got, backward as u32,
                "{name}: forward (s_k - 1) and backward (REF_LEN - walked span) \
                 computations of g_k must agree"
            );
        }
    }

    /// Supplementary alignments are DROPPED. The CIGAR is always written in the **molecule's own
    /// 5' -> 3' order**, so `900H1000M` (a LEADING hard clip) is the **3'** segment.
    #[test]
    fn test_supplementary_alignments_are_dropped_rather_than_merged() {
        use Kind::*;
        // tx0 is 2000 nt and the junction sits at 1980, close to the TES: the 3' TES filter keeps
        // only reads whose alignment end is within `tes_slack` of the transcript end. The 3'
        // segment deliberately scores HIGHER, so the max-AS merge would otherwise report 1980.
        let five_prime = RecordBuf::builder()
            .set_name("split")
            .set_flags(Flags::empty())
            .set_reference_sequence_id(0)
            .set_alignment_start(Position::new(1).unwrap())
            .set_mapping_quality(MappingQuality::new(60).unwrap())
            .set_cigar([Op::new(Match, 1980)].into_iter().collect())
            .set_data(
                [
                    (TAG_PT, Value::from(100i32)),
                    (TAG_NM, Value::from(1i32)),
                    (TAG_AS, Value::from(900i32)),
                ]
                .into_iter()
                .collect(),
            )
            .build();
        let three_prime = RecordBuf::builder()
            .set_name("split")
            .set_flags(Flags::SUPPLEMENTARY)
            .set_reference_sequence_id(0)
            .set_alignment_start(Position::new(1981).unwrap())
            .set_mapping_quality(MappingQuality::new(60).unwrap())
            .set_cigar(
                [Op::new(HardClip, 1980), Op::new(Match, 20)]
                    .into_iter()
                    .collect(),
            )
            .set_data(
                [
                    (TAG_PT, Value::from(100i32)),
                    (TAG_NM, Value::from(1i32)),
                    (TAG_AS, Value::from(2000i32)),
                ]
                .into_iter()
                .collect(),
            )
            .build();
        let orphan = RecordBuf::builder()
            .set_name("orphan")
            .set_flags(Flags::SUPPLEMENTARY)
            .set_reference_sequence_id(1)
            .set_alignment_start(Position::new(1).unwrap())
            .set_mapping_quality(MappingQuality::new(60).unwrap())
            .set_cigar(
                [Op::new(HardClip, 50), Op::new(Match, 1950)]
                    .into_iter()
                    .collect(),
            )
            .set_data(
                [
                    (TAG_PT, Value::from(100i32)),
                    (TAG_NM, Value::from(1i32)),
                    (TAG_AS, Value::from(1500i32)),
                ]
                .into_iter()
                .collect(),
            )
            .build();
        let control = synth_record_with_leading_cigar("control", 2, 30, &[], Some(1000));

        let path = write_synthetic_bam(
            "supplementary_dropped",
            vec![five_prime, three_prime, orphan, control],
        );
        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

        let mut ids: Vec<&str> = graph.molecules.iter().map(|m| m.read_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec!["control", "split"],
            "the supplementary-only read must vanish, and the split read must \
             survive on its primary alone"
        );
        let mid = RecordBuf::builder()
            .set_name("mid_junction_primary")
            .set_flags(Flags::empty())
            .set_reference_sequence_id(2)
            .set_alignment_start(Position::new(1).unwrap())
            .set_mapping_quality(MappingQuality::new(60).unwrap())
            .set_cigar([Op::new(Match, 900)].into_iter().collect())
            .set_data(
                [
                    (TAG_PT, Value::from(100i32)),
                    (TAG_NM, Value::from(1i32)),
                    (TAG_AS, Value::from(900i32)),
                ]
                .into_iter()
                .collect(),
            )
            .build();
        let mid_path = write_synthetic_bam("mid_junction", vec![mid]);
        let mid_graph = extract_evidence(&mid_path, 35).unwrap();
        let _ = std::fs::remove_file(&mid_path);
        assert!(
            mid_graph.molecules.is_empty(),
            "a primary ending at an interior junction is dropped by the 3' TES \
             filter on its own, with or without the supplementary check"
        );
        let mol = graph
            .molecules
            .iter()
            .find(|m| m.read_id == "split")
            .unwrap();
        assert_eq!(
            mol.candidate_count, 1,
            "the dropped supplementary segment must not create a second candidate"
        );
        assert_eq!(
            graph.candidate_5p_gap[mol.candidate_offset as usize], 0,
            "g_k must come from the primary -- the molecule's 5' segment at \
             reference 1, so g_k = 0. Taking the higher-scoring 3' segment \
             instead would have reported 1000, an interior junction coordinate, \
             and by omission in the other direction a small spurious g_k would \
             read as evidence FOR the candidate."
        );
        assert_eq!(
            graph.candidate_5p_gap[graph
                .molecules
                .iter()
                .find(|m| m.read_id == "control")
                .unwrap()
                .candidate_offset as usize],
            30,
            "the control read is unaffected by the filter"
        );
    }

    /// A forward alignment that STOPS SHORT of the transcript's 3' end. `end_offset` is how many
    /// bases short of `REF_LEN` the alignment's last base sits, so `end_offset = 0` is the
    /// full-length case. The filter drops the record iff `end_offset > tes_slack`.
    pub fn synth_record_short_of_tes(
        name: &str,
        ref_id: usize,
        start: usize,
        end_offset: usize,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        assert!(
            end_offset < REF_LEN && start <= REF_LEN - end_offset,
            "synth_record_short_of_tes: start {start} + end_offset {end_offset} \
             does not fit inside REF_LEN {REF_LEN}"
        );
        let span = REF_LEN - end_offset - start + 1;
        let cigar: Cigar = [Op::new(Kind::Match, span)].into_iter().collect();
        let mut tags = vec![(TAG_PT, Value::from(100i32)), (TAG_NM, Value::from(nm))];
        if let Some(a) = as_score {
            tags.push((TAG_AS, Value::from(a)));
        }
        RecordBuf::builder()
            .set_name(name)
            .set_flags(Flags::empty())
            .set_reference_sequence_id(ref_id)
            .set_alignment_start(Position::new(start).expect("start is 1-based and non-zero"))
            .set_mapping_quality(MappingQuality::new(mapq).unwrap())
            .set_cigar(cigar)
            .set_data(tags.into_iter().collect())
            .build()
    }

    /// A full-length forward match carrying `pt:i`, with SAM `flags` set by the caller. This makes
    /// the FLAG-only drop buckets (`dropped_unmapped`, `dropped_supplementary`) observable.
    pub fn synth_record_with_flags(
        name: &str,
        ref_id: usize,
        flags: Flags,
        as_score: Option<i32>,
    ) -> RecordBuf {
        let cigar: Cigar = [Op::new(Kind::Match, REF_LEN)].into_iter().collect();
        let mut tags = vec![(TAG_PT, Value::from(100i32)), (TAG_NM, Value::from(0i32))];
        if let Some(a) = as_score {
            tags.push((TAG_AS, Value::from(a)));
        }
        RecordBuf::builder()
            .set_name(name)
            .set_flags(flags)
            .set_reference_sequence_id(ref_id)
            .set_alignment_start(Position::new(1).expect("1-based and non-zero"))
            .set_mapping_quality(MappingQuality::new(60).unwrap())
            .set_cigar(cigar)
            .set_data(tags.into_iter().collect())
            .build()
    }

    /// Writes `records` under a 3-transcript header (tx0/tx1/tx2, `REF_LEN` bp each) to a BAM in
    /// the system temp dir named after `tag` and the pid. `tag` must differ per caller.
    pub fn write_synthetic_bam(tag: &str, records: Vec<RecordBuf>) -> std::path::PathBuf {
        write_synthetic_bam_named(tag, &["tx0", "tx1", "tx2"], records)
    }

    /// `write_synthetic_bam` with the `@SQ` names chosen by the caller. The production
    /// `read_reference_universe` deliberately does NOT de-duplicate `@SQ` names, so a header naming
    /// the same transcript twice reaches the universe-padding loop twice.
    pub fn write_synthetic_bam_named(
        tag: &str,
        names: &[&str],
        records: Vec<RecordBuf>,
    ) -> std::path::PathBuf {
        let mut header = sam::Header::builder();
        for &name in names {
            header = header.add_reference_sequence(
                name,
                Map::<ReferenceSequence>::new(const { NonZero::new(REF_LEN).unwrap() }),
            );
        }
        let header = header.build();

        let path = std::env::temp_dir().join(format!(
            "telescope_evidence_named_{}_test_{}.bam",
            tag,
            std::process::id()
        ));
        let mut writer = bam::io::Writer::new(std::fs::File::create(&path).unwrap());
        writer.write_header(&header).unwrap();
        for r in &records {
            writer.write_alignment_record(&header, r).unwrap();
        }
        writer.try_finish().unwrap();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    /// The in-loop probe, which the end-of-stream guard cannot stand in for: 10,000 records with NO
    /// poly(A), then records that DO have it.
    #[test]
    fn test_the_in_loop_probe_catches_poly_a_that_only_starts_after_10000_records() {
        use noodles_sam::alignment::record_buf::data::field::Value;

        let mut recs = Vec::new();
        for i in 0..(POLY_A_PROBE_RECORDS + 1) {
            let mut r = synth_record_with_gap(&format!("late_{i}"), 0, 0, false, Some(1000), 1, 60);
            r.data_mut().clear();
            if i < POLY_A_PROBE_RECORDS {
                r.data_mut().insert(TAG_NM, Value::from(1i32)); // no pt:i
            } else {
                r.data_mut().insert(TAG_PT, Value::from(80i32));
            }
            recs.push(r);
        }
        let path = write_synthetic_bam("late_polya", recs);
        let err = extract_evidence(&path, 35).expect_err(
            "the first 10,000 records carry no poly(A); the probe must stop the run \
             even though later records do",
        );
        let _ = std::fs::remove_file(&path);
        let msg = format!("{err:#}");
        assert!(
            msg.contains("among the first 10000"),
            "must name the probe depth, not the end-of-stream count, got: {msg}"
        );
    }

    /// The provenance flag is set by the source, not inferred from the data.
    #[test]
    fn test_the_placeholder_flag_is_source_derived_not_inferred_from_log_pal() {
        use noodles_sam::alignment::record_buf::data::field::Value;

        // `src_pt` carries a real tail length; `src_none` has none. Only ONE is stripped.
        let mut a = synth_record_with_gap("src_pt", 0, 0, false, Some(1000), 1, 60);
        a.data_mut().clear();
        a.data_mut().insert(TAG_PT, Value::from(90i32));
        let mut b = synth_record_with_gap("src_none", 1, 0, false, Some(1000), 1, 60);
        b.data_mut().clear();
        b.data_mut().insert(TAG_NM, Value::from(1i32));
        let path = write_synthetic_bam("provenance", vec![a, b]);

        let measured = extract_evidence_with(&path, 35, &PolyASource::BamTag(TAG_PT)).unwrap();
        assert!(
            !measured.poly_a_is_placeholder,
            "a bam_tag graph is a measurement, whatever its log_pal values are"
        );

        let none_src = extract_evidence_with(&path, 35, &PolyASource::None).unwrap();
        assert!(
            none_src.poly_a_is_placeholder,
            "a PolyASource::None graph must be flagged, or the em.rs guard cannot fire"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The graph's molecules come out in `read_id` order, which is why the emitted parquet is bit-reproducible. This extracts EIGHT TIMES, NOT ONCE: a 3-key `HashMap` iterates in sorted order by chance 1 time in 3! = 6.
    #[test]
    fn test_molecules_are_emitted_in_read_id_order() {
        use noodles_sam::alignment::record_buf::data::field::Value;
        let names = ["r7", "r2", "r5", "r1", "r8", "r3", "r6", "r4"];
        let recs: Vec<_> = names
            .iter()
            .map(|n| {
                // ref_id stays 0: it indexes the header's reference list, a single sequence here.
                let mut r = synth_record_with_gap(n, 0, 0, false, Some(1000), 1, 60);
                r.data_mut().clear();
                r.data_mut().insert(TAG_PT, Value::from(90i32));
                r
            })
            .collect();
        let path = write_synthetic_bam("sorted_molecules", recs);

        for draw in 0..8 {
            let g = extract_evidence_with(&path, 35, &PolyASource::BamTag(TAG_PT)).unwrap();
            let ids: Vec<&str> = g.molecules.iter().map(|m| m.read_id.as_str()).collect();
            let mut want = ids.clone();
            want.sort_unstable();
            assert_eq!(
                ids, want,
                "molecules must be emitted sorted by read_id (draw {} of 8). If this \
                 fails, the `read_map` drain is unordered again and the emitted \
                 parquet stops being bit-reproducible — see the determinism note \
                 at the drain site.",
                draw
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The two CLI spellings, and the rejections. The spelling is load-bearing: the guard's FATAL message quotes `--polyA-source`, so a parser that only accepted some spelling would name a flag that cannot be typed.
    #[test]
    fn test_parse_poly_a_source_accepts_the_two_forms_and_rejects_the_rest() {
        assert_eq!(
            parse_poly_a_source("bam_tag:pt").unwrap(),
            PolyASource::BamTag(TAG_PT)
        );
        assert_eq!(
            parse_poly_a_source("bam_tag:Zt").unwrap(),
            PolyASource::BamTag(Tag::from(*b"Zt"))
        );
        assert_eq!(parse_poly_a_source(" none ").unwrap(), PolyASource::None);

        // Every one of these must be a message naming the flag, not a panic and not a silent default.
        for bad in [
            "",
            "pt",
            "bam_tag:",
            "bam_tag:p",
            "bam_tag:pts",
            "tsv:",
            "tsv:/tmp/tailfindr.tsv",
            "BAM_TAG:pt",
        ] {
            let err = parse_poly_a_source(bad).expect_err(&format!("{bad:?} must be rejected"));
            let msg = format!("{err:#}");
            assert!(
                msg.contains("--polyA-source"),
                "the rejection for {bad:?} must name the flag to retype, got: {msg}"
            );
        }
    }

    /// The absence of poly(A) is a named error, not an empty graph.
    #[test]
    fn test_poly_a_enters_as_the_bam_tag_pt_and_bad_tags_drop_the_read() {
        use noodles_sam::alignment::record_buf::data::field::Value;

        // Well-formed: the helper writes `pt:i = 100`, and log2(100) = 6.643856.
        let good = synth_record_with_gap("pt64", 0, 0, false, Some(1000), 1, 60);

        // pt:i present but zero -> dropped (a molecule cannot have no tail)
        let mut zero = synth_record_with_gap("pt0", 1, 0, false, Some(1000), 1, 60);
        zero.data_mut().clear();
        zero.data_mut().insert(TAG_PT, Value::from(0i32));

        // pt:i absent entirely -> dropped. This is the case a non-Dorado BAM hits on EVERY record.
        let mut absent = synth_record_with_gap("pt_none", 2, 0, false, Some(1000), 1, 60);
        absent.data_mut().clear();
        absent.data_mut().insert(TAG_NM, Value::from(1i32));

        let path = write_synthetic_bam("pt_contract", vec![good, zero, absent]);
        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

        let ids: Vec<&str> = graph.molecules.iter().map(|m| m.read_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["pt64"],
            "only the read with a positive integer pt:i may survive"
        );
        let want = (100.0_f32).log2();
        assert!(
            (graph.molecules[0].log_pal - want).abs() < 1e-6,
            "y_r must be log2(pt:i): log2(100) = {want}, got {}",
            graph.molecules[0].log_pal
        );
    }

    /// A BAM with no `pt:i` on any record is a HARD ERROR, not an empty graph: every record is dropped at the tag read, so the caller cannot tell "this BAM has no poly(A)" from "this sample has no reads".
    #[test]
    fn test_a_bam_without_pt_is_a_hard_error_not_an_empty_graph() {
        use noodles_sam::alignment::record_buf::data::field::Value;

        let mut a = synth_record_with_gap("no_pt_a", 0, 0, false, Some(1000), 1, 60);
        a.data_mut().clear();
        a.data_mut().insert(TAG_NM, Value::from(1i32));
        let mut b = synth_record_with_gap("no_pt_b", 1, 0, false, Some(1000), 1, 60);
        b.data_mut().clear();
        b.data_mut().insert(TAG_NM, Value::from(1i32));

        let path = write_synthetic_bam("no_pt_at_all", vec![a, b]);
        let err = extract_evidence(&path, 35).expect_err(
            "a BAM with no pt:i must be a hard error: an empty graph here is \
             indistinguishable downstream from a sample with no reads",
        );
        let _ = std::fs::remove_file(&path);

        let msg = format!("{err:#}");
        assert!(
            msg.contains("FATAL") && msg.contains("bam_tag:pt"),
            "the error must name the ruling's guard and the source it was looking for, got: {msg}"
        );
    }

    /// `none` drops nothing, and the global-absence guard must not fire on it: the `None` arm reports a HIT on every record, so `hits == 0` is unreachable. The placeholder must be 1, not 0.
    #[test]
    fn test_the_none_source_keeps_every_read_and_never_trips_the_guard() {
        use noodles_sam::alignment::record_buf::data::field::Value;

        let mut a = synth_record_with_gap("none_a", 0, 0, false, Some(1000), 1, 60);
        a.data_mut().clear();
        a.data_mut().insert(TAG_NM, Value::from(1i32));
        let bam_path = write_synthetic_bam("none_src", vec![a]);

        let graph = extract_evidence_with(&bam_path, 35, &PolyASource::None).unwrap();
        let _ = std::fs::remove_file(&bam_path);

        assert_eq!(
            graph.molecules.len(),
            1,
            "`none` is declared-absent, not declared-invalid: the read must survive"
        );
        assert_eq!(
            graph.molecules[0].log_pal, 0.0,
            "the placeholder is log2(1); the CLI forces poly_a_weight = 0 so it \
             can never reach the E-step"
        );
    }

    #[test]
    fn test_molecule_evidence_layout() {
        // Slice-offset invariant: `candidate_offset + candidate_count` is the next slice start.
        let mol = MoleculeEvidence {
            read_id: "read_1".to_string(),
            log_pal: 6.0,
            candidate_offset: 0,
            candidate_count: 2,
        };
        assert_eq!(mol.candidate_offset + mol.candidate_count as u32, 2);
        assert!((mol.log_pal - 6.0_f32).abs() < f32::EPSILON);
    }

    /// End-to-end test of the alignment-evidence extraction path via the real `extract_evidence`.
    /// Covers the three cases that can silently mis-assign AS: two transcripts with different AS;
    /// the SAME transcript twice with different AS; and a record lacking `AS:i`.
    #[test]
    fn test_extract_evidence_alignment_scores() {
        let path = write_synthetic_bam(
            "as_scores",
            vec![
                // (a) r1 → tx0 (AS 1000, MAPQ 60) and tx1 (AS 900, MAPQ 30).
                synth_record("r1", 0, Some(1000), 2, 60),
                synth_record("r1", 1, Some(900), 5, 30),
                // (b) r2 → tx1 twice: AS 100/MAPQ 0 first, then AS 800/MAPQ 60.
                synth_record("r2", 1, Some(100), 9, 0),
                synth_record("r2", 1, Some(800), 1, 60),
                // (c) r3 → tx2 with no AS:i at all.
                synth_record("r3", 2, None, 4, 10),
            ],
        );

        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

        // Lockstep invariant: every pool has one entry per candidate slot.
        assert_eq!(graph.candidate_as.len(), graph.candidate_pool.len());
        assert_eq!(graph.candidate_nm.len(), graph.candidate_pool.len());
        assert_eq!(graph.candidate_mapq.len(), graph.candidate_pool.len());
        assert_eq!(graph.candidate_5p_gap.len(), graph.candidate_pool.len());
        let total: u64 = graph
            .molecules
            .iter()
            .map(|m| m.candidate_count as u64)
            .sum();
        assert_eq!(total as usize, graph.candidate_pool.len());

        // Molecule order is by read_id, so index by read_id. Every pool is returned by name.
        #[derive(Debug, PartialEq)]
        struct Slices {
            pool: Vec<u32>,
            as_: Vec<i32>,
            nm: Vec<i32>,
            mapq: Vec<u8>,
            gap: Vec<u32>,
        }
        let slice = |read_id: &str| -> Slices {
            let mol = graph
                .molecules
                .iter()
                .find(|m| m.read_id == read_id)
                .unwrap_or_else(|| panic!("{read_id} missing from graph"));
            let o = mol.candidate_offset as usize;
            let n = mol.candidate_count as usize;
            Slices {
                pool: graph.candidate_pool[o..o + n].to_vec(),
                as_: graph.candidate_as[o..o + n].to_vec(),
                nm: graph.candidate_nm[o..o + n].to_vec(),
                mapq: graph.candidate_mapq[o..o + n].to_vec(),
                gap: graph.candidate_5p_gap[o..o + n].to_vec(),
            }
        };

        // (a) Two distinct transcripts with two distinct AS, index-aligned with their transcript ids.
        let Slices {
            pool: txs,
            as_,
            nm,
            mapq,
            gap,
        } = slice("r1");
        let mut order: Vec<usize> = (0..txs.len()).collect();
        order.sort_by_key(|&i| txs[i]);
        assert_eq!(
            order.iter().map(|&i| txs[i]).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            order.iter().map(|&i| as_[i]).collect::<Vec<_>>(),
            vec![1000, 900]
        );
        assert_eq!(order.iter().map(|&i| nm[i]).collect::<Vec<_>>(), vec![2, 5]);
        assert_eq!(
            order.iter().map(|&i| mapq[i]).collect::<Vec<_>>(),
            vec![60, 30]
        );
        assert_eq!(gap, vec![0, 0], "start 1 forward ⇒ g_k = s_k − 1 = 0");

        // (b) Max-AS merge: only ONE candidate slot survives, carrying the winner's evidence.
        let Slices {
            pool: txs,
            as_,
            nm,
            mapq,
            gap,
        } = slice("r2");
        assert_eq!(txs, vec![1]);
        assert_eq!(as_, vec![800]);
        assert_eq!(nm, vec![1]);
        assert_eq!(mapq, vec![60]);
        assert_eq!(gap, vec![0]);

        // (c) Missing AS:i → 0 (uniform A_rk for this read).
        let Slices {
            pool: txs,
            as_,
            nm,
            mapq,
            gap,
        } = slice("r3");
        assert_eq!(txs, vec![2]);
        assert_eq!(as_, vec![0]);
        assert_eq!(nm, vec![4]);
        assert_eq!(mapq, vec![10]);
        assert_eq!(gap, vec![0]);
    }

    /// Pins the frozen 1-based convention for the forward 5' gap `g_k = s_k − 1`: start 6 must give `g_k = 5`, not 6, and `g_k <= 20` is the frozen window edge.
    #[test]
    fn test_5p_gap_is_forward_formula_and_one_based() {
        let path = write_synthetic_bam(
            "gap5p",
            vec![
                // Forward, g_k = 5 → start 6. Pins the 1-based convention: `s_k − 1 = 5`, NOT 6.
                synth_record_with_gap("f5", 0, 5, false, Some(1000), 1, 60),
                // Forward, g_k = 20 → start 21. Exactly the frozen `g_k <= 20` window edge.
                synth_record_with_gap("f20", 1, 20, false, Some(900), 1, 60),
            ],
        );

        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(graph.candidate_5p_gap.len(), graph.candidate_pool.len());

        let gap_of = |read_id: &str| -> u32 {
            let mol = graph
                .molecules
                .iter()
                .find(|m| m.read_id == read_id)
                .unwrap_or_else(|| panic!("{read_id} missing from graph"));
            let o = mol.candidate_offset as usize;
            assert_eq!(mol.candidate_count, 1, "{read_id} must have one candidate");
            graph.candidate_5p_gap[o]
        };

        assert_eq!(gap_of("f5"), 5, "forward ⇒ g_k = s_k − 1, so 6 − 1 = 5");
        assert_eq!(gap_of("f20"), 20, "forward start 21 ⇒ g_k = 20 exactly");
    }

    /// Strand is a REFUSED input, not a handled case: the frozen spec was validated on a forward-only population, and this crate reads no strand flag at all.
    #[test]
    fn test_reverse_alignment_is_refused_rather_than_given_the_forward_formula() {
        let path = write_synthetic_bam(
            "strand_refused",
            vec![
                // reverse record: start 1 on REF_LEN (2000) nt, ending 30 bases SHORT of the 3' end.
                synth_record_with_gap("rev", 0, 30, true, Some(800), 1, 60),
            ],
        );
        let err = extract_evidence(&path, 35).expect_err(
            "a reverse-strand record must be refused: this crate reads no strand \
             flag and applies no strand correction, so `g_k = start_pos - 1` on a \
             reverse alignment is not a value anything downstream can use",
        );
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("reverse-strand") && msg.contains("rev"),
            "the error must NAME the condition and the offending read, so an \
             operator can tell this apart from the other hard errors \
             extract_evidence raises. Got: {msg}"
        );
        let _ = std::fs::remove_file(&path);

        let fwd = write_synthetic_bam(
            "strand_forward_control",
            vec![synth_record_with_gap("fwd", 0, 30, false, Some(800), 1, 60)],
        );
        let graph = extract_evidence(&fwd, 35).expect(
            "the forward population is the validated one and must keep working; \
             if this fails the guard is not what rejected the reverse fixture",
        );
        let _ = std::fs::remove_file(&fwd);
        assert!(
            graph.molecules.iter().any(|m| m.read_id == "fwd"),
            "control: the forward read must be in the graph"
        );
    }

    /// The 5' gap must follow the record that WON the max-AS merge: the frozen spec defines `s_k` as the start of the read's BEST alignment to candidate k.
    #[test]
    fn test_5p_gap_follows_the_max_as_winner() {
        let path = write_synthetic_bam(
            "gap5p_merge",
            vec![
                // Same (read, transcript) twice: the second record has the higher AS.
                synth_record_with_gap("r1", 0, 5, false, Some(100), 1, 0),
                synth_record_with_gap("r1", 0, 40, false, Some(800), 1, 60),
            ],
        );

        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

        let mol = graph.molecules.iter().find(|m| m.read_id == "r1").unwrap();
        let o = mol.candidate_offset as usize;
        assert_eq!(mol.candidate_count, 1, "the repeat (read, tx) pair merges");
        assert_eq!(graph.candidate_as[o], 800);
        assert_eq!(
            graph.candidate_5p_gap[o], 40,
            "the stored gap must be the max-AS record's (40), not the \
             first-seen one's (5)"
        );
    }

    /// `candidate_count_histogram` must be exercised from inside this file.
    #[test]
    fn test_candidate_count_histogram() {
        // Build a FlatEvidenceGraph by hand: 4 molecules with candidate_counts {1, 2, 2, 3}.
        let molecules = vec![
            MoleculeEvidence {
                read_id: "r1".into(),
                log_pal: 6.0,
                candidate_offset: 0,
                candidate_count: 1,
            },
            MoleculeEvidence {
                read_id: "r2".into(),
                log_pal: 6.5,
                candidate_offset: 1,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "r3".into(),
                log_pal: 7.0,
                candidate_offset: 3,
                candidate_count: 2,
            },
            MoleculeEvidence {
                read_id: "r4".into(),
                log_pal: 7.5,
                candidate_offset: 5,
                candidate_count: 3,
            },
        ];
        let candidate_pool = vec![0, 1, 2, 3, 4, 5, 6, 7]; // 8 = 1 + 2 + 2 + 3
        let graph = FlatEvidenceGraph {
            molecules,
            candidate_pool,
            candidate_as: vec![0; 8],
            candidate_nm: vec![0; 8],
            candidate_mapq: vec![0; 8],
            candidate_5p_gap: vec![0; 8],
            transcript_names: vec![],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };

        let hist = graph.candidate_count_histogram();
        // Sorted ascending by candidate_count; types are (u16, u64).
        assert_eq!(
            hist,
            vec![(1_u16, 1_u64), (2_u16, 2_u64), (3_u16, 1_u64)],
            "candidate_count_histogram must bucket molecules correctly and sort by candidate_count"
        );
    }

    /// Guards the memory contract of the flatten step: the five pools are reserved at the summed candidate count, so no pool carries a spare slot. The total is deliberately 6.
    #[test]
    fn test_candidate_pools_are_exactly_sized() {
        let path = write_synthetic_bam(
            "exact_capacity",
            vec![
                // 3 molecules x 2 distinct transcripts = 6 candidate slots.
                synth_record("r1", 0, Some(1000), 2, 60),
                synth_record("r1", 1, Some(900), 5, 30),
                synth_record("r2", 0, Some(800), 1, 60),
                synth_record("r2", 1, Some(700), 6, 30),
                synth_record("r3", 0, Some(600), 3, 60),
                synth_record("r3", 1, Some(500), 7, 30),
            ],
        );

        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(graph.molecules.len(), 3);
        assert_eq!(graph.candidate_pool.len(), 6);
        assert_eq!(graph.candidate_pool.capacity(), graph.candidate_pool.len());
        assert_eq!(graph.candidate_as.capacity(), graph.candidate_as.len());
        assert_eq!(graph.candidate_nm.capacity(), graph.candidate_nm.len());
        assert_eq!(graph.candidate_mapq.capacity(), graph.candidate_mapq.len());
        assert_eq!(
            graph.candidate_5p_gap.capacity(),
            graph.candidate_5p_gap.len()
        );
    }

    /// Histogram on an empty graph returns an empty vec.
    #[test]
    fn test_candidate_count_histogram_empty() {
        let graph = FlatEvidenceGraph {
            molecules: vec![],
            candidate_pool: vec![],
            candidate_as: vec![],
            candidate_nm: vec![],
            candidate_mapq: vec![],
            candidate_5p_gap: vec![],
            transcript_names: vec![],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };
        assert!(graph.candidate_count_histogram().is_empty());
    }

    /// The 3' TES filter, at its exact boundary: `ref_end + tes_slack < ref_len` -> drop, so a read ending exactly `tes_slack` bases short of the transcript's 3' end is KEPT and one ending `tes_slack + 1` short is DROPPED.
    #[test]
    fn test_tes_filter_boundary_is_exact_and_the_ledger_closes() {
        const TES_SLACK: usize = 35;

        // One read at the last allowed offset (kept), one one base further out (dropped).
        let recs = vec![
            synth_record_short_of_tes("at_boundary", 0, 1, TES_SLACK, Some(1000), 1, 60),
            synth_record_short_of_tes("past_boundary", 0, 1, TES_SLACK + 1, Some(1000), 1, 60),
            synth_record("full_length", 0, Some(1000), 1, 60),
        ];
        let path = write_synthetic_bam("tes_boundary", recs);
        let graph = extract_evidence(&path, TES_SLACK).unwrap();
        std::fs::remove_file(&path).ok();

        let s = graph.stats;
        assert_eq!(s.records_entered, 3, "{s:?}");
        assert_eq!(
            s.dropped_tes_window, 1,
            "exactly one read is past the window: {s:?}"
        );
        assert_eq!(s.records_with_candidates, 2, "{s:?}");
        assert_eq!(s.ledger_deficit(), 0, "ledger must close: {s:?}");
        assert_eq!(s.dropped_total(), 1, "{s:?}");

        // The two surviving reads are the ones the boundary predicts.
        let kept: Vec<&str> = graph.molecules.iter().map(|m| m.read_id.as_str()).collect();
        assert!(kept.contains(&"at_boundary"), "kept {kept:?}");
        assert!(kept.contains(&"full_length"), "kept {kept:?}");
        assert!(!kept.contains(&"past_boundary"), "kept {kept:?}");

        // Buckets this fixture cannot exercise. Pinned non-zero by `test_flag_drop_buckets_are_observable`.
        assert_eq!(s.dropped_unmapped, 0, "{s:?}");
        assert_eq!(s.dropped_supplementary, 0, "{s:?}");
        assert_eq!(s.dropped_no_poly_a_tag, 0, "{s:?}");
        assert_eq!(s.dropped_invalid_poly_a, 0, "{s:?}");
    }

    /// The two FLAG-only drop buckets are reachable and non-zero: the mutation target for the increments.
    #[test]
    fn test_flag_drop_buckets_are_observable() {
        use noodles_sam::alignment::record::Flags;

        let recs = vec![
            synth_record_with_flags("unmapped", 0, Flags::UNMAPPED, Some(1000)),
            synth_record_with_flags("supp", 0, Flags::SUPPLEMENTARY, Some(1000)),
            synth_record("survivor", 0, Some(1000), 1, 60),
        ];
        let path = write_synthetic_bam("flag_buckets", recs);
        let graph = extract_evidence(&path, 35).unwrap();
        std::fs::remove_file(&path).ok();

        let s = graph.stats;
        assert_eq!(s.records_entered, 3, "{s:?}");
        assert_eq!(s.dropped_unmapped, 1, "{s:?}");
        assert_eq!(s.dropped_supplementary, 1, "{s:?}");
        assert_eq!(s.records_with_candidates, 1, "{s:?}");
        assert_eq!(s.ledger_deficit(), 0, "ledger must close: {s:?}");
        assert_eq!(
            graph.molecules.len(),
            1,
            "only the survivor reaches the graph"
        );
    }

    // RED PROOF for the production `ledger_deficit() == 0` assertion. Every other test in this
    // module exercises that assertion's GREEN side; nothing had ever made it go red.
    //
    // These fixtures are raw BAM BYTES rather than `write_synthetic_bam` records, and they have
    // to be: the values under test are "no reference" (refID = -1), "no position" (pos = -1) and
    // "no read name" (`*\0`), none of which is representable in noodles' record API.
    //
    // Each file is ONE record, identical apart from the one field named: FLAG 0 (mapped, forward,
    // primary), 20M at offset 980 of a 1000 bp `tx1`, `pt:i:70`, `tx1` the only reference.

    /// `refID = -1` on a record whose FLAG says mapped: the exit is UPSTREAM of the RHS term.
    #[test]
    #[should_panic(expected = "evidence filter ledger does not close")]
    fn test_ledger_guard_panics_on_a_record_with_no_reference_sequence_id() {
        let path = write_raw_bam("notid", BAM_NO_REFID);
        // No cleanup: the panic unwinds past it.
        let _ = extract_evidence(&path, 35);
    }

    /// `pos = -1`, so `alignment_start().get()` yields nothing and the `_ => continue` is taken.
    #[test]
    #[should_panic(expected = "evidence filter ledger does not close")]
    fn test_ledger_guard_panics_on_a_record_with_no_alignment_start() {
        let path = write_raw_bam("nopos", BAM_NO_POS);
        let _ = extract_evidence(&path, 35);
    }

    /// The third of the three malformed-record exits. QNAME is the two bytes `*\0`; noodles'
    /// `fields::name()` returns `None` for exactly two byte patterns -- `*\0` and `\0\0` -- and for
    /// nothing else, so this is the only read-name that reaches the `None => continue`.
    #[test]
    #[should_panic(expected = "evidence filter ledger does not close")]
    fn test_ledger_guard_panics_on_a_record_with_no_read_name() {
        let path = write_raw_bam("nameless", BAM_NO_NAME);
        let _ = extract_evidence(&path, 35);
    }

    /// The pre-change behaviour, kept as a fixture-level assertion. It does NOT run
    /// `extract_evidence` -- that panics now, by design. The RHS term was incremented before the
    /// name check, so a record dropped there was counted on BOTH sides: deficit 0, no molecule.
    #[test]
    fn test_the_old_rhs_placement_balanced_the_ledger_while_losing_the_record() {
        // The bug, arithmetic only: the increment ran BEFORE the name check, so a record dropped there was counted as one that reached the push, and incremented NO bucket of its own.
        let s = FilterCounts {
            records_entered: 1,
            records_with_candidates: 1,
            ..Default::default()
        };
        assert_eq!(
            s.dropped_total(),
            0,
            "nothing recorded the drop -- that is why the deficit was blind: {s:?}"
        );
        assert_eq!(
            s.ledger_deficit(),
            0,
            "THE BLINDNESS, ARITHMETICALLY: conservation holds while the record \
             is gone. This is what the fixture returned before the \
             increment moved: Ok, deficit 0, zero molecules. {s:?}"
        );

        // And the fix: with the increment downstream of :642, the record leaves the RHS term uncounted.
        let fixed = FilterCounts {
            records_entered: 1,
            ..Default::default()
        };
        // records_with_candidates stays 0 -- the drop is upstream now.
        assert_eq!(
            fixed.ledger_deficit(),
            1,
            "after the move the same fixture trips the assertion: {fixed:?}"
        );
    }

    const BAM_NO_REFID: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02,
        0x00, 0x83, 0x00, 0x73, 0x72, 0xf4, 0x65, 0xd4, 0x61, 0x60, 0x60, 0x70, 0xf0, 0x70, 0xe1,
        0x0c, 0xf3, 0xb3, 0x32, 0xd4, 0x33, 0xe3, 0x0c, 0xf6, 0xb7, 0x4a, 0xce, 0xcf, 0x2f, 0x4a,
        0xc9, 0xcc, 0x4b, 0x2c, 0x49, 0xe5, 0x72, 0x08, 0x0e, 0xe4, 0x0c, 0xf6, 0xb3, 0x2a, 0xa9,
        0x30, 0xe4, 0xf4, 0x01, 0xca, 0x1b, 0x18, 0x18, 0x70, 0x31, 0x02, 0xd5, 0xb3, 0x00, 0x31,
        0x50, 0x8c, 0xe1, 0x05, 0x33, 0x03, 0x83, 0x3f, 0x90, 0xfd, 0x1f, 0x08, 0xae, 0x00, 0xd9,
        0x6c, 0x36, 0x0c, 0x0c, 0x20, 0x79, 0x11, 0xa8, 0x18, 0x03, 0x14, 0x14, 0xa5, 0x26, 0xa6,
        0x38, 0x32, 0x38, 0x00, 0xa5, 0x84, 0x3c, 0x60, 0x50, 0x0e, 0x0b, 0x28, 0x28, 0xc9, 0x74,
        0x03, 0x2a, 0x07, 0x00, 0xbb, 0x76, 0x14, 0x0f, 0x97, 0x00, 0x00, 0x00, 0x1f, 0x8b, 0x08,
        0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00, 0x1b, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    const BAM_NO_POS: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02,
        0x00, 0x81, 0x00, 0x73, 0x72, 0xf4, 0x65, 0xd4, 0x61, 0x60, 0x60, 0x70, 0xf0, 0x70, 0xe1,
        0x0c, 0xf3, 0xb3, 0x32, 0xd4, 0x33, 0xe3, 0x0c, 0xf6, 0xb7, 0x4a, 0xce, 0xcf, 0x2f, 0x4a,
        0xc9, 0xcc, 0x4b, 0x2c, 0x49, 0xe5, 0x72, 0x08, 0x0e, 0xe4, 0x0c, 0xf6, 0xb3, 0x2a, 0xa9,
        0x30, 0xe4, 0xf4, 0x01, 0xca, 0x1b, 0x18, 0x18, 0x70, 0x31, 0x02, 0xd5, 0xb3, 0x00, 0x31,
        0x50, 0x8c, 0xe1, 0x05, 0x33, 0x03, 0x83, 0x3f, 0x03, 0x04, 0xfc, 0x07, 0x02, 0x36, 0x1b,
        0x06, 0x06, 0x90, 0xbc, 0x08, 0x94, 0x0f, 0x95, 0x62, 0x28, 0x4a, 0x4d, 0x4c, 0x71, 0x64,
        0x70, 0x00, 0x4a, 0x09, 0x79, 0xc0, 0xa0, 0x1c, 0x16, 0x50, 0x50, 0x92, 0xe9, 0x06, 0x54,
        0x0e, 0x00, 0x84, 0xdd, 0x2e, 0x33, 0x97, 0x00, 0x00, 0x00, 0x1f, 0x8b, 0x08, 0x04, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00, 0x1b, 0x00, 0x03, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    const BAM_NO_NAME: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02,
        0x00, 0x7f, 0x00, 0x73, 0x72, 0xf4, 0x65, 0xd4, 0x61, 0x60, 0x60, 0x70, 0xf0, 0x70, 0xe1,
        0x0c, 0xf3, 0xb3, 0x32, 0xd4, 0x33, 0xe3, 0x0c, 0xf6, 0xb7, 0x4a, 0xce, 0xcf, 0x2f, 0x4a,
        0xc9, 0xcc, 0x4b, 0x2c, 0x49, 0xe5, 0x72, 0x08, 0x0e, 0xe4, 0x0c, 0xf6, 0xb3, 0x2a, 0xa9,
        0x30, 0xe4, 0xf4, 0x01, 0xca, 0x1b, 0x18, 0x18, 0x70, 0x31, 0x02, 0xd5, 0xb3, 0x00, 0x31,
        0x50, 0x8c, 0xe1, 0x05, 0x33, 0x03, 0x83, 0x37, 0x03, 0x04, 0x5c, 0x01, 0xb2, 0x99, 0x6c,
        0x18, 0x18, 0x40, 0xf2, 0x22, 0x40, 0xfc, 0x1f, 0x08, 0xa0, 0x52, 0x0c, 0x5a, 0x0c, 0x0e,
        0x40, 0x61, 0x21, 0x0f, 0x18, 0x94, 0xc3, 0x02, 0x0a, 0x4a, 0x32, 0xdd, 0x80, 0x4a, 0x01,
        0x63, 0x44, 0x63, 0x8a, 0x93, 0x00, 0x00, 0x00, 0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    fn write_raw_bam(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "telescope_evidence_{}_test_{}.bam",
            tag,
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }
}
