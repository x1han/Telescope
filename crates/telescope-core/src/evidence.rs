use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context, Result};
use noodles_bam as bam;
use noodles_sam::alignment::record::data::field::Tag;
use noodles_sam::alignment::Record as _;

const TAG_PT: Tag = Tag::new(b'p', b't');

const TAG_AS: Tag = Tag::new(b'A', b'S');

const TAG_NM: Tag = Tag::new(b'N', b'M');

#[derive(Debug, Clone)]
pub struct MoleculeEvidence {
    pub read_id: String,
    pub log_pal: f32,
    pub candidate_offset: u32,
    pub candidate_count: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FilterCounts {
    pub records_entered: u64,
    pub dropped_unmapped: u64,

    pub dropped_supplementary: u64,
    pub dropped_no_poly_a_tag: u64,

    pub dropped_invalid_poly_a: u64,

    pub dropped_no_span: u64,
    pub dropped_empty_span: u64,

    pub dropped_tes_window: u64,

    pub records_with_candidates: u64,
}

impl FilterCounts {
    pub fn dropped_total(&self) -> u64 {
        self.dropped_unmapped
            + self.dropped_supplementary
            + self.dropped_no_poly_a_tag
            + self.dropped_invalid_poly_a
            + self.dropped_no_span
            + self.dropped_empty_span
            + self.dropped_tes_window
    }

    pub fn ledger_deficit(&self) -> i64 {
        self.records_entered as i64 - (self.dropped_total() + self.records_with_candidates) as i64
    }
}

#[derive(Debug)]
pub struct FlatEvidenceGraph {
    pub molecules: Vec<MoleculeEvidence>,

    pub candidate_pool: Vec<u32>,

    pub candidate_as: Vec<i32>,

    pub candidate_nm: Vec<i32>,

    pub candidate_mapq: Vec<u8>,

    pub candidate_5p_gap: Vec<u32>,

    pub candidate_3p_gap: Vec<u32>,
    pub transcript_names: Vec<String>,

    pub poly_a_is_placeholder: bool,

    pub stats: FilterCounts,
}

pub(crate) fn assert_candidate_pools_in_lockstep(graph: &FlatEvidenceGraph) {
    assert_eq!(
        graph.candidate_pool.len(),
        graph.candidate_as.len(),
        "candidate_as desynchronised from candidate_pool: a missing `push`/`extend` \
         reads the alignment score of the WRONG candidate"
    );
    assert_eq!(
        graph.candidate_pool.len(),
        graph.candidate_nm.len(),
        "candidate_nm desynchronised from candidate_pool: a missing `push`/`extend` \
         reads the edit distance of the WRONG candidate"
    );
    assert_eq!(
        graph.candidate_pool.len(),
        graph.candidate_mapq.len(),
        "candidate_mapq desynchronised from candidate_pool: a missing `push`/`extend` \
         reads the mapping quality of the WRONG candidate"
    );
    assert_eq!(
        graph.candidate_pool.len(),
        graph.candidate_5p_gap.len(),
        "candidate_5p_gap desynchronised from candidate_pool: a missing \
         `push`/`extend` reads the 5' gap of the WRONG candidate"
    );
    assert_eq!(
        graph.candidate_pool.len(),
        graph.candidate_3p_gap.len(),
        "candidate_3p_gap desynchronised from candidate_pool: a missing \
         `push`/`extend` reads the 3' gap of the WRONG candidate"
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolyASource {
    BamTag(Tag),

    None,
}

pub fn parse_poly_a_source(spec: &str) -> Result<PolyASource> {
    let spec = spec.trim();
    if spec == "none" {
        return Ok(PolyASource::None);
    }
    if let Some(tag) = spec.strip_prefix("bam_tag:") {
        let bytes = tag.as_bytes();

        if bytes.len() != 2 || !bytes.iter().all(u8::is_ascii_alphanumeric) {
            anyhow::bail!("--poly-a-source bam_tag: needs a 2-character SAM tag, got `{tag}`");
        }
        return Ok(PolyASource::BamTag(Tag::from([bytes[0], bytes[1]])));
    }
    anyhow::bail!("--poly-a-source must be one of `bam_tag:<TAG>` or `none`; got `{spec}`");
}

const POLY_A_PROBE_RECORDS: usize = 10_000;

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

    let mut poly_a_records_seen: u64 = 0;
    let mut poly_a_hits: u64 = 0;

    let mut fc = FilterCounts::default();

    struct TempReadEntry {
        poly_a_len: i32,
        candidates: Vec<CandidateAlign>,
    }

    struct CandidateAlign {
        tx_id: u32,
        as_score: i32,
        nm: i32,
        mapq: u8,
        gap5p: u32,
        gap3p: u32,
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

        if flags.is_supplementary() {
            fc.dropped_supplementary += 1;
            continue;
        }

        poly_a_records_seen += 1;

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
                 `--poly-a-source none` for a pure-alignment run. \
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
                        continue;
                    }
                },
                _ => {
                    fc.dropped_no_poly_a_tag += 1;
                    continue;
                }
            },
            PolyASource::None => 1,
        };
        if poly_a_len > 0 {
            poly_a_hits += 1;
        }

        let ref_id = match record.reference_sequence_id() {
            Some(Ok(id)) => id,
            _ => continue,
        };
        let ref_len = transcript_lengths[ref_id];

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
            continue;
        }
        let ref_end = start_pos + span - 1;

        if ref_end + tes_slack < ref_len {
            fc.dropped_tes_window += 1;
            continue;
        }

        let gap5p: u32 = u32::try_from(start_pos.saturating_sub(1)).unwrap_or(u32::MAX);

        let gap3p: u32 = u32::try_from(ref_len.saturating_sub(ref_end)).unwrap_or(u32::MAX);

        let read_id = match record.name() {
            Some(name) => String::from_utf8_lossy(name).to_string(),
            None => continue,
        };

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

        let tx_id = ref_id as u32;
        let align = CandidateAlign {
            tx_id,
            as_score,
            nm,
            mapq,
            gap5p,
            gap3p,
        };

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
             `--poly-a-source none` for a pure-alignment run. \
             Refusing to continue: returning an empty graph here \
             is indistinguishable downstream from a sample with no reads."
        );
    }

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
    let mut candidate_3p_gap: Vec<u32> = Vec::with_capacity(total_candidates);

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
            candidate_3p_gap.push(c.gap3p);
        }

        molecules.push(MoleculeEvidence {
            read_id,

            log_pal: (entry.poly_a_len.max(1) as f32).log2(),
            candidate_offset,
            candidate_count,
        });
    }

    let graph = FlatEvidenceGraph {
        molecules,
        candidate_pool,
        candidate_as,
        candidate_nm,
        candidate_mapq,
        candidate_5p_gap,
        candidate_3p_gap,
        transcript_names,
        poly_a_is_placeholder: matches!(poly_a, PolyASource::None),
        stats: fc,
    };
    assert_candidate_pools_in_lockstep(&graph);

    assert_eq!(
        fc.ledger_deficit(),
        0,
        "evidence filter ledger does not close: {fc:?} -- a record left the \
         stream through an uncounted path (a new drop site without its counter, \
         or an early return between the filters and the candidate push)"
    );

    Ok(graph)
}

impl FlatEvidenceGraph {
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
#[doc(hidden)]
pub mod test_support {
    use super::*;

    use std::num::NonZero;

    use noodles_core::Position;

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

    pub const REF_LEN: usize = 2000;

    pub fn synth_record(
        name: &str,
        ref_id: usize,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        synth_record_with_gap(name, ref_id, 0, false, as_score, nm, mapq)
    }

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

    #[test]
    fn test_gap5p_is_invariant_to_the_leading_cigar_ops() {
        use Kind::*;
        let gap = 50usize;

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

    #[test]
    fn test_gap5p_matches_an_independent_cdna_backward_computation() {
        use Kind::*;

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

    #[test]
    fn test_supplementary_alignments_are_dropped_rather_than_merged() {
        use Kind::*;

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

    pub fn synth_record_past_tes(
        name: &str,
        ref_id: usize,
        overhang: usize,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        let cigar: Cigar = [Op::new(Kind::Match, REF_LEN + overhang)]
            .into_iter()
            .collect();
        let mut tags = vec![(TAG_PT, Value::from(100i32)), (TAG_NM, Value::from(nm))];
        if let Some(a) = as_score {
            tags.push((TAG_AS, Value::from(a)));
        }
        RecordBuf::builder()
            .set_name(name)
            .set_flags(Flags::empty())
            .set_reference_sequence_id(ref_id)
            .set_alignment_start(Position::new(1).expect("start is 1-based and non-zero"))
            .set_mapping_quality(MappingQuality::new(mapq).unwrap())
            .set_cigar(cigar)
            .set_data(tags.into_iter().collect())
            .build()
    }

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

    pub fn write_synthetic_bam(tag: &str, records: Vec<RecordBuf>) -> std::path::PathBuf {
        write_synthetic_bam_named(tag, &["tx0", "tx1", "tx2"], records)
    }

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

    #[test]
    fn test_the_in_loop_probe_catches_poly_a_that_only_starts_after_10000_records() {
        use noodles_sam::alignment::record_buf::data::field::Value;

        let mut recs = Vec::new();
        for i in 0..(POLY_A_PROBE_RECORDS + 1) {
            let mut r = synth_record_with_gap(&format!("late_{i}"), 0, 0, false, Some(1000), 1, 60);
            r.data_mut().clear();
            if i < POLY_A_PROBE_RECORDS {
                r.data_mut().insert(TAG_NM, Value::from(1i32));
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

    #[test]
    fn test_the_placeholder_flag_is_source_derived_not_inferred_from_log_pal() {
        use noodles_sam::alignment::record_buf::data::field::Value;

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

    #[test]
    fn test_molecules_are_emitted_in_read_id_order() {
        use noodles_sam::alignment::record_buf::data::field::Value;
        let names = ["r7", "r2", "r5", "r1", "r8", "r3", "r6", "r4"];
        let recs: Vec<_> = names
            .iter()
            .map(|n| {
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
                msg.contains("--poly-a-source"),
                "the rejection for {bad:?} must name the flag to retype, got: {msg}"
            );
        }
    }

    #[test]
    fn test_poly_a_enters_as_the_bam_tag_pt_and_bad_tags_drop_the_read() {
        use noodles_sam::alignment::record_buf::data::field::Value;

        let good = synth_record_with_gap("pt64", 0, 0, false, Some(1000), 1, 60);

        let mut zero = synth_record_with_gap("pt0", 1, 0, false, Some(1000), 1, 60);
        zero.data_mut().clear();
        zero.data_mut().insert(TAG_PT, Value::from(0i32));

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
            "the error must name the input guard and the source it was looking for, got: {msg}"
        );
    }

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
        let mol = MoleculeEvidence {
            read_id: "read_1".to_string(),
            log_pal: 6.0,
            candidate_offset: 0,
            candidate_count: 2,
        };
        assert_eq!(mol.candidate_offset + mol.candidate_count as u32, 2);
        assert!((mol.log_pal - 6.0_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extract_evidence_alignment_scores() {
        let path = write_synthetic_bam(
            "as_scores",
            vec![
                synth_record("r1", 0, Some(1000), 2, 60),
                synth_record("r1", 1, Some(900), 5, 30),
                synth_record("r2", 1, Some(100), 9, 0),
                synth_record("r2", 1, Some(800), 1, 60),
                synth_record("r3", 2, None, 4, 10),
            ],
        );

        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

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

    #[test]
    fn test_5p_gap_is_forward_formula_and_one_based() {
        let path = write_synthetic_bam(
            "gap5p",
            vec![
                synth_record_with_gap("f5", 0, 5, false, Some(1000), 1, 60),
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

    #[test]
    fn test_reverse_alignment_is_refused_rather_than_given_the_forward_formula() {
        let path = write_synthetic_bam(
            "strand_refused",
            vec![synth_record_with_gap("rev", 0, 30, true, Some(800), 1, 60)],
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

    #[test]
    fn test_5p_gap_follows_the_max_as_winner() {
        let path = write_synthetic_bam(
            "gap5p_merge",
            vec![
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

    #[test]
    fn test_3p_gap_is_ref_len_minus_ref_end_clamped_at_zero() {
        const TES_SLACK: usize = 35;
        let path = write_synthetic_bam(
            "gap3p",
            vec![
                synth_record_short_of_tes("at_tes", 0, 1, 0, Some(1000), 1, 60),
                synth_record_short_of_tes("short20", 1, 1, 20, Some(900), 1, 60),
                synth_record_short_of_tes("short35", 2, 1, 35, Some(800), 1, 60),
                synth_record_past_tes("past_tes", 0, 100, Some(700), 1, 60),
            ],
        );

        let graph = extract_evidence(&path, TES_SLACK).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(
            graph.candidate_3p_gap.len(),
            graph.candidate_pool.len(),
            "the 3' gap pool must be in lockstep with the candidate pool"
        );

        let gap_of = |read_id: &str| -> u32 {
            let mol = graph
                .molecules
                .iter()
                .find(|m| m.read_id == read_id)
                .unwrap_or_else(|| panic!("{read_id} missing from graph"));
            let o = mol.candidate_offset as usize;
            assert_eq!(mol.candidate_count, 1, "{read_id} must have one candidate");
            graph.candidate_3p_gap[o]
        };

        assert_eq!(
            gap_of("at_tes"),
            0,
            "an alignment ending ON the 3' end has d_k = 0"
        );
        assert_eq!(
            gap_of("short20"),
            20,
            "d_k = ref_len - ref_end = 2000 - 1980 = 20"
        );
        assert_eq!(
            gap_of("short35"),
            35,
            "d_k = 35 = tes_slack, the BOUND the 3' TES filter implies"
        );
        assert_eq!(
            gap_of("past_tes"),
            0,
            "an alignment that runs PAST the 3' end has MAXIMAL 3' support: d_k \
             must be clamped to 0, not wrapped to a huge value"
        );
    }

    #[test]
    fn test_the_3p_gap_is_bounded_by_tes_slack() {
        const TES_SLACK: usize = 35;
        let path = write_synthetic_bam(
            "gap3p_bound",
            vec![
                synth_record_short_of_tes("inside", 0, 1, 35, Some(1000), 1, 60),
                synth_record_short_of_tes("outside", 1, 1, 36, Some(900), 1, 60),
            ],
        );

        let graph = extract_evidence(&path, TES_SLACK).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(
            graph.molecules.iter().any(|m| m.read_id == "inside"),
            "a read ending exactly tes_slack bases short must survive"
        );
        assert!(
            !graph.molecules.iter().any(|m| m.read_id == "outside"),
            "a read ending tes_slack + 1 bases short must be dropped by the 3' \
             TES filter, so no candidate with d_k > tes_slack can exist"
        );
        assert_eq!(
            graph.stats.dropped_tes_window, 1,
            "the dropped record must be counted in the 3' TES bucket"
        );
    }

    #[test]
    fn test_candidate_count_histogram() {
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
        let candidate_pool = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let graph = FlatEvidenceGraph {
            molecules,
            candidate_pool,
            candidate_as: vec![0; 8],
            candidate_nm: vec![0; 8],
            candidate_mapq: vec![0; 8],
            candidate_5p_gap: vec![0; 8],
            candidate_3p_gap: vec![0; 8],
            transcript_names: vec![],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };

        let hist = graph.candidate_count_histogram();

        assert_eq!(
            hist,
            vec![(1_u16, 1_u64), (2_u16, 2_u64), (3_u16, 1_u64)],
            "candidate_count_histogram must bucket molecules correctly and sort by candidate_count"
        );
    }

    #[test]
    fn test_candidate_pools_are_exactly_sized() {
        let path = write_synthetic_bam(
            "exact_capacity",
            vec![
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

    #[test]
    fn test_candidate_count_histogram_empty() {
        let graph = FlatEvidenceGraph {
            molecules: vec![],
            candidate_pool: vec![],
            candidate_as: vec![],
            candidate_nm: vec![],
            candidate_mapq: vec![],
            candidate_5p_gap: vec![],
            candidate_3p_gap: vec![],
            transcript_names: vec![],
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };
        assert!(graph.candidate_count_histogram().is_empty());
    }

    #[test]
    fn test_tes_filter_boundary_is_exact_and_the_ledger_closes() {
        const TES_SLACK: usize = 35;

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

        let kept: Vec<&str> = graph.molecules.iter().map(|m| m.read_id.as_str()).collect();
        assert!(kept.contains(&"at_boundary"), "kept {kept:?}");
        assert!(kept.contains(&"full_length"), "kept {kept:?}");
        assert!(!kept.contains(&"past_boundary"), "kept {kept:?}");

        assert_eq!(s.dropped_unmapped, 0, "{s:?}");
        assert_eq!(s.dropped_supplementary, 0, "{s:?}");
        assert_eq!(s.dropped_no_poly_a_tag, 0, "{s:?}");
        assert_eq!(s.dropped_invalid_poly_a, 0, "{s:?}");
    }

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

    #[test]
    #[should_panic(expected = "evidence filter ledger does not close")]
    fn test_ledger_guard_panics_on_a_record_with_no_reference_sequence_id() {
        let path = write_raw_bam("notid", BAM_NO_REFID);

        let _ = extract_evidence(&path, 35);
    }

    #[test]
    #[should_panic(expected = "evidence filter ledger does not close")]
    fn test_ledger_guard_panics_on_a_record_with_no_alignment_start() {
        let path = write_raw_bam("nopos", BAM_NO_POS);
        let _ = extract_evidence(&path, 35);
    }

    #[test]
    #[should_panic(expected = "evidence filter ledger does not close")]
    fn test_ledger_guard_panics_on_a_record_with_no_read_name() {
        let path = write_raw_bam("nameless", BAM_NO_NAME);
        let _ = extract_evidence(&path, 35);
    }

    #[test]
    fn test_a_record_dropped_before_the_name_check_is_counted_on_both_sides() {
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
        assert_eq!(s.ledger_deficit(), 0, "deficit check: {s:?}");

        let fixed = FilterCounts {
            records_entered: 1,
            ..Default::default()
        };

        assert_eq!(
            fixed.ledger_deficit(),
            1,
            "the same fixture trips the assertion: {fixed:?}"
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
