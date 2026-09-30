//! Module 2: Transcriptome Evidence Extractor
//!
//! Streams a minimap2 transcriptome BAM (AIDRS-aligned), extracts Dorado `pt:i`
//! poly(A) tail length per record plus per-candidate alignment evidence (`AS:i`,
//! `NM:i`, MAPQ), filters by 3' TES proximity, and constructs a flat indexed
//! candidate graph for TEL-EM (Module 3).
//!
//! Input contract (spec v0.1.6 F1): one BAM file, AIDRS-aligned transcriptome
//! BAM carrying `pt:i` tag per record. No Dorado BAM dependency.
//!
//! Source spec: `workspace/specs/telescope-v0.1.6.md` §Module 2.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context, Result};
use noodles_bam as bam;
use noodles_sam::alignment::record::data::field::Tag;
use noodles_sam::alignment::Record as _;

/// Dorado poly(A) tail length SAM tag: `pt:i`.
const TAG_PT: Tag = Tag::new(b'p', b't');

/// minimap2 alignment score SAM tag: `AS:i` (primary; drives Module 3's A_rk).
const TAG_AS: Tag = Tag::new(b'A', b'S');

/// Edit distance to reference SAM tag: `NM:i` (audit only).
const TAG_NM: Tag = Tag::new(b'N', b'M');

/// One single-molecule evidence tuple: read identity, log2 poly(A) length,
/// and the slice of `candidate_pool` containing this read's candidate
/// transcript IDs.
#[derive(Debug, Clone)]
pub struct MoleculeEvidence {
    pub read_id: String,
    pub log_pal: f32,
    pub candidate_offset: u32,
    pub candidate_count: u16,
}

/// Compact memory graph containing all reads and a flattened pool of
/// transcript candidate IDs (one slice per molecule), plus the per-candidate
/// alignment evidence in lockstep parallel pools. Designed for zero-copy
/// downstream consumption by Module 3.
///
/// Layout invariant (all four pools): `candidate_pool`, `candidate_as`,
/// `candidate_nm` and `candidate_mapq` are molecule-major / candidate-minor
/// concatenations of identical length. A molecule owns the range
/// `[candidate_offset, candidate_offset + candidate_count)` in *each* of them;
/// index `candidate_pool[o + c]` and `candidate_as[o + c]` describe the same
/// (read, transcript) pair.
#[derive(Debug)]
pub struct FlatEvidenceGraph {
    pub molecules: Vec<MoleculeEvidence>,
    /// Candidate transcript IDs (index into `transcript_names`).
    pub candidate_pool: Vec<u32>,
    /// minimap2 alignment score `AS:i` per candidate. A record without `AS:i`
    /// contributes 0, which only leaves A_rk uniform when EVERY candidate of
    /// that read is tag-less (see the fallback semantics in `extract_evidence`).
    pub candidate_as: Vec<i32>,
    /// Edit distance `NM:i` per candidate (audit only; not used by TEL-EM).
    pub candidate_nm: Vec<i32>,
    /// Mapping quality per candidate (audit only; not used by TEL-EM).
    pub candidate_mapq: Vec<u8>,
    pub transcript_names: Vec<String>,
}

/// Parse the transcriptome BAM and emit the candidate graph.
///
/// `tes_slack` (in bp) accepts reads whose 3' alignment end falls within
/// `tes_slack` of the transcript's 3' end (default 35; spec F2 / CLI).
pub fn extract_evidence<P: AsRef<Path>>(
    bam_path: P,
    tes_slack: usize,
) -> Result<FlatEvidenceGraph> {
    let file = File::open(&bam_path)
        .with_context(|| format!("Failed to open BAM: {:?}", bam_path.as_ref()))?;
    let mut reader = bam::io::Reader::new(BufReader::new(file));
    let header = reader.read_header().context("Failed to read BAM header")?;

    // Build transcript name index + target length table.
    // `reference_sequences()` returns an ordered map keyed by ReferenceSequenceId;
    // iteration order matches `@SQ` lines in the header (deterministic).
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

    // Temporary per-read accumulator.
    //
    // MEMORY (COPD29, 16.4 GB BAM): adding A_rk moved peak RSS 2.99 GB → ~4.1 GB.
    // That growth is transient co-existence, not a leak — the peak is inside
    // `extract_evidence`, while this `read_map` is live and growing the flat
    // pools below it are growing too. `CandidateAlign` is `{u32, i32, i32, u8}` =
    // 16 B with padding, versus the bare `u32` (4 B) this vector held before
    // F-EM-4; at ~4.9 candidates/read and a doubling capacity of 8 that is
    // ~3.9 M × 96 B ≈ 375 MB, plus malloc headers and alignment on ~3.9 M
    // entries (~125 MB), plus up to 171 MB of doubling slack on the three new
    // pools, plus arena fragmentation across 11.7 M allocations of a new size
    // class — ~0.7–1.1 GB, consistent with the observed increase.
    //
    // `candidate_nm` / `candidate_mapq` are retained deliberately: the expert
    // reviewers required MAPQ/NM for audit. A struct-of-slots refactor would
    // remove most of the per-entry padding and is out of scope here.
    struct TempReadEntry {
        poly_a_len: i32,
        candidates: Vec<CandidateAlign>,
    }

    // One (read, transcript) candidate plus the alignment evidence of the
    // record that won the max-AS merge.
    struct CandidateAlign {
        tx_id: u32,
        as_score: i32,
        nm: i32,
        mapq: u8,
    }

    let mut read_map: HashMap<String, TempReadEntry> = HashMap::new();

    let mut record = bam::Record::default();
    while reader.read_record(&mut record)? != 0 {
        // Skip unmapped reads — they carry no alignment evidence.
        let flags = record.flags();
        if flags.is_unmapped() {
            continue;
        }

        // 1. Extract `pt:i` tag (Dorado native polyA length). Cast i64 → i32
        //    with bounds check (polyA lengths > 2^31 are biologically impossible).
        let data = record.data();
        let poly_a_len: i32 = match data.get(&TAG_PT) {
            Some(Ok(value)) => match value.as_int() {
                Some(len) if len > 0 && len <= i32::MAX as i64 => len as i32,
                _ => continue, // skip non-positive or >i32::MAX
            },
            _ => continue, // skip missing or unparseable tag
        };

        // 2. Reference sequence index + alignment 3' end position.
        // noodles-bam methods return `Option<Result<T, io::Error>>`.
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
            _ => continue,
        };
        if span == 0 {
            continue; // empty alignment
        }
        let ref_end = start_pos + span - 1;

        // 3' TES boundary filter: read must terminate within `tes_slack`
        // of the transcript's 3' end (1-based coordinate system).
        if ref_end + tes_slack < ref_len {
            continue;
        }

        // 3. Read name (QNAME).
        let read_id = match record.name() {
            Some(name) => String::from_utf8_lossy(name).to_string(),
            None => continue,
        };

        // 4. Alignment evidence. `AS:i` is the primary signal consumed by
        //    Module 3's A_rk term; `NM:i` and MAPQ are retained for audit.
        //    A missing / non-integer tag falls back to 0. That is neutral ONLY
        //    when EVERY candidate of the read is tag-less: `max_j AS_rj` is then
        //    also 0, `ln A_rk ≡ 0`, and A_rk is uniform and independent of λ.
        //    In the mixed case the tag-less candidate is scored AS = 0, so
        //    `ln A_rk = (0 − max_j AS_rj)/λ`, and the size of that penalty is
        //    what decides whether the candidate survives at all.
        //
        //    Measured per-read max AS on the COPD29 BAM
        //    (minimap2 2.28-r1209, `-ax map-ont`), 600k records:
        //        genome-wide   p10 462   p50 1654   p90 4338   max 27978
        //        PGAM1 pair                       p50 3206   p90 3338
        //    The output stage truncates `w < 0.01`, i.e. ≈ ln 99 = 4.595 nats
        //    of log-odds against a single competitor. So the penalty crosses
        //    that threshold once `max AS > 4.595·λ`:
        //        λ = 1000  ->  max AS > 4595  (~top 10–12% of reads)
        //        λ = 100   ->  max AS >  460  (most reads)
        //    At the PGAM1 median (3206) λ = 1000 gives −3.2 nats, which is
        //    short of 4.595 and so does not delete the candidate on its own.
        //    Measured rate of records lacking `AS:i`: 0 of 600,000, so this is
        //    a documentation caveat, not a live defect.
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

        // Accumulate candidate transcripts per read. O(n) dedup per insert
        // is acceptable for typical ONT multimappers (≤10 candidates).
        //
        // A repeated (read, transcript) pair — primary + supplementary, or two
        // secondaries landing on the same transcript — must MERGE rather than
        // be dropped: the record with the highest AS wins and its NM / MAPQ
        // ride along. Losing the AS here would silently mis-assign A_rk to
        // the wrong transcript.
        let tx_id = ref_id as u32;
        let align = CandidateAlign {
            tx_id,
            as_score,
            nm,
            mapq,
        };
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

    // Flatten into `FlatEvidenceGraph`.
    //
    // MEMORY: reserve the four pools at their exact final length instead of
    // growing them by `push`. The final length is not a guess — it is the total
    // number of distinct (read, transcript) pairs accumulated above, and that
    // total is knowable before a single element is pushed. Growing instead
    // leaves 2^25 = 33,554,432 slots for the 18,601,634 candidates measured on
    // the COPD29 BAM (3,906,840 reads, 4.76 candidates/read): 14.9 M unused
    // slots × 13 B/pool element = 194 MB of slack, carried live for the whole
    // of Module 3 on top of the 242 MB actually used.
    //
    // Measured end-to-end (`telescope quant` on that BAM, `/usr/bin/time -v`,
    // 2 runs each, non-overlapping): peak RSS 3,844,148 and 3,876,884 KB with
    // `push` growth → 3,712,644 and 3,675,896 KB reserved, i.e. −170 MB (−4.3%).
    // `extract_evidence` on its own moves much less (2,273,924 → 2,254,228 KB)
    // because `read_map` alone already reaches ~2,050 MB before the first pool
    // element exists, and this loop drains `read_map` as it goes — so the pools
    // never coexist with a full `read_map`. The win is the slack, which Module 3
    // pays for the whole run. Shortening `read_map` itself is the remaining
    // term and needs the map's per-read `Vec<CandidateAlign>` removed, which
    // would also change the molecule emission order (HashMap iteration order →
    // BAM order). Deliberately not done here.
    //
    // Guard on the estimate: the `is_empty` filter mirrors the `continue`
    // below, so the reserved capacity is an UPPER bound on what the loop
    // pushes, never an over-allocation; a molecule whose candidates were all
    // merged away contributes 0 to both sides. It is exact on this input (an
    // entry is only ever created immediately before a `push`), and a
    // hypothetical under-count would degrade to today's `push` growth rather
    // than to a wrong graph.
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

    for (read_id, entry) in read_map {
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
        }

        molecules.push(MoleculeEvidence {
            read_id,
            log_pal: (entry.poly_a_len as f32).log2(),
            candidate_offset,
            candidate_count,
        });
    }

    // Lockstep invariant: the three alignment-evidence pools are index-aligned
    // with `candidate_pool` (see `FlatEvidenceGraph`).
    debug_assert_eq!(candidate_pool.len(), candidate_as.len());
    debug_assert_eq!(candidate_pool.len(), candidate_nm.len());
    debug_assert_eq!(candidate_pool.len(), candidate_mapq.len());

    Ok(FlatEvidenceGraph {
        molecules,
        candidate_pool,
        candidate_as,
        candidate_nm,
        candidate_mapq,
        transcript_names,
    })
}

impl FlatEvidenceGraph {
    /// Distribution of `candidate_count` across molecules: returns a sorted
    /// vec of `(candidate_count, n_molecules_with_that_count)` pairs.
    /// Useful for diagnosing multi-mapping load (Phase A telemetry).
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZero;

    use noodles_core::Position;
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

    /// Build a synthetic BAM record: 100 bp match against `ref_id`, carrying
    /// `pt:i`, `NM:i`, MAPQ and (optionally) `AS:i`.
    fn synth_record(
        name: &str,
        ref_id: usize,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        let cigar: Cigar = [Op::new(Kind::Match, 100)].into_iter().collect();
        let mut tags = vec![(TAG_PT, Value::from(100i32)), (TAG_NM, Value::from(nm))];
        if let Some(a) = as_score {
            tags.push((TAG_AS, Value::from(a)));
        }
        RecordBuf::builder()
            .set_name(name)
            .set_flags(Flags::empty())
            .set_reference_sequence_id(ref_id)
            .set_alignment_start(Position::MIN)
            .set_mapping_quality(MappingQuality::new(mapq).unwrap())
            .set_cigar(cigar)
            .set_data(tags.into_iter().collect())
            .build()
    }

    /// Writes `records` under a 3-transcript header (tx0/tx1/tx2, 100 bp each)
    /// to a BAM in the system temp dir named after `tag` and the process id.
    /// Returns its path; the caller removes it. `tag` must differ per caller —
    /// the file name is the only thing keeping concurrent tests apart.
    ///
    /// Molecule ordering out of `extract_evidence` follows HashMap iteration
    /// order, so callers must look molecules up by `read_id`.
    fn write_synthetic_bam(tag: &str, records: Vec<RecordBuf>) -> std::path::PathBuf {
        let mut header = sam::Header::builder();
        for name in ["tx0", "tx1", "tx2"] {
            header = header.add_reference_sequence(
                name,
                Map::<ReferenceSequence>::new(const { NonZero::new(100).unwrap() }),
            );
        }
        let header = header.build();

        let path = std::env::temp_dir().join(format!(
            "telescope_evidence_{}_test_{}.bam",
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

    #[test]
    fn test_molecule_evidence_layout() {
        // Verifies the slice-offset invariant: `candidate_offset + candidate_count`
        // is exactly the start of the next molecule's slice in `candidate_pool`.
        let mol = MoleculeEvidence {
            read_id: "read_1".to_string(),
            log_pal: 6.0,
            candidate_offset: 0,
            candidate_count: 2,
        };
        assert_eq!(mol.candidate_offset + mol.candidate_count as u32, 2);
        assert!((mol.log_pal - 6.0_f32).abs() < f32::EPSILON);
    }

    /// End-to-end test of the alignment-evidence extraction path via the real
    /// `extract_evidence` on a synthetic BAM. Covers the three cases that can
    /// silently mis-assign AS to the wrong transcript:
    ///
    /// (a) one read → two transcripts with different AS;
    /// (b) one read → the SAME transcript twice with different AS — the
    ///     max-AS merge must keep the higher-scoring record (and its NM/MAPQ);
    /// (c) one read whose record lacks `AS:i` — must fall back to 0, which
    ///     makes A_rk uniform for that read.
    ///
    /// Also asserts the lockstep invariant between the parallel pools.
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
        let total: u64 = graph
            .molecules
            .iter()
            .map(|m| m.candidate_count as u64)
            .sum();
        assert_eq!(total as usize, graph.candidate_pool.len());

        // Molecule order follows HashMap iteration → index by read_id.
        let slice = |read_id: &str| -> (Vec<u32>, Vec<i32>, Vec<i32>, Vec<u8>) {
            let mol = graph
                .molecules
                .iter()
                .find(|m| m.read_id == read_id)
                .unwrap_or_else(|| panic!("{read_id} missing from graph"));
            let o = mol.candidate_offset as usize;
            let n = mol.candidate_count as usize;
            (
                graph.candidate_pool[o..o + n].to_vec(),
                graph.candidate_as[o..o + n].to_vec(),
                graph.candidate_nm[o..o + n].to_vec(),
                graph.candidate_mapq[o..o + n].to_vec(),
            )
        };

        // (a) Two distinct transcripts with two distinct AS, index-aligned with
        // their transcript ids (candidate insertion order is BAM order, but sort
        // by tx id so the assertion does not depend on it).
        let (txs, as_, nm, mapq) = slice("r1");
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

        // (b) Max-AS merge: only ONE candidate slot survives, carrying the
        // winning record's evidence (AS 800, not the first-seen 100).
        let (txs, as_, nm, mapq) = slice("r2");
        assert_eq!(txs, vec![1]);
        assert_eq!(as_, vec![800]);
        assert_eq!(nm, vec![1]);
        assert_eq!(mapq, vec![60]);

        // (c) Missing AS:i → 0 (uniform A_rk for this read).
        let (txs, as_, nm, mapq) = slice("r3");
        assert_eq!(txs, vec![2]);
        assert_eq!(as_, vec![0]);
        assert_eq!(nm, vec![4]);
        assert_eq!(mapq, vec![10]);
    }

    /// Phase A verifier-required test: candidate_count_histogram must be
    /// exercised from inside this file (>=2 occurrences of the symbol).
    /// Builds a synthetic FlatEvidenceGraph with a known candidate-count
    /// distribution and asserts the histogram returns the right counts in
    /// sorted order.
    #[test]
    fn test_candidate_count_histogram() {
        // Build a FlatEvidenceGraph by hand: 4 molecules with candidate_counts
        // {1, 2, 2, 3}. candidate_pool must be a contiguous run where the
        // slice [mol.candidate_offset .. mol.candidate_offset + mol.candidate_count)
        // holds that molecule's candidate ids (any ids are fine — histogram
        // doesn't read pool contents, only `candidate_count`).
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
            transcript_names: vec![],
        };

        let hist = graph.candidate_count_histogram();
        // Sorted ascending by candidate_count; types are (u16, u64).
        assert_eq!(
            hist,
            vec![(1_u16, 1_u64), (2_u16, 2_u64), (3_u16, 1_u64)],
            "candidate_count_histogram must bucket molecules correctly and sort by candidate_count"
        );
    }

    /// Guards the memory contract of the flatten step: the four pools are
    /// reserved at the summed candidate count, so no pool carries a single
    /// spare slot. Reverting them to `Vec::new()` + `push` (doubling growth)
    /// over-allocates any non-power-of-two total and turns this red — which is
    /// the point, because that doubling is what the COPD29 peak-RSS
    /// regression came from. The total here is deliberately 6, so a
    /// doubling-grown pool lands on capacity 8.
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
            transcript_names: vec![],
        };
        assert!(graph.candidate_count_histogram().is_empty());
    }
}
