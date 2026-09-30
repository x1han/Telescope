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
/// Layout invariant (all five pools): `candidate_pool`, `candidate_as`,
/// `candidate_nm`, `candidate_mapq` and `candidate_5p_gap` are molecule-major /
/// candidate-minor concatenations of identical length. A molecule owns the
/// range `[candidate_offset, candidate_offset + candidate_count)` in *each* of
/// them; index `candidate_pool[o + c]` and `candidate_as[o + c]` describe the
/// same (read, transcript) pair. A MISSING `extend` in the cohort path
/// (`resolver::pool_evidence_graphs`) desynchronises a pool from
/// `candidate_pool` and reads it from the wrong BAM's coordinate frame; the
/// only guard is a `debug_assert_eq!`, which is compiled out in release.
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
    /// Observed 5' gap `g_k` per candidate, in the candidate transcript's own
    /// cDNA coordinates: the number of transcript bases 5' of the read's 5' end
    /// in its best alignment to that candidate. Strand-corrected at extraction
    /// (see `extract_evidence`) so consumers never see a strand flag.
    ///
    /// Consumed by Module 3's 5'-anchoring term `ln_5p_k = β if g_k ≤ window`.
    /// It is FIXED per (read, candidate) — exactly like `AS` — so it is computed
    /// once here rather than per EM iteration; in the E-step it reduces to one
    /// `u32` compare and an add of a constant. It is a prior, not an EM
    /// parameter, and must never be folded into an accumulator.
    pub candidate_5p_gap: Vec<u32>,
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
    // pools below it are growing too. `CandidateAlign` is `{u32, i32, i32, u8,
    // u32}` = 20 B with padding, versus the bare `u32` (4 B) this vector held
    // before F-EM-4; at ~4.9 candidates/read and a doubling capacity of 8 that
    // is ~3.9 M entries, plus malloc headers and alignment on ~3.9 M entries,
    // plus doubling slack on the four evidence pools below, plus arena
    // fragmentation across 11.7 M allocations — ~0.7–1.1 GB, consistent with
    // the observed increase.
    //
    // The RSS figures quoted in this block were MEASURED on the four-field
    // (16 B) `CandidateAlign` / four-pool layout. The 5th pool and the 5th
    // struct field post-date them: they add 4 B per candidate in `read_map`
    // (~16 MB at ~3.9 M entries) and 4 B per candidate in each pool (~74 MB at
    // 18,601,634 candidates). Those are ARITHMETIC PROJECTIONS, not
    // measurements — re-derive the totals from a fresh `/usr/bin/time -v` run
    // before quoting them as measured. The trailing `u32` is placed last so it
    // does not disturb the field order the measured figures assumed; under
    // `repr(Rust)` reordering to save padding is not guaranteed anyway.
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
    //
    // `gap5p` rides along with the AS winner for free: the merge below is a
    // whole-struct `*existing = align`, so a repeat alignment to the same
    // transcript replaces this slot wholesale. That is what makes it the
    // *best* alignment's 5' gap rather than the first-seen record's — "the
    // read's best alignment TO CANDIDATE k" is the max-AS record, not the
    // 5'-closest one. On an exact AS tie the merge uses strict `>`, so the
    // first-seen record in BAM order wins; that is deterministic but arbitrary,
    // and it is the same tie-break that already decides `as_score`/`nm`/`mapq`.
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

        // 3. Observed 5' gap `g_k`, in this candidate's own cDNA coordinates.
        //
        //    The reference of a transcriptome BAM IS the transcript cDNA
        //    (`transcript_names` comes from the `@SQ` header), oriented 5'→3',
        //    so `start_pos` / `ref_end` are already cDNA coordinates. No
        //    genomic conversion is involved, and none is wanted.
        //
        //    STRAND (this is the one place the frozen spec is silent, so it is
        //    decided here explicitly rather than inherited):
        //
        //      forward  — the read's 5' end is the alignment's LEFT end, so the
        //                 observed gap is `s_k − 1 = start_pos − 1`.
        //      reverse  — the read is reverse-complemented against the cDNA, so
        //                 its 5' end is the alignment's RIGHT end and the
        //                 observed gap is `T_k − ref_end` (the transcript bases
        //                 lying beyond the alignment's high end).
        //
        //    `g_k = start_pos − 1` applied to a reverse record measures the gap
        //    at the molecule's 3' END and is silently wrong for every antisense
        //    alignment — roughly a coin flip in an ONT cDNA library, which is
        //    why `is_reverse_complemented()` was never read anywhere in this
        //    crate before now. Storing the ALREADY-CORRECTED gap (rather than
        //    the start plus a strand flag) means every downstream consumer,
        //    including Module 3's E-step, cannot get this wrong by omission.
        //
        //    noodles' `alignment_start()` is 1-BASED, so the forward case is
        //    already exactly the spec's `s_k − 1`; do NOT subtract a second 1
        //    (the spec's "1-based reference_start" is bait for rust-htslib,
        //    whose `reference_start()` is 0-based — this crate does not use it).
        //    The TES filter above guarantees `ref_end ≤ ref_len` is not
        //    required for correctness of the forward branch, and the reverse
        //    branch saturates at 0 rather than wrapping if a malformed BAM
        //    reports `ref_end > T_k`.
        let gap5p: u32 = if flags.is_reverse_complemented() {
            u32::try_from(ref_len.saturating_sub(ref_end)).unwrap_or(u32::MAX)
        } else {
            u32::try_from(start_pos.saturating_sub(1)).unwrap_or(u32::MAX)
        };

        // 4. Read name (QNAME).
        let read_id = match record.name() {
            Some(name) => String::from_utf8_lossy(name).to_string(),
            None => continue,
        };

        // 5. Alignment evidence. `AS:i` is the primary signal consumed by
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
        //    At the SHIPPED default λ = 20 (was 1000 — see the provenance note
        //    on `ALN_WEIGHT_LAMBDA_DEFAULT` in the CLI), the whole alignment
        //    term spans only `[−ΔAS_prune/λ, 0] = [−2.5, 0]` nats, because
        //    Module 3 prunes every candidate with `ΔAS > 50` before the term
        //    can matter. Two consequences, and they replace the λ=1000
        //    threshold analysis this block used to carry:
        //
        //      (a) A_rk alone can NEVER delete a candidate. The output stage
        //          truncates `w < 0.01`, i.e. ≈ ln 99 = 4.595 nats of log-odds
        //          against a single competitor, and reaching that costs
        //          `ΔAS > 4.595·20 = 91.9` — already past the ΔAS > 50 prune,
        //          so such a candidate never reaches the output stage at all.
        //          A_rk's job at λ = 20 is therefore to ORDER candidates, not
        //          to delete them. Under the old λ = 1000 the same threshold
        //          was `max AS > 4595`, i.e. the top 10–12% of reads DID have
        //          tag-less candidates deleted outright by A_rk alone.
        //      (b) A tag-less candidate (AS → 0) now takes the maximum possible
        //          penalty, −2.5 nats, on essentially every read (max AS ≫ 92
        //          at the p10 of 462). It survives the cutoff unless a
        //          competitor is also weak, but it can never be deleted by the
        //          alignment term. Measured rate of records lacking `AS:i`:
        //          0 of 600,000, so this remains a documentation caveat rather
        //          than a live defect.
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
            gap5p,
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
    // MEMORY: reserve the five pools at their exact final length instead of
    // growing them by `push`. The final length is not a guess — it is the total
    // number of distinct (read, transcript) pairs accumulated above, and that
    // total is knowable before a single element is pushed. Growing instead
    // leaves 2^25 = 33,554,432 slots for the 18,601,634 candidates measured on
    // the COPD29 BAM (3,906,840 reads, 4.76 candidates/read): 14.9 M unused
    // slots × 17 B/pool element = 254 MB of slack, carried live for the whole
    // of Module 3 on top of the 316 MB actually used. (The element size was
    // 13 B across four pools — `u32 + i32 + i32 + u8` — before
    // `candidate_5p_gap` added a fifth `u32`.)
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
    // The RSS figures above were measured on the FOUR-pool layout. The fifth
    // pool adds ~74 MB at that candidate count; that is arithmetic, not a
    // measurement. Re-measure before quoting it as one.
    //
    // Guard on the estimate: the `is_empty` filter mirrors the `continue`
    // below, so the reserved capacity is an UPPER bound on what the loop
    // pushes, never an over-allocation; a molecule whose candidates were all
    // merged away contributes 0 to both sides. It is exact on this input (an
    // entry is only ever created immediately before a `push`), and a
    // hypothetical under-count would degrade to today's `push` growth rather
    // than to a wrong graph.
    //
    // NOTE: Module 3's `ΔAS > 50` prune deliberately does NOT run here. It
    // needs the read's completed candidate list, so it cannot be applied in
    // the per-record loop above; putting it here instead would leave every
    // pool's `capacity()` an over-estimate of its `len()`, reversing the exact
    // reservation measured above. It runs in the E-step as a `-inf` push,
    // which is also where it stays switchable per run.
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
            candidate_5p_gap.push(c.gap5p);
        }

        molecules.push(MoleculeEvidence {
            read_id,
            log_pal: (entry.poly_a_len as f32).log2(),
            candidate_offset,
            candidate_count,
        });
    }

    // Lockstep invariant: the four alignment-evidence pools are index-aligned
    // with `candidate_pool` (see `FlatEvidenceGraph`).
    debug_assert_eq!(candidate_pool.len(), candidate_as.len());
    debug_assert_eq!(candidate_pool.len(), candidate_nm.len());
    debug_assert_eq!(candidate_pool.len(), candidate_mapq.len());
    debug_assert_eq!(candidate_pool.len(), candidate_5p_gap.len());

    Ok(FlatEvidenceGraph {
        molecules,
        candidate_pool,
        candidate_as,
        candidate_nm,
        candidate_mapq,
        candidate_5p_gap,
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
/// Synthetic-BAM helpers shared with `resolver`'s tests.
///
/// These are `#[cfg(test)]`-only and exist because the cohort pooling path
/// (`resolver::pool_evidence_graphs`) needs a real BAM on disk, and duplicating
/// a second writer there would let the two copies drift — which is precisely
/// the class of bug the pooling path is prone to.
#[cfg(test)]
pub(crate) mod test_support {
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

    /// Length of every reference in `write_synthetic_bam`'s header. Was 100 bp
    /// while every fixture aligned from `Position::MIN` with a 100M CIGAR, which
    /// pinned `g_k ≡ 0` for every synthetic candidate and made a 5'-term test
    /// unable to distinguish a real gap from a constant. Lengthening it is what
    /// allows a partial-length read, and the read length is then DERIVED from
    /// the gap so the fixture stays inside the 3' TES filter.
    pub(crate) const REF_LEN: usize = 2000;

    /// Build a synthetic BAM record: a full-length forward match against
    /// `ref_id` starting at 1, carrying `pt:i`, `NM:i`, MAPQ and (optionally)
    /// `AS:i`. `g_k = 0` for every record this builds.
    pub(crate) fn synth_record(
        name: &str,
        ref_id: usize,
        as_score: Option<i32>,
        nm: i32,
        mapq: u8,
    ) -> RecordBuf {
        synth_record_with_gap(name, ref_id, 0, false, as_score, nm, mapq)
    }

    /// As `synth_record`, but with an explicit observed 5' GAP `g_k` (in bp)
    /// and strand, from which the coordinates are derived:
    ///
    ///   forward — the read starts `g_k + 1` bases into the transcript;
    ///   reverse — the read starts at 1 and ends `g_k` bases SHORT of the
    ///             transcript's 3' end, because on a reverse alignment the
    ///             read's 5' end is the alignment's HIGH end.
    ///
    /// Both cases have span `REF_LEN − g_k`, so the read's 3' end lands at
    /// `REF_LEN` (forward) or `REF_LEN − g_k` (reverse) and always satisfies
    /// the 3' TES filter used by the tests. Deriving the coordinates from the
    /// gap rather than fixing the start is what keeps this coherent: the TES
    /// filter REJECTS any read whose 3' end is more than `tes_slack` from the
    /// transcript's 3' end, so a fixture that pins the start and the CIGAR
    /// length independently silently empties the graph the moment the
    /// reference is lengthened.
    ///
    /// Note the consequence for a reverse record: it can only ever have
    /// `g_k <= tes_slack`. A reverse read with a large 5' gap is unreachable
    /// under this filter, and that is a property of the data, not of the
    /// fixture.
    ///
    /// `reverse` sets SAM `REVERSE_COMPLEMENTED` (0x10) — in noodles the
    /// accessor is `is_reverse_complemented()`, not `is_reverse()`.
    pub(crate) fn synth_record_with_gap(
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

    /// Writes `records` under a 3-transcript header (tx0/tx1/tx2, `REF_LEN` bp
    /// each) to a BAM in the system temp dir named after `tag` and the process
    /// id. Returns its path; the caller removes it. `tag` must differ per
    /// caller — the file name is the only thing keeping concurrent tests apart.
    ///
    /// Molecule ordering out of `extract_evidence` follows HashMap iteration
    /// order, so callers must look molecules up by `read_id`.
    pub(crate) fn write_synthetic_bam(tag: &str, records: Vec<RecordBuf>) -> std::path::PathBuf {
        let mut header = sam::Header::builder();
        for name in ["tx0", "tx1", "tx2"] {
            header = header.add_reference_sequence(
                name,
                Map::<ReferenceSequence>::new(const { NonZero::new(REF_LEN).unwrap() }),
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
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

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
        assert_eq!(graph.candidate_5p_gap.len(), graph.candidate_pool.len());
        let total: u64 = graph
            .molecules
            .iter()
            .map(|m| m.candidate_count as u64)
            .sum();
        assert_eq!(total as usize, graph.candidate_pool.len());

        // Molecule order follows HashMap iteration → index by read_id.
        // Every pool is returned by name; a `..` in a destructure below would
        // silently stop asserting the new one. Named as a struct rather than a
        // 5-tuple so the field order cannot be transposed without a compile
        // error (two of the five pools are `u32`).
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

        // (a) Two distinct transcripts with two distinct AS, index-aligned with
        // their transcript ids (candidate insertion order is BAM order, but sort
        // by tx id so the assertion does not depend on it). All records here
        // start at 1 forward, so every gap is 0.
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

        // (b) Max-AS merge: only ONE candidate slot survives, carrying the
        // winning record's evidence (AS 800, not the first-seen 100).
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

    /// Regression test for the strand-corrected 5' gap `g_k`.
    ///
    /// `g_k` is the number of candidate-cDNA bases lying 5' of the read's 5'
    /// end, so the two strands measure opposite ENDS of the alignment:
    /// forward reads → `s_k − 1`, reverse reads → `T_k − ref_end`. Applying the
    /// forward formula to a reverse record is a ~50% silent error in a library
    /// with a non-negligible antisense fraction, and it is invisible in a
    /// fixture where every record is full-length and forward — which, with a
    /// 100 bp reference, is every record the old helper could build.
    ///
    /// `rev` is the sharp case: it starts at position 1, so the forward formula
    /// would report `g_k = 0` — the strongest possible 5' claim — when the read
    /// is in fact 30 bp from the 3' end.
    ///
    /// Mutation target: dropping the `is_reverse_complemented()` branch (or
    /// computing `start_pos − 1` unconditionally) reports 0 for `rev`, and the
    /// `f20` case fails too if the off-by-one slips.
    #[test]
    fn test_5p_gap_is_strand_corrected() {
        let path = write_synthetic_bam(
            "gap5p",
            vec![
                // Forward, g_k = 5 → start 6. Pins the 1-based convention:
                // `s_k − 1 = 5`, NOT 6.
                synth_record_with_gap("f5", 0, 5, false, Some(1000), 1, 60),
                // Forward, g_k = 20 → start 21. Exactly the frozen `g_k <= 20`
                // window edge.
                synth_record_with_gap("f20", 1, 20, false, Some(900), 1, 60),
                // Reverse, g_k = 30 → start 1, ref_end = REF_LEN − 30.
                synth_record_with_gap("rev", 2, 30, true, Some(800), 1, 60),
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
        assert_eq!(
            gap_of("rev"),
            30,
            "reverse ⇒ g_k = T_k − ref_end = 30, NOT start − 1 = 0"
        );
    }

    /// The 5' gap must follow the record that WON the max-AS merge, because the
    /// frozen spec defines `s_k` as the start of the read's BEST alignment to
    /// candidate k. `CandidateAlign` rides the gap through the whole-struct
    /// `*existing = align` overwrite; a side table keyed by (read, transcript)
    /// would instead hold the first-seen record's start and pass this test
    /// wrongly.
    #[test]
    fn test_5p_gap_follows_the_max_as_winner() {
        let path = write_synthetic_bam(
            "gap5p_merge",
            vec![
                // Same (read, transcript) twice: the second record has the
                // higher AS and therefore owns the stored gap.
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
            candidate_5p_gap: vec![0; 8],
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

    /// Guards the memory contract of the flatten step: the five pools are
    /// reserved at the summed candidate count, so no pool carries a single
    /// spare slot. Reverting them to `Vec::new()` + `push` (doubling growth)
    /// over-allocates any non-power-of-two total and turns this red — which is
    /// the point, because that doubling is what the COPD29 peak-RSS
    /// regression came from. The total here is deliberately 6, so a
    /// doubling-grown pool lands on capacity 8.
    ///
    /// The fixture's AS gaps (100/100, 100/100, 100/100) are irrelevant here
    /// precisely because the `ΔAS > 50` prune runs in the E-step, not in
    /// `extract_evidence` — this test is about allocation, and `graph` is
    /// observed before any pruning could occur.
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
        };
        assert!(graph.candidate_count_histogram().is_empty());
    }
}
