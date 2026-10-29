//! Module: Transcriptome Evidence Extractor
//!
//! Streams a transcriptome BAM, extracts Dorado `pt:i` poly(A) tail length per
//! record plus per-candidate alignment evidence (`AS:i`, `NM:i`, MAPQ), filters
//! by 3' TES proximity, and constructs a flat indexed candidate graph for the
//! downstream EM solver.
//!
//! Input contract: one BAM file, transcriptome-aligned, carrying a `pt:i` tag
//! per record.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context, Result};
use noodles_bam as bam;
use noodles_sam::alignment::record::data::field::Tag;
use noodles_sam::alignment::Record as _;

/// Dorado poly(A) tail length SAM tag: `pt:i`. The default for
/// `PolyASource::BamTag`.
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

/// Record-level accounting for the filters in `extract_evidence_with`.
///
/// WHY A LEDGER AND NOT A COUNTER. This function drops records at SIX
/// distinct sites, and a count of only the largest one leaves a residual that
/// no reader can attribute: it is indistinguishable from a bug. So this is a
/// closed ledger, not a tally -- `ledger_deficit()` is asserted to be 0 inside
/// the library before the graph is returned, which makes a future new drop
/// site FAIL LOUDLY at the point it is added instead of quietly skewing one
/// number forever.
///
/// SCOPE, stated precisely because it is easy to over-claim. These are RECORD
/// counts, not read counts and not candidate counts. One read contributes 1
/// record per alignment it has, so `records_entered` exceeds the molecule
/// count. The record-level `records_with_candidates` is NOT the molecule count
/// either: the push is a MERGE, so a second record for the same (read,
/// transcript) contributes 1 here and 0 new molecules. The candidate
/// conservation law lives in the EM solver's own candidate ledger, which is a
/// different and finer-grained accounting at a different stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FilterCounts {
    /// Every record read from the BAM, before any filter. The LHS of the
    /// identity below.
    pub records_entered: u64,
    /// `FLAG & 4` -- carries no alignment evidence.
    pub dropped_unmapped: u64,
    /// `FLAG & 2048` -- dropped, not merged (see the SAM-semantics note at the
    /// filter; merging a split read's segments is unsound from a transcriptome
    /// BAM alone).
    pub dropped_supplementary: u64,
    /// The requested poly(A) tag is absent or unparseable on this record.
    pub dropped_no_poly_a_tag: u64,
    /// The poly(A) tag parsed to a non-positive value or to one above
    /// `i32::MAX`. Distinct from the above because the extraction SUCCEEDED
    /// and the value was rejected.
    pub dropped_invalid_poly_a: u64,
    /// `alignment_span()` returned `None` or `Err` -- the CIGAR could not be
    /// resolved to a reference span.
    pub dropped_no_span: u64,
    /// The span resolved to 0.
    pub dropped_empty_span: u64,
    /// 3' TES boundary: the alignment ends more than `tes_slack` before the
    /// transcript's 3' end.
    pub dropped_tes_window: u64,
    /// Records that survived every filter and reached candidate construction.
    ///
    /// This is the identity's RHS, and it is a RECORD count: a record whose
    /// (read, transcript) pair already exists is merged away here and adds no
    /// molecule. See the scope note above.
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

    /// `records_entered - (dropped_total + records_with_candidates)`.
    ///
    /// MUST be 0. A non-zero value means a record left the stream through a
    /// path that is not in the ledger -- an early `return`, a `?`, or a drop
    /// site added without its counter. This is the assertion that gives the
    /// other seven fields their meaning.
    pub fn ledger_deficit(&self) -> i64 {
        self.records_entered as i64 - (self.dropped_total() + self.records_with_candidates) as i64
    }
}

/// Compact memory graph containing all reads and a flattened pool of
/// transcript candidate IDs (one slice per molecule), plus the per-candidate
/// alignment evidence in lockstep parallel pools.
///
/// Layout invariant (all five pools): `candidate_pool`, `candidate_as`,
/// `candidate_nm`, `candidate_mapq` and `candidate_5p_gap` are molecule-major /
/// candidate-minor concatenations of identical length. A molecule owns the
/// range `[candidate_offset, candidate_offset + candidate_count)` in *each* of
/// them; index `candidate_pool[o + c]` and `candidate_as[o + c]` describe the
/// same (read, transcript) pair. A MISSING `extend` in the cohort path
/// (`resolver::pool_evidence_graphs`) desynchronises a pool from
/// `candidate_pool` and reads it from the wrong BAM's coordinate frame. Both
/// sites are guarded by an UNCONDITIONAL `assert_eq!`, so a desynchronised pool
/// aborts instead of producing a wrong 5' gap in a release build.
#[derive(Debug)]
pub struct FlatEvidenceGraph {
    pub molecules: Vec<MoleculeEvidence>,
    /// Candidate transcript IDs (index into `transcript_names`).
    pub candidate_pool: Vec<u32>,
    /// minimap2 alignment score `AS:i` per candidate. A record without `AS:i`
    /// contributes 0, which only leaves A_rk uniform when EVERY candidate of
    /// that read is tag-less (see the fallback semantics in `extract_evidence`).
    pub candidate_as: Vec<i32>,
    /// Edit distance `NM:i` per candidate (audit only; not used by the EM
    /// solver).
    pub candidate_nm: Vec<i32>,
    /// Mapping quality per candidate (audit only; not used by the EM solver).
    pub candidate_mapq: Vec<u8>,
    /// Observed 5' gap `g_k` per candidate, in the candidate transcript's own
    /// cDNA coordinates: the number of transcript bases 5' of the read's 5' end
    /// in its best alignment to that candidate.
    ///
    /// NO strand correction is performed, and none is needed: transcriptome
    /// alignments are forward by construction, and `extract_evidence` REFUSES
    /// any reverse-strand record rather than reading it as forward.
    ///
    /// Consumed by the EM solver's 5'-anchoring term `ln_5p_k = β if g_k ≤
    /// window`. It is FIXED per (read, candidate) — exactly like `AS` — so it
    /// is computed once here rather than per EM iteration; in the E-step it
    /// reduces to one `u32` compare and an add of a constant. It is a prior,
    /// not an EM parameter, and must never be folded into an accumulator.
    pub candidate_5p_gap: Vec<u32>,
    pub transcript_names: Vec<String>,
    /// PROVENANCE, not measurement: true iff this graph was built with
    /// `PolyASource::None`, so every molecule's `log_pal` is the placeholder
    /// rather than an observed tail length.
    ///
    /// This is a property of the GRAPH, not of a molecule, which is why it is
    /// one bool here and not `Option<f32>` on `MoleculeEvidence`: it is O(1) on
    /// the struct and touches no Parquet schema, because `FlatEvidenceGraph`
    /// is never serialised.
    ///
    /// It exists so `em::solve_and_write_profile` can refuse the combination
    /// that is silently wrong: a placeholder graph at `poly_a_weight > 0`
    /// feeds a constant emission into the E-step. The CLI forces the weight to
    /// 0, but `solve_and_write_profile` is a public function and a library
    /// caller never touches the CLI.
    pub poly_a_is_placeholder: bool,
    /// Record-level filter ledger for the extraction that produced this graph.
    /// See [`FilterCounts`] for why this is a closed identity rather than a
    /// single tally, and for what these counts are NOT (they are not reads and
    /// not molecules).
    pub stats: FilterCounts,
}

/// Where per-read poly(A) length comes from.
///
/// `pt:i` is Dorado's native tail-length tag, so it is the default and it is
/// what every BAM in the 10-sample cohort carries. It is a basecaller-specific
/// tag, and a BAM re-aligned from a producer that does not emit it has no
/// `pt:i` at all; before this type existed that case produced an EMPTY GRAPH
/// and exit code 0: every record dropped at the tag read, with nothing to
/// distinguish "this BAM has no poly(A)" from "this sample has no reads". That
/// is the fail-quiet path `PolyASource::None` exists to answer.
///
/// A third variant, `Tsv` (`read_id<TAB>poly_a_len`), existed once and was
/// removed as a physical deletion, not deprecated: `--polyA-source tsv:<path>`
/// is now a parse error, not a warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolyASource {
    /// Read a two-character SAM tag from each record. `None` on a record is a
    /// drop, exactly as before — that is correct when most records carry the
    /// tag, and a global absence is a hard error rather than a silent success.
    BamTag(Tag),
    /// Declared absence of poly(A). No record is dropped for want of it and
    /// `log_pal` is filled with a finite placeholder.
    ///
    /// THE PLACEHOLDER IS NOT DATA. It is `log_pal = 0.0` -- `log2(1)`, the
    /// origin of the Gaussian's own space -- so pairing this with
    /// `poly_a_weight > 0` would feed a constant, confident-looking emission
    /// into the E-step and quietly distort every posterior. The CLI therefore
    /// FORCES `poly_a_weight = 0` when this variant is selected, so the
    /// inconsistent pairing cannot be expressed rather than merely warned
    /// about.
    ///
    /// It must be FINITE, and the obvious alternative is not. `log2(0)` is
    /// `-inf`, and `poly_a_weight = 0` does not protect the M-step from that:
    /// the Gaussian is multiplied in the E-step only, while `y` reaches the
    /// S1/S2 accumulators unconditionally. A single read at `y = -inf` sends
    /// `mu` to `-inf` and `sigma2` to NaN for its transcript. `0.0` is finite
    /// and survives that path; it still biases `mu` toward the origin, which is
    /// a known and accepted cost of an alignment-only run, not a corruption.
    None,
}

/// Parse the `--polyA-source` CLI value into a [`PolyASource`].
///
/// Two forms:
///   `bam_tag:<TAG>` Dorado's native tail-length tag (the default)
///   `none`          declared-absent; alignment-only, `poly_a_weight` forced 0
///
/// The tag is a parameter rather than the literal `pt` so a re-basecaller that
/// emits a different tag can be used without a code change.
pub fn parse_poly_a_source(spec: &str) -> Result<PolyASource> {
    let spec = spec.trim();
    if spec == "none" {
        return Ok(PolyASource::None);
    }
    if let Some(tag) = spec.strip_prefix("bam_tag:") {
        let bytes = tag.as_bytes();
        // SAM tags are exactly two alphanumeric characters. Checking here
        // rather than unwrapping keeps a typo from a panic in a long batch run.
        if bytes.len() != 2 || !bytes.iter().all(u8::is_ascii_alphanumeric) {
            anyhow::bail!("--polyA-source bam_tag: needs a 2-character SAM tag, got `{tag}`");
        }
        return Ok(PolyASource::BamTag(Tag::from([bytes[0], bytes[1]])));
    }
    anyhow::bail!("--polyA-source must be one of `bam_tag:<TAG>` or `none`; got `{spec}`");
}

/// Number of records scanned before deciding a poly(A) source is globally
/// absent. This is a sampling depth, not a threshold on the data.
const POLY_A_PROBE_RECORDS: usize = 10_000;

/// Parse the transcriptome BAM and emit the candidate graph, reading poly(A)
/// from Dorado's native `pt:i` tag.
///
/// Equivalent to `extract_evidence_with(.., &PolyASource::BamTag(TAG_PT))`, and
/// **fails** rather than returning an empty graph when no record in the first
/// [`POLY_A_PROBE_RECORDS`] carries the tag.
pub fn extract_evidence<P: AsRef<Path>>(
    bam_path: P,
    tes_slack: usize,
) -> Result<FlatEvidenceGraph> {
    extract_evidence_with(bam_path, tes_slack, &PolyASource::BamTag(TAG_PT))
}

/// As [`extract_evidence`], with an explicit poly(A) source.
pub fn extract_evidence_with<P: AsRef<Path>>(
    bam_path: P,
    tes_slack: usize,
    poly_a: &PolyASource,
) -> Result<FlatEvidenceGraph> {
    let file = File::open(&bam_path)
        .with_context(|| format!("Failed to open BAM: {:?}", bam_path.as_ref()))?;
    let mut reader = bam::io::Reader::new(BufReader::new(file));
    let header = reader.read_header().context("Failed to read BAM header")?;

    // Build transcript name index + target length table. `reference_sequences()`
    // iterates in header order, so the order is deterministic.
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

    // poly(A) comes either from a SAM tag on the record or is declared absent;
    // there is no external-table source.
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
    // Records seen and records carrying poly(A), for the global-absence guard.
    // Both guards below are keyed on `hits == 0`, and the `None` arm reports a
    // hit on every record, so neither can fire on an alignment-only run.
    let mut poly_a_records_seen: u64 = 0;
    let mut poly_a_hits: u64 = 0;

    // Record-level filter ledger. Asserted closed at the end of this function
    // (`fc.ledger_deficit() == 0`), so adding a drop site without its counter
    // fails the build's next test run instead of silently skewing a number.
    let mut fc = FilterCounts::default();

    // Temporary per-read accumulator.
    //
    // MEMORY: adding the per-candidate alignment evidence moved peak RSS on a
    // 16.4 GB BAM from 2.99 GB to ~4.1 GB. That growth is transient
    // co-existence, not a leak -- the peak is inside `extract_evidence`, while
    // this map is live and the flat pools below it are growing too.
    //
    // `candidate_nm` / `candidate_mapq` are retained deliberately: MAPQ and NM
    // are required for audit. A struct-of-slots refactor would remove most of
    // the per-entry padding and is out of scope here.
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
    // *best* alignment's 5' gap rather than the first-seen record's. On an
    // exact AS tie the merge uses strict `>`, so the first-seen record in BAM
    // order wins; that is deterministic but arbitrary, and it is the same
    // tie-break that already decides `as_score`/`nm`/`mapq`.
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
        // Skip unmapped reads — they carry no alignment evidence.
        let flags = record.flags();
        if flags.is_unmapped() {
            fc.dropped_unmapped += 1;
            continue;
        }
        // REVERSE-STRAND GUARD.
        //
        // Nothing downstream of this loop reads a strand flag, and no code in
        // this crate reads `FLAG & 0x10`. Transcriptome alignments are forward
        // by construction, but that is a property of the DATA, not of the
        // CODE, and it is exactly the kind of property that stops holding the
        // moment the tool is pointed at a genome BAM.
        //
        // Hence a hard failure rather than a silent pass. `gap5p = start_pos
        // - 1` on cDNA coordinates would be computed from an orientation the
        // rest of the pipeline does not model, and the resulting `g_k` would
        // feed the 5'-anchoring term as if it were meaningful. Refusing the
        // input is the only outcome that is correct without a strand model.
        // `is_reverse()` is `FLAG & 0x10`.
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
        // SUPPLEMENTARY (`FLAG & 2048`) records are DROPPED, not merged.
        //
        // SAM semantics: the CIGAR is always written in the molecule's own
        // 5'->3' order, so a LEADING hard clip means "this record starts `H`
        // bases into the read" -- i.e. the read's 5' portion is NOT aligned
        // here and this record is the molecule's **3'** segment. A TRAILING
        // hard clip means the opposite: the molecule's **5'** segment, whose 3'
        // remainder lives in another record. A leading `H` is therefore
        // evidence AGAINST a record being the 5' source, never for.
        //
        // Dropping is the right call and it is what NanoCount, Salmon and
        // IsoQuant do. Merging segments is unsound here for three reasons:
        // the max-AS merge can keep the 3' segment's interior start and report
        // it as a molecule 5' end (a small `g_k` then reads as "the read
        // arrived", i.e. as EVIDENCE FOR the candidate); chimeric long reads
        // are exactly the records most likely to be split; and
        // `reference_start` carries no query offset, so "which segment is 5'"
        // cannot be recovered reliably from a transcriptome BAM alone.
        //
        // The filter is a small live correction on every input path, not
        // defence in depth: supplementary records are ~0.1-0.3% of alignments.
        if flags.is_supplementary() {
            fc.dropped_supplementary += 1;
            continue;
        }

        // Counted for EVERY record that survives the flag filters, BEFORE the
        // poly(A) lookup. Counting only the records where a value was found
        // would make the global-absence guard unsatisfiable: `hits` and `seen`
        // would rise together and the "seen but no hits" state could never be
        // observed.
        poly_a_records_seen += 1;

        // GLOBAL-ABSENCE GUARD, in-loop form. Sampled over the first
        // POLY_A_PROBE_RECORDS records so a BAM with no poly(A) fails at record
        // 10,000 instead of at end-of-stream on a 27 M-record file.
        //
        // IT MUST SIT HERE, ABOVE THE LOOKUP. Below the lookup it is dead
        // code: every arm of the `match` below `continue`s on a record with no
        // poly(A), and `poly_a_hits` is incremented by the same statement that
        // survives those `continue`s -- so reaching the check implies
        // `hits >= 1`, making `hits == 0` unsatisfiable.
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

        // 1. Extract the poly(A) length for this record. Cast i64 → i32
        //    with bounds check (polyA lengths > 2^31 are biologically impossible).
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
                // Declared-absent. Nothing is dropped; the placeholder is NOT
                // a measurement and the CLI forces poly_a_weight = 0 to keep
                // it out of the E-step. 1, not 0: `0i32.log2()` is `-inf`, and
                // `-inf` is not confined to the E-step -- it reaches the S1/S2
                // accumulators, which the `poly_a_weight` gate does not cover.
                1
            }
        };
        if poly_a_len > 0 {
            poly_a_hits += 1;
        }
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

        // 3' TES boundary filter: read must terminate within `tes_slack`
        // of the transcript's 3' end (1-based coordinate system).
        if ref_end + tes_slack < ref_len {
            fc.dropped_tes_window += 1;
            continue;
        }

        // THE RHS TERM LIVES BELOW, NOT HERE. Read to the end of the `read_id`
        // match and you will find `fc.records_with_candidates += 1` sitting
        // immediately above `let entry = read_map.entry(read_id)`, i.e. after
        // the LAST exit in the record loop. That placement is the whole point.
        //
        // INVARIANT, and it is the one a future edit can break: there must be
        // no `continue`, `return`, `?`, `bail!` or early `panic!` between the
        // increment and the `candidates.push` below. If you add one, move the
        // increment past it in the same commit -- otherwise a record is
        // counted as delivered and then lost, with the ledger still balancing
        // and no counter recording the loss.
        //
        // WHY THIS IS NOT A SCHEMA CHANGE. `FilterCounts`' field list is
        // untouched -- no bucket was added, nothing downstream of Schema v1
        // sees a new field. What moved is the MEANING of one existing counter,
        // from "passed every filter" to "actually produced a candidate". The
        // second is the field's literal name, and it is the only placement
        // from which every exit is upstream.

        // 3. Observed 5' gap `g_k`, in this candidate's own cDNA coordinates.
        //
        //    The reference of a transcriptome BAM IS the transcript cDNA
        //    (`transcript_names` comes from the `@SQ` header), oriented 5'→3',
        //    so `start_pos` / `ref_end` are already cDNA coordinates. No
        //    genomic conversion is involved, and none is wanted.
        //
        //    STRAND — KNOWN GAP, NOT HANDLED. The frozen spec was validated on
        //    a FORWARD-ONLY population, and reverse records are REFUSED above
        //    rather than handled here. If a general path ever carries antisense
        //    alignments, this needs a separately validated definition -- it
        //    must not be reintroduced as an inline guess.
        //
        //    noodles' `alignment_start()` is 1-BASED, so this is already
        //    exactly the spec's `s_k − 1`; do NOT subtract a second 1 (the
        //    spec's "1-based reference_start" is bait for rust-htslib, whose
        //    `reference_start()` is 0-based -- this crate does not use it).
        let gap5p: u32 = u32::try_from(start_pos.saturating_sub(1)).unwrap_or(u32::MAX);

        // 4. Read name (QNAME).
        let read_id = match record.name() {
            Some(name) => String::from_utf8_lossy(name).to_string(),
            None => continue,
        };

        // 5. Alignment evidence. `AS:i` is the primary signal consumed by the
        //    EM solver's A_rk term; `NM:i` and MAPQ are retained for audit.
        //    A missing / non-integer tag falls back to 0. That is neutral ONLY
        //    when EVERY candidate of the read is tag-less: `max_j AS_rj` is then
        //    also 0, `ln A_rk ≡ 0`, and A_rk is uniform and independent of λ.
        //    In the mixed case the tag-less candidate is scored AS = 0, so
        //    `ln A_rk = (0 − max_j AS_rj)/λ`, and the size of that penalty is
        //    what decides whether the candidate survives at all.
        //
        //    A_rk alone can never delete a candidate at the shipped λ = 20: the
        //    output stage truncates `w < 0.01`, i.e. ≈ ln 99 = 4.595 nats of
        //    log-odds against a single competitor, and reaching that costs
        //    `ΔAS > 4.595·20 = 91.9` -- already past the ΔAS > 50 prune, so
        //    such a candidate never reaches the output stage at all. A_rk's job
        //    at λ = 20 is therefore to ORDER candidates, not to delete them.
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
        // A repeated (read, transcript) pair must MERGE rather than be
        // dropped: the record with the highest AS wins and its NM / MAPQ ride
        // along. Losing the AS here would silently mis-assign A_rk to the
        // wrong transcript.
        let tx_id = ref_id as u32;
        let align = CandidateAlign {
            tx_id,
            as_score,
            nm,
            mapq,
            gap5p,
        };

        // THE RHS TERM OF THE LEDGER IDENTITY. Everything above is computed;
        // everything below is bookkeeping. This is the LAST point in the loop
        // at which a record can still leave, so it is the only placement from
        // which every exit (refID, pos, read name) sits upstream and therefore
        // lands on the LHS alone.
        //
        // Counted here, the record has definitely produced a candidate: the
        // `match` immediately below either pushes a new entry or merges into an
        // existing one for the same read, and there is no third branch.
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

    // END-OF-STREAM version of the same guard. The in-loop check fires at
    // POLY_A_PROBE_RECORDS, which a BAM smaller than that never reaches -- and
    // a small BAM is exactly the case a fixture, a test and a smoke run all
    // use. Without this, the fail-quiet path survives precisely where it is
    // most likely to be exercised.
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

    // Flatten into `FlatEvidenceGraph`.
    //
    // MEMORY: reserve the five pools at their exact final length instead of
    // growing them by `push`. The final length is not a guess -- it is the
    // total number of distinct (read, transcript) pairs accumulated above, and
    // that total is knowable before a single element is pushed. Growing
    // instead leaves doubling slack carried live for the whole downstream run,
    // on top of the space actually used; measured end-to-end, reserving cut
    // peak RSS by about 4% against push growth.
    //
    // The win is the slack, which the downstream module pays for the whole
    // run. Shortening `read_map` itself is the remaining term and needs the
    // map's per-read `Vec<CandidateAlign>` removed, which would also change
    // the molecule emission order (HashMap iteration order -> BAM order).
    // Deliberately not done here.
    //
    // Guard on the estimate: the `is_empty` filter mirrors the `continue`
    // below, so the reserved capacity is an UPPER bound on what the loop
    // pushes, never an over-allocation.
    //
    // NOTE: the downstream `ΔAS > 50` prune deliberately does NOT run here. It
    // needs the read's completed candidate list, so it cannot be applied in the
    // per-record loop above; putting it here instead would leave every pool's
    // `capacity()` an over-estimate of its `len()`, reversing the exact
    // reservation. It runs in the E-step as a `-inf` push, which is also where
    // it stays switchable per run.
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

    // DETERMINISM: `read_map` is a `HashMap`, so iterating it yields entries in
    // an order that depends on a per-process random hash seed. That order
    // decides the order of `molecules`, which decides the order of the
    // `init_counts[tx_id] += share` accumulation in the E-step's alpha
    // initialisation, which is a FLOAT sum. Different addition order is a
    // different rounding, and it propagated into every posterior.
    //
    // Sorting by `read_id` removes the randomness. This changes the MODEL not
    // at all -- it fixes the order of an existing sum, so every value is one of
    // the values the unordered version could already produce.
    //
    // Do not "optimise" this away. This is the only thing making the emitted
    // parquet bit-reproducible, which is what makes a regression gate able to
    // compare md5s at all.
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
            // `poly_a_len` is >= 1 on every path that reaches here, so this is
            // finite. The guard is a cheap assertion that the three sources
            // cannot be changed into a `-inf` source unnoticed: an infinite `y`
            // propagates into the S1/S2 accumulators, which are outside the
            // `poly_a_weight` gate.
            log_pal: (entry.poly_a_len.max(1) as f32).log2(),
            candidate_offset,
            candidate_count,
        });
    }

    // Lockstep invariant: the alignment-evidence pools are index-aligned with
    // `candidate_pool` (see `FlatEvidenceGraph`).
    //
    // UNCONDITIONAL `assert_eq!`, not `debug_assert_eq!`: the release profile
    // compiles debug assertions out, and the failure mode should be a crash
    // rather than silently misaligned evidence.
    assert_eq!(candidate_pool.len(), candidate_as.len());
    assert_eq!(candidate_pool.len(), candidate_nm.len());
    assert_eq!(candidate_pool.len(), candidate_mapq.len());
    assert_eq!(candidate_pool.len(), candidate_5p_gap.len());

    // THE LEDGER CLOSES HERE, or the library is wrong. A new drop site added
    // above without its counter, or an early `?`/`return` added between the
    // filters and the push, makes this non-zero and fails the test suite --
    // which is the entire reason this is a sum and not a single tally.
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
    /// Distribution of `candidate_count` across molecules: returns a sorted
    /// vec of `(candidate_count, n_molecules_with_that_count)` pairs.
    /// Useful for diagnosing multi-mapping load.
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
/// Synthetic-BAM helpers shared with `resolver`'s tests, and with the
/// end-to-end integration tests in `tests/`, which are a SEPARATE crate and so
/// cannot see `#[cfg(test)]` items. That is what the `test-support` feature is
/// for; `#[doc(hidden)]` keeps it out of the published docs. The alternative —
/// a second BAM writer inside `tests/` — would let the copies drift.
#[doc(hidden)]
pub mod test_support {
    use super::*;

    use std::num::NonZero;

    use noodles_core::Position;

    /// RE-EXPORTED so a caller in another crate can build a flagged record
    /// without depending on `noodles_sam` itself; re-implementing the fixture
    /// there is exactly the copy that drifts.
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

    /// Length of every reference in `write_synthetic_bam`'s header. Was 100 bp
    /// while every fixture aligned from `Position::MIN` with a 100M CIGAR, which
    /// pinned `g_k ≡ 0` for every synthetic candidate and made a 5'-term test
    /// unable to distinguish a real gap from a constant.
    pub const REF_LEN: usize = 2000;

    /// Build a synthetic BAM record: a full-length forward match against
    /// `ref_id` starting at 1, carrying `pt:i`, `NM:i`, MAPQ and (optionally)
    /// `AS:i`. `g_k = 0` for every record this builds.
    pub fn synth_record(
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
    /// `REF_LEN` (forward) or `REF_LEN − g_k` (reverse). Deriving the
    /// coordinates from the gap rather than fixing the start is what keeps this
    /// coherent: the TES filter REJECTS any read whose 3' end is more than
    /// `tes_slack` from the transcript's 3' end, so a fixture that pinned the
    /// start and the CIGAR length independently would silently empty the graph
    /// the moment the reference is lengthened.
    ///
    /// Note the consequence for a reverse record: it can only ever have
    /// `g_k <= tes_slack`, because the TES filter runs first.
    ///
    /// `reverse` sets SAM `REVERSE_COMPLEMENTED` (0x10) — in noodles the
    /// accessor is `is_reverse_complemented()`, not `is_reverse()`.
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

    /// As [`synth_record`], but with NO reference sequence id at all: the
    /// builder's `set_reference_sequence_id` is simply never called, so the
    /// encoded record carries tid = -1.
    ///
    /// Note samtools forces FLAG 4 here and KEEPS MAPQ, which is what lets the
    /// record clear a MAPQ gate and die at `reference_sequence_id()` instead,
    /// one line further down.
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

    /// As [`synth_record`], but with NO read name: `set_name` is never called.
    ///
    /// The mirror image of [`synth_record_no_refid`] — `QNAME=*` is legal SAM
    /// and survives the round trip through samtools verbatim.
    ///
    /// Note this record keeps FLAG 0 and a valid tid, so it clears every filter
    /// up to `record.name()` -- which is the point. It dies at the LAST branch,
    /// the one that discards a record it could otherwise have written to the
    /// pool, so a missing name is a silent loss unless something counts it.
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

    /// As [`synth_record`], but with NO alignment start: `set_alignment_start`
    /// is never called, so the encoder writes POS = 0 and the decoder's
    /// `Position::new(0)` fails.
    ///
    /// The constraint is on the RUST CONSTRUCTOR -- `Position::new(0)` is
    /// invalid because positions are 1-based -- not on the encoding: an unset
    /// field is written as 0, and the decoder cannot turn 0 into a `Position`.
    /// The two facts face each other and the path is open.
    ///
    /// What is on the wire is `0`, NOT a `-1` sentinel -- which is how the tid
    /// case encodes absence, and the two are NOT the same mechanism. Anyone
    /// generalising from that case to this one will predict `pos=*` and be
    /// wrong.
    ///
    /// Requires a valid `ref_id`: `reference_sequence_id()` is checked BEFORE
    /// `alignment_start()`, so a record with no tid would land in
    /// `filtered_no_refid` and never reach this branch.
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

    /// As [`synth_record_with_gap`], but with an explicit leading CIGAR.
    ///
    /// `leading` are the ops that sit 5' of the first `Match`, given as
    /// `(kind, length)` pairs. The trailing `Match` length is **derived**, not
    /// supplied: it is whatever is needed for the alignment's reference span to
    /// end exactly at `REF_LEN`, so the read clears the 3' TES filter for any
    /// leading-op combination. Deriving it is the whole point -- a fixture that
    /// pinned the CIGAR length independently would silently fall out of the
    /// graph the moment the leading ops changed the reference span, and a
    /// missing molecule is indistinguishable from a rejected one.
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

    /// Module 2 `gap5p` under real CIGARs.
    ///
    /// The physics being pinned: `gap5p = s_k − 1` counts the reference bases
    /// between the transcript's 5' end and where the alignment begins. A
    /// leading op that consumes **read** bases (soft-clip `S`, insertion `I`,
    /// padding `P`) or **neither** (hard-clip `H`) does not move the
    /// reference start, so it must not move `gap5p`. A leading op that
    /// consumes **reference** (deletion `D`, skip `N`) sits *at* the start
    /// position rather than before it, so it also must not move `gap5p`.
    ///
    /// If any of these ever moved the answer, the 5' term would be reading a
    /// quantity that depends on the aligner's CIGAR rendering rather than on
    /// where the molecule physically starts -- i.e. two aligners, or one aligner
    /// across two versions, would silently produce different posteriors from
    /// the same molecules.
    #[test]
    fn test_gap5p_is_invariant_to_the_leading_cigar_ops() {
        use Kind::*;
        let gap = 50usize;
        // A different `ref_id` per case, because every read here is a
        // single-candidate read and molecules are keyed by read name, not by
        // transcript. All land on distinct transcripts so a mix-up between
        // them would show up as a wrong gap rather than a missing molecule.
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

    /// `gap5p` agrees with an independent cDNA-coordinate computation, for every
    /// CIGAR family, through the real `extract_evidence` path.
    ///
    /// The reference computation is deliberately a different route to the same
    /// number: it walks the alignment's own reference span from the 3' end
    /// backwards, whereas production reads the start position forwards. If
    /// noodles' `alignment_start()` and `alignment_span()` disagreed about
    /// where an alignment begins or how far it reaches, the two would part.
    #[test]
    fn test_gap5p_matches_an_independent_cdna_backward_computation() {
        use Kind::*;
        // (name, ref_id, gap, leading ops) — gaps span both sides of the frozen
        // window edge at 20, because a bug that shifted g_k by one would only
        // change an ASSIGNMENT on the wrong side of it, and this test is about
        // the coordinate, not the term.
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

            // Independent route: WALK THE CIGAR to count the reference bases
            // the alignment consumes, then measure the 5' end from the known
            // 3' terminus instead of from the start field. Production reads
            // `alignment_start() - 1` forwards; this reads the same geometry
            // backwards out of the CIGAR. If noodles' start were wrong, or if a
            // leading op shifted it, the two disagree.
            //
            // M / D / N / = / X consume reference; S / H / I / P do not.
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

    /// Supplementary alignments are DROPPED, and the CIGAR direction is stated
    /// the way the SAM spec states it.
    ///
    /// The CIGAR is always written in the **molecule's own 5' -> 3' order**,
    /// regardless of the strand the molecule aligned to. So:
    ///
    /// | CIGAR | meaning | which segment is it? |
    /// |---|---|---|
    /// | `900M1100H` | 1100 read bases 3' of this alignment are clipped | the molecule's **5'** segment |
    /// | `900H1000M` | 900 read bases 5' of this alignment are clipped | the molecule's **3'** segment |
    ///
    /// A **leading** hard clip is evidence AGAINST a record being the 5' source.
    ///
    /// # What is pinned
    ///
    /// * A read split across two records contributes only its primary. The
    ///   supplementary 3' segment is dropped, so `gap5p` comes from the primary
    ///   -- the molecule's actual 5' segment -- and not from an interior
    ///   junction coordinate.
    /// * A read whose ONLY record is supplementary contributes nothing.
    #[test]
    fn test_supplementary_alignments_are_dropped_rather_than_merged() {
        use Kind::*;
        // tx0 is 2000 nt and the junction sits at 1980, close to the TES.
        //
        // That placement is load-bearing, not cosmetic. The 3' TES filter keeps
        // only reads whose alignment end is within `tes_slack` of the
        // transcript end, so a 5' segment ending at an INTERIOR junction is
        // discarded by that filter whatever the supplementary flag says. On a
        // transcriptome reference the TES filter is therefore already a de-facto
        // supplementary filter, and the explicit check in `extract_evidence` is
        // the second, cheap line of defence.
        //
        // Molecule 5' segment: reference 1..=1980, no leading clip (1980M).
        // Its 3' remainder: reference 1981..=2000, LEADING hard clip -- what a
        // 3' segment looks like. The 3' segment deliberately scores HIGHER, so
        // the max-AS merge would otherwise win and report the junction
        // coordinate 1980 as the molecule's 5' end.
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
        // A read whose only record is supplementary must vanish entirely.
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
        // Control: a plain forward read that must survive untouched.
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
        // The structural fact the fixture placement relies on.
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

    /// A forward alignment that STOPS SHORT of the transcript's 3' end.
    ///
    /// What this helper adds over the general `gap` helpers, which derive the
    /// span from `REF_LEN` so the read always ends on the last base:
    ///
    /// 1. NO ACCOUNTING. A read dropped by the TES filter was
    ///    indistinguishable from a read that never existed.
    /// 2. NO BOUNDARY. `molecules.is_empty()` stays true whether the filter
    ///    rejects at `end_offset > tes_slack` or `>= tes_slack`, and whether it
    ///    compares `ref_end` or `ref_end + 1`. An off-by-one in the condition
    ///    is invisible to it. `synth_record_short_of_tes` puts a read exactly
    ///    at `tes_slack` and one at `tes_slack + 1` in the same run, which
    ///    pins both sides.
    ///
    /// `end_offset` is how many bases short of `REF_LEN` the alignment's last
    /// base sits, so `end_offset = 0` is the full-length case every other helper
    /// produces. The filter drops the record iff `end_offset > tes_slack`, which
    /// makes the boundary exactly testable rather than approximately.
    ///
    /// `start` is 1-based. The span is derived as `REF_LEN − end_offset −
    /// start + 1`, so the CIGAR and the coordinates can never disagree.
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

    /// A full-length forward match carrying `pt:i`, with SAM `flags` set by the
    /// caller instead of derived from a 5' gap.
    ///
    /// This exists to make the FLAG-only drop buckets (`dropped_unmapped`,
    /// `dropped_supplementary`) observable. Neither was reachable from any
    /// other fixture: every helper in this file emits a mapped, non-
    /// supplementary forward match, so both counters read 0 on the entire
    /// suite. That is not a harmless untested branch -- it means an assertion
    /// of the form `dropped_unmapped == 0` elsewhere is a tautology, which is
    /// the same trap this ledger was built to avoid.
    ///
    /// SCOPE: the record keeps a valid reference and a full-length CIGAR even
    /// when asked for `UNMAPPED`, which is not what a real unmapped record
    /// looks like. That is deliberate and sufficient: both filters read FLAGS
    /// and `continue` before the reference is touched, so this pins the flag
    /// branch and nothing else.
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

    /// Writes `records` under a 3-transcript header (tx0/tx1/tx2, `REF_LEN` bp
    /// each) to a BAM in the system temp dir named after `tag` and the process
    /// id. Returns its path; the caller removes it. `tag` must differ per
    /// caller — the file name is the only thing keeping concurrent tests apart.
    ///
    /// Molecule ordering out of `extract_evidence` follows `read_id` order, so
    /// callers must look molecules up by `read_id`.
    pub fn write_synthetic_bam(tag: &str, records: Vec<RecordBuf>) -> std::path::PathBuf {
        write_synthetic_bam_named(tag, &["tx0", "tx1", "tx2"], records)
    }

    /// `write_synthetic_bam` with the `@SQ` names chosen by the caller, so a
    /// caller can build a reference universe it cannot otherwise produce.
    ///
    /// The production `read_reference_universe` deliberately does NOT
    /// de-duplicate `@SQ` names ("silently collapsing one would turn a corrupt
    /// reference into a plausible-looking universe"), so a header naming the
    /// same transcript twice reaches the universe-padding loop twice. Under the
    /// fixed tx0/tx1/tx2 header that state is unreachable, which would have
    /// left the conservation `ensure!` with no way to be shown to fire at all.
    ///
    /// `names` is passed through verbatim, duplicates included. An empty name
    /// is rejected by the header builder, so a caller cannot accidentally
    /// construct the OTHER degenerate header case (a nameless reference).
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

    /// The in-loop probe, which the end-of-stream guard cannot stand in for.
    ///
    /// The fixture is the case only this guard can catch: 10,000 records with
    /// NO poly(A), then records that DO have it. The end-of-stream check passes
    /// on this file -- poly(A) exists, so the run is not a global absence --
    /// and the 10,000-record probe is what must stop it.
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
    /// `log_pal == 0.0` in a REAL graph is not evidence of a placeholder: a BAM
    /// legitimately contains reads with no usable `pt:i`, and those are dropped
    /// at extraction rather than filled in, so a real graph can and does reach
    /// 0.0. Deciding the flag from the values would reject ordinary runs.
    #[test]
    fn test_the_placeholder_flag_is_source_derived_not_inferred_from_log_pal() {
        use noodles_sam::alignment::record_buf::data::field::Value;

        // `src_pt` carries a real tail length; `src_none` has none, exactly
        // like a BAM where some reads lack a usable `pt:i`. Both must survive
        // the tag lookup for the measured arm, so only ONE is stripped.
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

    /// The graph's molecules come out in `read_id` order, and that order is the
    /// whole reason the emitted parquet is bit-reproducible.
    ///
    /// `read_map` is a `HashMap`, so an unsorted drain yields a per-process
    /// random order. That order propagates: molecules -> the `init_counts`
    /// float accumulation -> `alpha` -> every posterior.
    ///
    /// This asserts the INVARIANT rather than re-measuring the symptom, because
    /// the symptom is cross-process.
    ///
    /// WHY THIS EXTRACTS EIGHT TIMES, NOT ONCE. A 3-key `HashMap` iterates in
    /// sorted order by chance 1 time in 3! = 6, so a single extraction could
    /// pass by luck and certify an unpinned fix. Each `extract_evidence` call
    /// builds a FRESH `HashMap` with a fresh random seed, so repeating the
    /// extraction samples the real distribution.
    ///
    /// The read_ids are also given out of order, so a graph that merely
    /// preserved BAM order fails too — the property is sortedness, not
    /// "unchanged".
    #[test]
    fn test_molecules_are_emitted_in_read_id_order() {
        use noodles_sam::alignment::record_buf::data::field::Value;
        // Deliberately scrambled relative to both the sort order and the
        // numeric order of the names.
        let names = ["r7", "r2", "r5", "r1", "r8", "r3", "r6", "r4"];
        let recs: Vec<_> = names
            .iter()
            .map(|n| {
                // ref_id stays 0: it indexes the header's reference list, which
                // this fixture declares with a single sequence. Varying it makes
                // `write_alignment_record` reject the record outright.
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

    /// The two CLI spellings, and the rejections. The spelling is load-bearing
    /// beyond cosmetics: the guard's FATAL message quotes `--polyA-source` to
    /// the user, so a parser that only accepted some spelling would produce an
    /// error naming a flag that cannot be typed.
    ///
    /// `tsv:` is in the REJECTION list, not the acceptance list: the TSV
    /// poly(A) source was deleted outright (not deprecated), so both a
    /// well-formed TSV spec and a bare `tsv:` prefix must now be errors naming
    /// the flag.
    #[test]
    fn test_parse_poly_a_source_accepts_the_two_forms_and_rejects_the_rest() {
        assert_eq!(
            parse_poly_a_source("bam_tag:pt").unwrap(),
            PolyASource::BamTag(TAG_PT)
        );
        // The tag is a parameter, not a literal, so a re-basecaller's tag works
        // without a code change.
        assert_eq!(
            parse_poly_a_source("bam_tag:Zt").unwrap(),
            PolyASource::BamTag(Tag::from(*b"Zt"))
        );
        assert_eq!(parse_poly_a_source(" none ").unwrap(), PolyASource::None);

        // Every one of these must be a message naming the flag, not a panic and
        // not a silent fallback to the default -- a typo that quietly ran with
        // `bam_tag:pt` would reintroduce exactly the fail-quiet this exists to
        // remove.
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
        // log2 is the base the whole Gaussian lives in.
        let good = synth_record_with_gap("pt64", 0, 0, false, Some(1000), 1, 60);

        // pt:i present but zero -> dropped (a molecule cannot have no tail)
        let mut zero = synth_record_with_gap("pt0", 1, 0, false, Some(1000), 1, 60);
        zero.data_mut().clear();
        zero.data_mut().insert(TAG_PT, Value::from(0i32));

        // pt:i absent entirely -> dropped. This is the case a non-Dorado BAM
        // hits on EVERY record.
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

    /// A BAM with no `pt:i` on any record is a HARD ERROR, not an empty graph.
    ///
    /// The failure it removes is real. Every record is dropped at the tag read,
    /// so the caller cannot tell "this BAM has no poly(A)" from "this sample has
    /// no reads"; the error surfaces much later, as an empty profile or a
    /// divide-by-zero, with nothing pointing back at the tag. `pt:i` is Dorado's
    /// native tail-length tag, and not every producer is obliged to emit it.
    ///
    /// The 2-record fixture cannot reach the 10,000-record in-loop probe, so
    /// this also pins the END-OF-STREAM guard -- without it the fail-quiet path
    /// survives in exactly the case a unit test or a smoke run exercises.
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

    /// `none` drops nothing, and the global-absence guard must not fire on it.
    ///
    /// Note what actually makes that true: the `None` arm reports a HIT on
    /// every record, so `hits == 0` is unreachable and both guards pass by
    /// construction. The placeholder must be 1, not 0: `0i32.log2()` is `-inf`,
    /// which would make `hits == 0` reachable and kill a healthy
    /// alignment-only run at record 10,000.
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

        // Molecule order is by read_id, so index by read_id.
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

    /// Pins the frozen 1-based convention for the forward 5' gap `g_k = s_k − 1`.
    ///
    /// `f5` is the off-by-one tripwire: start 6 must give `g_k = 5`, not 6.
    /// `f20` sits exactly on the `g_k <= 20` window edge, so an inclusive /
    /// exclusive slip at the boundary also fails here.
    ///
    /// Strand is deliberately NOT exercised here — reverse records are refused
    /// outright, so there is no `g_k` to validate for them.
    #[test]
    fn test_5p_gap_is_forward_formula_and_one_based() {
        let path = write_synthetic_bam(
            "gap5p",
            vec![
                // Forward, g_k = 5 → start 6. Pins the 1-based convention:
                // `s_k − 1 = 5`, NOT 6.
                synth_record_with_gap("f5", 0, 5, false, Some(1000), 1, 60),
                // Forward, g_k = 20 → start 21. Exactly the frozen `g_k <= 20`
                // window edge.
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

    /// Strand is a REFUSED input, not a handled case.
    ///
    /// The frozen spec `g_k = s_k − 1` was validated on a forward-only
    /// population, and this crate reads no strand flag at all, so a reverse
    /// record's `g_k` would be computed from an orientation nothing downstream
    /// models. The input is therefore refused.
    ///
    /// The forward population is unaffected: the control at the end runs the
    /// same fixture without the reverse record and requires it to succeed.
    /// Without that, `expect_err` would also be satisfied by any unrelated
    /// failure to read the file.
    #[test]
    fn test_reverse_alignment_is_refused_rather_than_given_the_forward_formula() {
        let path = write_synthetic_bam(
            "strand_refused",
            vec![
                // reverse record: start 1 on the fixture's REF_LEN (2000) nt reference,
                // ending 30 bases SHORT of the 3' end. See REF_LEN.
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

        // CONTROL. The same fixture, forward only, must still be read.
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

    /// `candidate_count_histogram` must be exercised from inside this file.
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
            // synthetic/test graph: `log_pal` is a real measurement
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

    /// Guards the memory contract of the flatten step: the five pools are
    /// reserved at the summed candidate count, so no pool carries a single
    /// spare slot. Reverting them to `Vec::new()` + `push` (doubling growth)
    /// over-allocates any non-power-of-two total and turns this red — which is
    /// the point, because that doubling is what the peak-RSS regression came
    /// from. The total here is deliberately 6, so a doubling-grown pool lands
    /// on capacity 8.
    ///
    /// The fixture's AS gaps are irrelevant here precisely because the
    /// `ΔAS > 50` prune runs in the E-step, not in `extract_evidence` — this
    /// test is about allocation, and `graph` is observed before any pruning
    /// could occur.
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
            // synthetic/test graph: `log_pal` is a real measurement
            poly_a_is_placeholder: false,
            stats: crate::evidence::FilterCounts::default(),
        };
        assert!(graph.candidate_count_histogram().is_empty());
    }

    /// The 3' TES filter, at its exact boundary, and the ledger that accounts
    /// for it.
    ///
    /// The filter is `ref_end + tes_slack < ref_len` -> drop, so a read ending
    /// exactly `tes_slack` bases short of the transcript's 3' end is KEPT and
    /// one ending `tes_slack + 1` short is DROPPED. An off-by-one in either
    /// direction (strict vs non-strict, `ref_end` vs `ref_end + 1`) is
    /// invisible to a test that only checks "some read was dropped".
    ///
    /// The pre-existing coverage does not pin that boundary: it asserts
    /// `molecules.is_empty()`, which is true at every offset from
    /// `tes_slack + 1` to `REF_LEN`. This test is the one that pins WHERE the
    /// edge is, and it is the first to assert the drop COUNT rather than only
    /// the drop's effect on the graph.
    ///
    /// The four buckets this fixture cannot trip are asserted zero HERE, and
    /// that assertion is only meaningful because
    /// `test_flag_drop_buckets_are_observable` pins two of them non-zero on
    /// real records.
    #[test]
    fn test_tes_filter_boundary_is_exact_and_the_ledger_closes() {
        const TES_SLACK: usize = 35;

        // One read at the last allowed offset (kept), one one base further out
        // (dropped), one full length (kept).
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
        // This assertion is only worth writing because a SIBLING test shows the
        // same expression going non-zero; a function that can only return 0
        // satisfies `== 0` trivially.
        assert_eq!(s.ledger_deficit(), 0, "ledger must close: {s:?}");
        assert_eq!(s.dropped_total(), 1, "{s:?}");

        // The two surviving reads are the ones the boundary predicts, named
        // rather than counted: a count of 2 would also be satisfied by the
        // wrong two.
        let kept: Vec<&str> = graph.molecules.iter().map(|m| m.read_id.as_str()).collect();
        assert!(kept.contains(&"at_boundary"), "kept {kept:?}");
        assert!(kept.contains(&"full_length"), "kept {kept:?}");
        assert!(!kept.contains(&"past_boundary"), "kept {kept:?}");

        // Buckets this fixture cannot exercise. Pinned non-zero by
        // `test_flag_drop_buckets_are_observable`.
        assert_eq!(s.dropped_unmapped, 0, "{s:?}");
        assert_eq!(s.dropped_supplementary, 0, "{s:?}");
        assert_eq!(s.dropped_no_poly_a_tag, 0, "{s:?}");
        assert_eq!(s.dropped_invalid_poly_a, 0, "{s:?}");
    }

    /// The two FLAG-only drop buckets are reachable and non-zero.
    ///
    /// This is the sibling that makes the zero-assertions in
    /// `test_tes_filter_boundary_is_exact_and_the_ledger_closes` non-trivial,
    /// and it is also the mutation target for the corresponding increments:
    /// delete either `fc.dropped_* += 1` and this test goes red.
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
        // This assertion is only worth writing because a SIBLING test shows the
        // same expression going non-zero; a function that can only return 0
        // satisfies `== 0` trivially.
        assert_eq!(s.ledger_deficit(), 0, "ledger must close: {s:?}");
        assert_eq!(
            graph.molecules.len(),
            1,
            "only the survivor reaches the graph"
        );
    }

    // ---------------------------------------------------------------------
    // RED PROOF for the production `ledger_deficit() == 0` assertion.
    //
    // Every other test in this module exercises that assertion's GREEN side.
    // Nothing had ever made it go red, which left it indistinguishable from
    // an assertion that could never fire.
    //
    // These fixtures are raw BAM BYTES rather than `write_synthetic_bam`
    // records, and they have to be: the values under test are "no reference"
    // (refID = -1), "no position" (pos = -1) and "no read name" (`*\0`), and in
    // noodles' record API the first two are `usize` and the third is non-empty.
    // None of the three is representable there -- they exist only in the
    // encoded record. `bam::io::Writer` cannot express these records, so a byte
    // fixture is the only way to reach the code under test.
    //
    // Each file is ONE record, identical apart from the one field named: FLAG 0
    // (mapped, forward, primary), 20M at offset 980 of a 1000 bp `tx1`,
    // `pt:i:70`, `tx1` the only reference. It clears every COUNTED filter --
    // unmapped, supplementary, poly(A) tag, poly(A) value, no CIGAR span,
    // empty span, 3' TES window -- so the drop site under test is the only one
    // that can be reached.
    // ---------------------------------------------------------------------

    /// `refID = -1` on a record whose FLAG says mapped. `reference_sequence_id()`
    /// yields nothing, so the `_ => continue` is taken. That is UPSTREAM of
    /// `fc.records_with_candidates += 1`, so the ledger identity does not
    /// balance and the assertion must panic.
    #[test]
    #[should_panic(expected = "evidence filter ledger does not close")]
    fn test_ledger_guard_panics_on_a_record_with_no_reference_sequence_id() {
        let path = write_raw_bam("notid", BAM_NO_REFID);
        // No cleanup: the panic unwinds past it. The file is 160 bytes in the
        // system temp dir and is overwritten by the pid-keyed name on rerun.
        let _ = extract_evidence(&path, 35);
    }

    /// `pos = -1`, so `alignment_start().get()` yields nothing and the
    /// `_ => continue` is taken. Also upstream of the RHS term; same ledger
    /// arithmetic as above.
    #[test]
    #[should_panic(expected = "evidence filter ledger does not close")]
    fn test_ledger_guard_panics_on_a_record_with_no_alignment_start() {
        let path = write_raw_bam("nopos", BAM_NO_POS);
        let _ = extract_evidence(&path, 35);
    }

    /// The third of the three malformed-record exits, and the one that used to be
    /// invisible.
    ///
    /// QNAME is the two bytes `*\0`. noodles' `fields::name()` returns `None`
    /// for exactly two byte patterns -- `*\0` and `\0\0` -- and for nothing
    /// else, so this is the only read-name that reaches the `None => continue`.
    /// (An EMPTY read name, `l_read_name = 0`, returns `Some("")` and does
    /// not; nor does QNAME `*` written through a SAM/BAM writer, which
    /// normalises it.)
    ///
    /// `fc.records_with_candidates += 1` sits after the name check, so this
    /// exit is upstream of the RHS term like the other two, and the assertion
    /// fires. `FilterCounts` gained no field; only the placement moved.
    ///
    /// All three exits -- refID, pos, read name -- now trip the same assertion,
    /// so the three `#[should_panic]` tests above and here are the complete
    /// set. Adding a fourth malformed-record exit downstream of the increment
    /// would reopen the bug silently; the invariant is stated at the increment.
    #[test]
    #[should_panic(expected = "evidence filter ledger does not close")]
    fn test_ledger_guard_panics_on_a_record_with_no_read_name() {
        let path = write_raw_bam("nameless", BAM_NO_NAME);
        let _ = extract_evidence(&path, 35);
    }

    /// The pre-change behaviour, kept as a fixture-level assertion.
    ///
    /// It does NOT run `extract_evidence` -- that panics now, by design. It
    /// reproduces the arithmetic the old placement produced, from the same
    /// counters the guard reads, and shows why that arithmetic balanced: the
    /// RHS term was incremented before the name check, so a record dropped
    /// there was counted on BOTH sides. 1 entered, 0 dropped, 1 "with
    /// candidates", deficit 0 -- and no molecule. Conservation held; the loss
    /// did not get recorded. That is the whole lesson, so it is asserted rather
    /// than described.
    #[test]
    fn test_the_old_rhs_placement_balanced_the_ledger_while_losing_the_record() {
        // The bug, arithmetic only: the increment ran BEFORE the name check, so
        // a record dropped at :642 was counted as one that reached the push. A
        // record dropped at :642 incremented NO bucket of its own.
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
             is gone. This is what the 2026-10-14 fixture returned before the \
             increment moved: Ok, deficit 0, zero molecules. {s:?}"
        );

        // And the fix: with the increment downstream of :642, the same record
        // leaves the RHS term uncounted and the guard sees it.
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

    // ---- fixture bytes ----
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
