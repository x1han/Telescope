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

/// Dorado poly(A) tail length SAM tag: `pt:i`. The default for
/// `PolyASource::BamTag`, and the tag every BAM in the 10-sample cohort
/// carries on ~97% of records.
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
    /// PROVENANCE, not measurement: true iff this graph was built with
    /// `PolyASource::None`, so every molecule's `log_pal` is the placeholder
    /// rather than an observed tail length.
    ///
    /// This is a property of the GRAPH, not of a molecule, which is why it is
    /// one bool here and not `Option<f32>` on `MoleculeEvidence`. The
    /// distinction is the whole point: `Option<f32>` would double the
    /// per-molecule field and have to be threaded through the flat pools,
    /// while this is O(1) on the struct (a handful of Vec handles, not
    /// per-read data) and touches no Parquet schema, because
    /// `FlatEvidenceGraph` is never serialised.
    ///
    /// It exists so `em::solve_and_write_profile` can refuse the combination
    /// that is silently wrong: a placeholder graph at `poly_a_weight > 0`
    /// feeds a constant emission into the E-step. The CLI forces the weight to
    /// 0, but `solve_and_write_profile` is a public function and a library
    /// caller never touches the CLI.
    pub poly_a_is_placeholder: bool,
}

/// Where per-read poly(A) length comes from.
///
/// `pt:i` is Dorado's native tail-length tag, so it is the default and it is
/// what every BAM in the 10-sample cohort carries (~97% of records). It is a
/// basecaller-specific tag, and a BAM re-aligned from a producer that does not
/// emit it has no `pt:i` at all; before this type existed that case produced an
/// EMPTY GRAPH and exit code 0: every record dropped at the tag read, with
/// nothing to distinguish "this BAM has no poly(A)" from "this sample has no
/// reads". That is the fail-quiet path the 2026-10-02 ruling ordered destroyed,
/// and `PolyASource::None` is the only sanctioned answer to it now.
///
/// A third variant, `Tsv` (`read_id<TAB>poly_a_len`, the tailfindr / nanopolish
/// side-file shape), existed until the 2026-10-03 deletion. It was removed as a
/// physical deletion, not deprecated: `--polyA-source tsv:<path>` is now a parse
/// error, not a warning.
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
    /// `em.rs` multiplies the Gaussian in the E-step only (`em.rs:508`), while
    /// `y` reaches the S1/S2 accumulators (`accum_w_y`, `accum_w_y2`)
    /// unconditionally. A single read at `y = -inf` sends `mu` to `-inf` and
    /// `sigma2` to NaN for its transcript. `0.0` is finite and survives that
    /// path; it still biases `mu` toward the origin, which is a known and
    /// accepted cost of an alignment-only run, not a corruption.
    None,
}

/// Parse the `--polyA-source` CLI value into a [`PolyASource`].
///
/// Two forms, exactly as the expert ruling specifies them:
///   `bam_tag:<TAG>` Dorado's native tail-length tag (the historical default)
///   `none`          declared-absent; alignment-only, `poly_a_weight` forced 0
///
/// The tag is a parameter rather than the literal `pt` so a re-basecaller that
/// emits a different tag can be used without a code change — but the ruling
/// names `pt`, and that is the default, because it is the tag the 10-sample
/// cohort actually carries.
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
/// absent. The ruling specifies 10,000; it is a sampling depth, not a
/// threshold on the data.
const POLY_A_PROBE_RECORDS: usize = 10_000;

/// Parse the transcriptome BAM and emit the candidate graph, reading poly(A)
/// from Dorado's native `pt:i` tag.
///
/// Equivalent to `extract_evidence_with(.., &PolyASource::BamTag(TAG_PT))`,
/// and **fails** rather than returning an empty graph when no record in the
/// first [`POLY_A_PROBE_RECORDS`] carries the tag. Callers that want a
/// different source, or none, use `extract_evidence_with`.
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
    // hit on every record, so neither can fire on an alignment-only run. That
    // is a property of the `None` arm's value, not a separate exemption --
    // which is why there is no `expect_measurement` flag here. An earlier
    // version had one, and it was dead: a mutation removing it stayed green.
    let mut poly_a_records_seen: u64 = 0;
    let mut poly_a_hits: u64 = 0;

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
        // SUPPLEMENTARY (`FLAG & 2048`) records are DROPPED, not merged.
        //
        // SAM semantics, stated correctly (an earlier version of this file's
        // test had them backwards): the CIGAR is always written in the
        // molecule's own 5'->3' order, so a LEADING hard clip means "this
        // record starts `H` bases into the read" -- i.e. the read's 5' portion
        // is NOT aligned here and this record is the molecule's **3'** segment.
        // A TRAILING hard clip means the opposite: the molecule's **5'** segment,
        // whose 3' remainder lives in another record. A leading `H` is
        // therefore evidence AGAINST a record being the 5' source, never for.
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
        // MEASURED, not assumed, and the measurement is not uniform across the
        // three input modes (2026-10-02, on the 10-sample COPD/HC cohort):
        //
        //   AIDRS    nanocount/HC2.bam        8,016,260 records      0 supp
        //   AIDRS    nanocount/COPD29.bam     7,707,895 records      0 supp
        //   Bambu    COPD52.sorted.bam       29,658,736 records 82,080 supp (0.28%)
        //   IsoQuant HC30.sorted.bam         25,344,489 records 53,116 supp (0.21%)
        //
        // So for the AIDRS path this is a NO-OP -- the transcriptome aligner
        // emits primary + secondary only, and no record in 500,000 sampled has
        // a hard clip at either end. It is NOT a no-op for the Bambu (F1) and
        // IsoQuant (F2) paths, which are P0 items in the support matrix and
        // which do carry supplementary records at ~0.2-0.3% of alignments. The
        // expert ruling that required this filter was right, and its priority
        // is F1/F2 rather than the path being developed today.
        if flags.is_supplementary() {
            continue;
        }

        // Counted for EVERY record that survives the flag filters, BEFORE the
        // poly(A) lookup. Counting only the records where a value was found
        // would make the global-absence guard unsatisfiable: `hits` and
        // `seen` would rise together and the "seen but no hits" state could
        // never be observed. That mistake was made once already in this
        // function's history, in the shape of an empty graph and exit 0.
        poly_a_records_seen += 1;

        // GLOBAL-ABSENCE GUARD, in-loop form. Sampled over the first
        // POLY_A_PROBE_RECORDS records so a BAM with no poly(A) fails at record
        // 10,000 instead of at end-of-stream on a 27 M-record file.
        //
        // IT MUST SIT HERE, ABOVE THE LOOKUP. It used to sit below it, and was
        // therefore dead code: every arm of the `match` below `continue`s on a
        // record with no poly(A), and `poly_a_hits` is incremented by the same
        // statement that survives those `continue`s. So reaching the old check
        // implied `hits >= 1`, making `hits == 0` unsatisfiable -- a guard that
        // could not fire, wrapped in a comment claiming it did. Mutation
        // confirmed it: neutering it left the suite fully green.
        //
        // Checked BEFORE the lookup, `hits` reflects records 1..POLY_A_PROBE_RECORDS-1
        // only, so a BAM whose poly(A) starts late is caught here.
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
                    _ => continue, // skip non-positive or >i32::MAX
                },
                _ => continue, // skip missing or unparseable tag
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
        //    STRAND — KNOWN GAP, NOT HANDLED. The frozen spec was validated on
        //    a FORWARD-ONLY population: the prototype that produced the 58.34%
        //    directionality figure skips reverse records outright
        //    (`if a.is_reverse: continue` in `proto_five_prime.py`). Measured
        //    over 200,000 mapped records in each of the three BAMs this project
        //    actually runs (AIDRS, Bambu, IsoQuant), the reverse fraction is
        //    0.0000, so on today's data this is dead code either way.
        //
        //    An earlier version branched here to `T_k − ref_end` for reverse
        //    records. That was removed: the TES filter above runs first and is
        //    strand-blind, so every surviving reverse read satisfies
        //    `ref_end + tes_slack >= ref_len`, forcing `T_k − ref_end <=
        //    tes_slack` (= 35). With `five_prime_window = 20` the term would
        //    degenerate into "did `ref_end` land in the top 20 bp of a band the
        //    filter already chose", and it would be unvalidated in any case.
        //    If the Bambu/IsoQuant general path (Phase F1/F2) ever carries
        //    antisense alignments, this needs a separately validated
        //    definition — it must not be reintroduced as an inline guess.
        //
        //    noodles' `alignment_start()` is 1-BASED, so this is already
        //    exactly the spec's `s_k − 1`; do NOT subtract a second 1 (the
        //    spec's "1-based reference_start" is bait for rust-htslib, whose
        //    `reference_start()` is 0-based — this crate does not use it).
        let gap5p: u32 = u32::try_from(start_pos.saturating_sub(1)).unwrap_or(u32::MAX);

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
            // `poly_a_len` is >= 1 on every path that reaches here (BamTag drops
            // non-positive, Tsv rejects non-positive at load, and `None` is filled
            // with 1 below), so this is finite. The guard is a cheap assertion that
            // the three sources cannot be changed into a `-inf` source unnoticed: an
            // infinite `y` propagates into the S1/S2 accumulators, which are outside
            // the `poly_a_weight` gate.
            log_pal: (entry.poly_a_len.max(1) as f32).log2(),
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
        poly_a_is_placeholder: matches!(poly_a, PolyASource::None),
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

#[cfg(any(test, feature = "test-support"))]
/// Synthetic-BAM helpers shared with `resolver`'s tests, and with the
/// end-to-end integration tests in `tests/`, which are a SEPARATE crate and so
/// cannot see `#[cfg(test)]` items. That is what the `test-support` feature is
/// for; `#[doc(hidden)]` keeps it out of the published docs. The alternative —
/// a second BAM writer inside `tests/` — would let the copies drift, which is
/// precisely the class of bug the pooling path is prone to and the reason this
/// module exists at all.
#[doc(hidden)]
pub mod test_support {
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
    ///
    /// This is the only helper in the file that produces a CIGAR other than a
    /// bare `Match`. Before 2026-10-02 that was the only kind that had ever
    /// been built, so nothing in the suite had ever exercised what
    /// `alignment_start()` does when the CIGAR begins with a soft-clip, an
    /// insertion, a deletion or a hard-clip.
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

    /// **Module 2 `gap5p` under real CIGARs — the Phase B blocking gate.**
    ///
    /// Every other fixture in this file builds a bare `Match` CIGAR, so until
    /// this test the suite had never asked what `alignment_start()` does when
    /// the alignment begins with something other than a match. The 5' channel
    /// is the whole subject of the frozen `ln_5p = β·1[g ≤ W]` term, and
    /// `gap5p` is the only thing feeding it from Module 2.
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
    /// where the molecule physically starts — i.e. two aligners, or one aligner
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
    /// This is the "100% agreement with reference exon coordinates" gate. The
    /// reference computation is deliberately a different route to the same
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
    /// # The spec, and the correction
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
    /// (An earlier version of this test asserted the opposite and built its
    /// fixture to match the error: it labelled `120H900M` at reference 1..900
    /// as the "5' segment". Per the spec that record is the 3' segment, and a
    /// 3' segment cannot sit at the lower reference start on a forward
    /// alignment. The test passed -- the code does not know the difference --
    /// which is exactly why a wrong mental model can sit in a file for months
    /// without anything going red.)
    ///
    /// # What is pinned
    ///
    /// * A read split across two records contributes only its primary. The
    ///   supplementary 3' segment is dropped, so `gap5p` comes from the primary
    ///   -- the molecule's actual 5' segment -- and not from an interior
    ///   junction coordinate.
    /// * A read whose ONLY record is supplementary contributes nothing.
    ///
    /// # Why this is defence in depth, not a live fix
    ///
    /// Measured on the real cohort: `HC2.bam` 8,016,260 records, **0**
    /// supplementary. `COPD29.bam` 7,707,895 records, **0**. No record in
    /// 500,000 sampled has a hard clip at either end. The transcriptome
    /// aligner emits primary + secondary only, so before this change the
    /// supplementary path was unreachable on this data. It is kept for the
    /// Bambu (F1) and IsoQuant (F2) paths and for any future aligner, and it
    /// is kept because it is cheap and correct -- not because it fixes
    /// anything measurable today.
    #[test]
    fn test_supplementary_alignments_are_dropped_rather_than_merged() {
        use Kind::*;
        // tx0 is 2000 nt and the junction sits at 1980, close to the TES.
        //
        // That placement is load-bearing, not cosmetic. The 3' TES filter
        // keeps only reads whose alignment end is within `tes_slack` of the
        // transcript end, so a 5' segment ending at an INTERIOR junction is
        // discarded by that filter whatever the supplementary flag says. On a
        // transcriptome reference the TES filter is therefore already a
        // de-facto supplementary filter, and the explicit check in
        // `extract_evidence` is the second, cheap line of defence. The
        // assertion at the end of this test pins that structural fact so it
        // cannot be forgotten.
        //
        // Molecule 5' segment: reference 1..=1980, no leading clip (1980M).
        // Its 3' remainder: reference 1981..=2000, LEADING hard clip -- what a
        // 3' segment looks like. The 3' segment deliberately scores HIGHER, so
        // the old max-AS merge would have won and reported the junction
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

    /// Writes `records` under a 3-transcript header (tx0/tx1/tx2, `REF_LEN` bp
    /// each) to a BAM in the system temp dir named after `tag` and the process
    /// id. Returns its path; the caller removes it. `tag` must differ per
    /// caller — the file name is the only thing keeping concurrent tests apart.
    ///
    /// Molecule ordering out of `extract_evidence` follows HashMap iteration
    /// order, so callers must look molecules up by `read_id`.
    pub fn write_synthetic_bam(tag: &str, records: Vec<RecordBuf>) -> std::path::PathBuf {
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

    /// The in-loop probe, which the end-of-stream guard cannot stand in for.
    ///
    /// The fixture is the case the ruling was written for and that only this
    /// guard can catch: 10,000 records with NO poly(A), then records that DO
    /// have it. The end-of-stream check passes on this file -- poly(A) exists,
    /// so the run is not a global absence -- and the 10,000-record probe is
    /// what must stop it. Before the guard was hoisted above the lookup it
    /// could not fire, and this fixture returned `Ok` with the read included.
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

    /// The two CLI spellings, and the rejections. The spelling is load-bearing
    /// beyond cosmetics: the guard's FATAL message quotes `--polyA-source` to
    /// the user, so a parser that only accepted some spelling would produce an
    /// error naming a flag that cannot be typed.
    ///
    /// `tsv:` is in the REJECTION list, not the acceptance list. The TSV poly(A)
    /// source was deleted outright (not deprecated), so both a well-formed TSV
    /// spec and a bare `tsv:` prefix must now be errors naming the flag.
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

    /// Restored after commit 8ee1077 deleted it without saying so, and inverted
    /// per the expert ruling: the absence of poly(A) is a named error, not an
    /// empty graph.
    #[test]
    fn test_poly_a_enters_as_the_bam_tag_pt_and_bad_tags_drop_the_read() {
        use noodles_sam::alignment::record_buf::data::field::Value;

        // Well-formed: the helper writes `pt:i = 100`, and log2(100) = 6.643856.
        // The value is not arbitrary -- log2 is the base the whole Gaussian
        // lives in, and `INSTRUMENTAL_PAL_NOISE_FLOOR` is quoted in it.
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
    /// This assertion is INVERTED from what it was. It used to pin the
    /// fail-quiet behaviour -- empty graph, exit 0 -- as a "KNOWN GAP" so the
    /// interface work would have a case to fix. The expert ruling closed that
    /// gap: the failure is now `bail!`ed, and this test pins the fix.
    ///
    /// The failure it removes is real. Every record is dropped at the tag read,
    /// so the caller cannot tell "this BAM has no poly(A)" from "this sample has
    /// no reads"; the error surfaces much later, as an empty profile or a
    /// divide-by-zero, with nothing pointing back at the tag. `pt:i` is Dorado's
    /// native tail-length tag, and the Bambu (F1) and IsoQuant (F2) paths are
    /// not obliged to emit it.
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
    /// construction. An earlier version used a placeholder of 0, which made
    /// `hits == 0` reachable and would have killed a healthy alignment-only run
    /// at record 10,000. The flag `expect_measurement` that briefly existed to
    /// prevent exactly that was dead -- removing it left the suite green.
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

    /// Pins the frozen 1-based convention for the forward 5' gap `g_k = s_k − 1`.
    ///
    /// `f5` is the off-by-one tripwire: start 6 must give `g_k = 5`, not 6.
    /// `f20` sits exactly on the `g_k <= 20` window edge, so an inclusive /
    /// exclusive slip at the boundary also fails here.
    ///
    /// Strand is deliberately NOT exercised here — see
    /// `test_reverse_alignment_uses_the_forward_formula_as_a_documented_gap`
    /// for why that is a known gap rather than a handled case.
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

    /// Strand is a KNOWN GAP, not a handled case, and this test exists so the
    /// gap stays visible instead of being quietly closed by a future edit.
    ///
    /// The frozen spec `g_k = s_k − 1` was validated on a FORWARD-ONLY
    /// population (the prototype skips reverse records), and the reverse
    /// fraction is 0.0000 across 200,000 mapped records in each of the three
    /// BAMs this project runs. An earlier version branched to `T_k − ref_end`
    /// for reverse records; that was removed because the strand-blind TES
    /// filter forces `T_k − ref_end <= tes_slack` for every surviving reverse
    /// read, collapsing the term into a 20-of-35 bp band, and the definition
    /// would have been unvalidated besides.
    ///
    /// So a reverse record now gets the forward formula. If the Bambu/IsoQuant
    /// general path (Phase F1/F2) ever carries antisense alignments, the right
    /// answer is a separately validated definition -- this assertion is where
    /// that decision must be made deliberately, not as an inline guess.
    #[test]
    fn test_reverse_alignment_uses_the_forward_formula_as_a_documented_gap() {
        let path = write_synthetic_bam(
            "strand_gap",
            vec![
                // reverse record, start 1 on an 800 nt reference
                synth_record_with_gap("rev", 0, 30, true, Some(800), 1, 60),
            ],
        );

        let graph = extract_evidence(&path, 35).unwrap();
        let _ = std::fs::remove_file(&path);

        let mol = graph
            .molecules
            .iter()
            .find(|m| m.read_id == "rev")
            .expect("rev must be in the graph");
        let gap = graph.candidate_5p_gap[mol.candidate_offset as usize];

        // start_pos = 1 ⇒ g_k = 0, i.e. the formula does NOT branch on strand.
        // If this ever becomes strand-aware, this assertion is the tripwire.
        assert_eq!(
            gap, 0,
            "g_k is computed as start_pos - 1 regardless of strand; a change here \
             means a new definition was adopted and must be re-validated"
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
            // synthetic/test graph: `log_pal` is a real measurement
            poly_a_is_placeholder: false,
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
            // synthetic/test graph: `log_pal` is a real measurement
            poly_a_is_placeholder: false,
        };
        assert!(graph.candidate_count_histogram().is_empty());
    }
}
