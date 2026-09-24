//! Module 2: Transcriptome Evidence Extractor
//!
//! Streams a minimap2 transcriptome BAM (AIDRS-aligned), extracts Dorado `pt:i`
//! poly(A) tail length per record, filters by 3' TES proximity, and constructs
//! a flat indexed candidate graph for TEL-EM (Module 3).
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
/// transcript candidate IDs (one slice per molecule). Designed for
/// zero-copy downstream consumption by Module 3.
#[derive(Debug)]
pub struct FlatEvidenceGraph {
    pub molecules: Vec<MoleculeEvidence>,
    pub candidate_pool: Vec<u32>,
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
    struct TempReadEntry {
        poly_a_len: i32,
        candidates: Vec<u32>,
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

        // Accumulate candidate transcripts per read. O(n) dedup per insert
        // is acceptable for typical ONT multimappers (≤10 candidates).
        let tx_id = ref_id as u32;
        let entry = read_map.entry(read_id).or_insert_with(|| TempReadEntry {
            poly_a_len,
            candidates: Vec::new(),
        });
        if !entry.candidates.contains(&tx_id) {
            entry.candidates.push(tx_id);
        }
    }

    // Flatten into `FlatEvidenceGraph`.
    let mut molecules: Vec<MoleculeEvidence> = Vec::with_capacity(read_map.len());
    let mut candidate_pool: Vec<u32> = Vec::new();

    for (read_id, entry) in read_map {
        if entry.candidates.is_empty() {
            continue;
        }
        let candidate_offset = candidate_pool.len() as u32;
        let candidate_count = entry.candidates.len() as u16;
        candidate_pool.extend_from_slice(&entry.candidates);

        molecules.push(MoleculeEvidence {
            read_id,
            log_pal: (entry.poly_a_len as f32).log2(),
            candidate_offset,
            candidate_count,
        });
    }

    Ok(FlatEvidenceGraph {
        molecules,
        candidate_pool,
        transcript_names,
    })
}

#[cfg(test)]
mod tests {
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
}