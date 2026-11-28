use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use crate::gtf::GtfGenes;
use anyhow::{bail, Context, Result};
use arrow::array::{Float64Builder, Int32Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use noodles_bam::io::Reader as BamReader;
use noodles_sam::alignment::record::data::field::Tag;
use noodles_sam::alignment::Record as _;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

const TAG_PT: Tag = Tag::new(b'p', b't');

const TAG_AS: Tag = Tag::new(b'A', b'S');

#[derive(Debug, Clone, Copy)]
pub struct AnchorParams {
    pub tes_slack: usize,

    pub min_coverage: f64,

    pub max_reads_per_sample: usize,

    pub gap_abs: f64,

    pub gap_rel: f64,
}

#[derive(Debug, Clone)]
pub struct AnchorRecord {
    pub sample_id: String,
    pub read_id: String,
    pub gene_id: String,
    pub transcript_id: String,
    pub poly_a_length: i32,
}

pub fn build_anchor_pool<P, Q>(
    bams_and_samples: &[(P, &str)],
    genes: &GtfGenes,
    output: Q,
    params: AnchorParams,
) -> Result<AnchorStats>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let mut all_anchors: Vec<AnchorRecord> = Vec::new();
    let mut stats = AnchorStats::default();

    for (bam, sample_id) in bams_and_samples {
        let (anchors, sample_stats) = extract_anchors_from_bam(bam, sample_id, genes, params)?;
        stats.add(&sample_stats);
        all_anchors.extend(anchors);
    }

    let (distinct_genes, distinct_transcripts) = pool_gene_tx_cardinality(&all_anchors);
    stats.distinct_genes = distinct_genes;
    stats.distinct_transcripts = distinct_transcripts;

    write_anchor_parquet(&all_anchors, &output)
        .with_context(|| format!("Failed to write anchor pool parquet: {:?}", output.as_ref()))?;

    Ok(stats)
}

fn pool_gene_tx_cardinality(anchors: &[AnchorRecord]) -> (usize, usize) {
    let genes: HashSet<&str> = anchors.iter().map(|a| a.gene_id.as_str()).collect();
    let transcripts: HashSet<&str> = anchors.iter().map(|a| a.transcript_id.as_str()).collect();
    (genes.len(), transcripts.len())
}

pub fn degenerate_gene_pool_warning(
    distinct_genes: usize,
    distinct_transcripts: usize,
) -> Option<String> {
    if distinct_transcripts == 0 || distinct_genes != distinct_transcripts {
        return None;
    }
    Some(format!(
        "[telescope][WARN] anchor pool has NO gene structure: {} distinct \
         transcript_id == {} distinct gene_id, i.e. every gene here has \
         exactly one transcript.\n\
         [telescope][WARN]   Consequence: every gene's pool IS its own \
         transcript's pool, so no two transcripts ever share fitted poly(A) \
         parameters through the L2 gene-level tier. At the DEFAULT thresholds \
         (tx_min_reads=10, gene_min_reads=30) this makes L2 unreachable: \
         reaching L2 already requires n < 10, and the gate then demands 30 \
         values from a pool of at most 9.\n\
         [telescope][WARN]   Those thresholds are build-emission flags and are \
         NOT visible here. Lowering --gene-min-reads below --tx-min-reads \
         (e.g. --gene-min-reads 3) DOES make L2 fire on this pool, but it is \
         then pooling a transcript with itself, which is the same fit as L1 at \
         a different threshold -- not gene-level borrowing. An `L2=0` in the \
         downstream stats at the default thresholds is THIS, not a fit \
         failure; do not read it as evidence that the gene fit did not \
         converge.\n\
         [telescope][WARN]   Remedy: pass --gtf <GTF> to build-anchors, so that transcripts \
         sharing a gene genuinely share a pool.",
        distinct_transcripts, distinct_genes,
    ))
}

#[derive(Debug, Clone, Copy)]
struct Staged {
    ref_id: u32,
    start: u32,
    end: u32,
    as_: i32,
    pt: i32,
}

struct ReadGroup {
    key: u128,

    name_off: u32,
    name_len: u32,
    cands: Vec<Staged>,
}

fn read_key(name: &[u8]) -> u128 {
    let mut lo = DefaultHasher::new();
    name.hash(&mut lo);
    let mut hi = DefaultHasher::new();
    0xA5_u8.hash(&mut hi);
    name.hash(&mut hi);
    (u128::from(hi.finish()) << 64) | u128::from(lo.finish())
}

enum GeneRelation {
    Same,
    Unknown,
    Cross,
}

fn resolve_gene(tx: &str, genes: &GtfGenes) -> Result<String> {
    match genes.gene_id(tx) {
        Some(g) => Ok(g.clone()),
        None if genes.is_empty() => Ok(tx.to_string()),
        None => bail!(
            "--gtf has {} transcripts but transcript {:?} is not among them. \
             Refusing to fall back to gene_id := transcript_id: with a real map \
             present a missing transcript means the annotation and the BAM's @SQ \
             header disagree, and the per-read same-gene veto would then be \
             decided on an invented gene id. Fix the annotation or drop the flag.",
            genes.len(),
            tx
        ),
    }
}

fn extract_anchors_from_bam<P: AsRef<Path>>(
    bam: P,
    sample_id: &str,
    genes: &GtfGenes,
    p: AnchorParams,
) -> Result<(Vec<AnchorRecord>, SampleStats)> {
    let path = bam.as_ref();
    let file = File::open(path).with_context(|| format!("Failed to open BAM: {:?}", path))?;
    let mut reader = BamReader::new(file);
    let header = reader.read_header().context("Failed to read BAM header")?;

    let mut ref_len: HashMap<usize, usize> = HashMap::new();
    let ref_seqs: Vec<String> = (0..header.reference_sequences().len())
        .map(|i| {
            let (name, ln) = match header.reference_sequences().get_index(i) {
                Some((n, sq)) => {
                    let l = sq.length().get();
                    (n.to_string(), l)
                }
                None => (String::new(), 0),
            };
            ref_len.insert(i, ln);
            name
        })
        .collect();

    let mut per_tx_count: HashMap<String, usize> = HashMap::new();

    let mut anchors: Vec<AnchorRecord> = Vec::new();
    let mut stats = SampleStats::default();

    let mut name_arena: Vec<u8> = Vec::new();
    let mut groups: Vec<ReadGroup> = Vec::new();
    let mut group_index: HashMap<u128, u32> = HashMap::new();
    let mut read_key_collisions: u64 = 0;

    let mut record = noodles_bam::Record::default();
    loop {
        let read = match reader.read_record(&mut record) {
            Ok(n) => n,
            Err(_) => {
                stats.parse_errors += 1;
                break;
            }
        };
        if read == 0 {
            break;
        }
        stats.total_records += 1;

        if record.flags().is_unmapped() {
            stats.filtered_unmapped += 1;
            continue;
        }

        if record.flags().is_supplementary() {
            stats.filtered_supplementary += 1;
            continue;
        }

        let ref_id = match record.reference_sequence_id() {
            Some(Ok(i)) => i,
            _ => {
                stats.filtered_no_refid += 1;
                continue;
            }
        };
        let alignment_start = match record.alignment_start() {
            Some(Ok(p)) => p.get(),
            _ => {
                stats.filtered_no_position += 1;
                continue;
            }
        };
        let alignment_end = match record.alignment_end() {
            Some(Ok(p)) => p.get(),
            _ => {
                stats.filtered_no_position += 1;
                continue;
            }
        };

        let name: &[u8] = match record.name() {
            Some(n) => n,
            None => {
                stats.filtered_no_read_id += 1;
                continue;
            }
        };

        let rl = *ref_len.get(&ref_id).unwrap_or(&0);
        if rl == 0 {
            stats.filtered_no_reflen += 1;
            continue;
        }

        let poly_a_length: i32 = match record.data().get(&TAG_PT) {
            Some(Ok(value)) => match value.as_int() {
                Some(n) if n > 0 && n <= i32::MAX as i64 => n as i32,
                _ => 0,
            },
            _ => 0,
        };

        let as_score: i32 = match record.data().get(&TAG_AS) {
            Some(Ok(value)) => match value.as_int() {
                Some(n) => i32::try_from(n).unwrap_or(0),
                None => 0,
            },
            _ => 0,
        };

        let key = read_key(name);
        let pos = match group_index.get(&key) {
            Some(&pos) => {
                let g = &groups[pos as usize];
                if &name_arena[g.name_off as usize..(g.name_off + g.name_len) as usize] != name {
                    read_key_collisions += 1;
                }
                pos
            }
            None => {
                let off = name_arena.len() as u32;
                name_arena.extend_from_slice(name);
                groups.push(ReadGroup {
                    key,
                    name_off: off,
                    name_len: name.len() as u32,
                    cands: Vec::new(),
                });
                let pos = (groups.len() - 1) as u32;
                group_index.insert(key, pos);
                pos
            }
        };
        groups[pos as usize].cands.push(Staged {
            ref_id: ref_id as u32,
            start: alignment_start as u32,
            end: alignment_end as u32,
            as_: as_score,
            pt: poly_a_length,
        });
    }

    assert_eq!(
        read_key_collisions, 0,
        "two distinct QNAMEs hashed to the same 128-bit grouping key, so their \
         records were merged into one read group. The per-read argmax then \
         picked one transcript for two molecules. This is the residual risk of \
         keying by hash instead of by name; see `read_key`."
    );

    let mut order: Vec<u32> = (0..groups.len() as u32).collect();
    order.sort_by_key(|&i| groups[i as usize].key);

    let mut scratch: Vec<(Staged, u64, bool)> = Vec::new();
    for &i in &order {
        let g = &mut groups[i as usize];
        stats.reads_seen += 1;

        g.cands
            .sort_by(|a, b| a.ref_id.cmp(&b.ref_id).then_with(|| b.as_.cmp(&a.as_)));

        scratch.clear();
        let mut eligible_records: u64 = 0;
        let mut i = 0;
        while i < g.cands.len() {
            let mut j = i + 1;
            while j < g.cands.len() && g.cands[j].ref_id == g.cands[i].ref_id {
                j += 1;
            }
            let mult = (j - i) as u64;
            let rep = g.cands[i];

            let rl = *ref_len.get(&(rep.ref_id as usize)).unwrap_or(&0);
            let eligible = if rl == 0 {
                stats.filtered_no_reflen += mult;
                false
            } else if (rep.end as usize + p.tes_slack) < rl {
                stats.filtered_3p_anchor += mult;
                false
            } else {
                let aln_len = if rep.end >= rep.start {
                    (rep.end - rep.start + 1) as f64
                } else {
                    0.0
                };
                if aln_len / (rl as f64) < p.min_coverage {
                    stats.filtered_coverage += mult;
                    false
                } else if rep.pt < 10 {
                    stats.filtered_polya_invalid += mult;
                    false
                } else {
                    true
                }
            };
            if eligible {
                eligible_records += mult;
            }
            scratch.push((rep, mult, eligible));
            i = j;
        }

        if eligible_records == 0 {
            stats.reads_no_eligible_candidate += 1;
            continue;
        }
        scratch.sort_by(|a, b| b.0.as_.cmp(&a.0.as_));

        if scratch.len() >= 2 && scratch[0].0.as_ == scratch[1].0.as_ {
            stats.rejected_tie += 1;
            stats.filtered_eligible_not_anchor += eligible_records;
            continue;
        }

        let best = scratch[0].0;
        let runner_up: Option<Staged> = scratch.get(1).map(|e| e.0);

        if !scratch[0].2 {
            stats.rejected_usurpation += 1;
            stats.filtered_eligible_not_anchor += eligible_records;
            continue;
        }

        let gap = (best.as_ as f64) - runner_up.map_or(0.0, |s| s.as_ as f64);
        if gap < p.gap_abs || gap < p.gap_rel * (best.as_ as f64) {
            if let Some(second) = runner_up {
                let g_best = resolve_gene(tx_name(&ref_seqs, best.ref_id), genes)?;
                let g_second = resolve_gene(tx_name(&ref_seqs, second.ref_id), genes)?;
                match relation(&g_best, &g_second, genes) {
                    GeneRelation::Same => stats.rejected_same_gene += 1,

                    GeneRelation::Cross => stats.rejected_cross_gene += 1,
                    GeneRelation::Unknown => stats.rejected_gap += 1,
                }
            } else {
                stats.rejected_gap += 1;
            }
            stats.filtered_eligible_not_anchor += eligible_records;
            continue;
        }

        if let Some(second) = runner_up {
            let rel = relation(
                &resolve_gene(tx_name(&ref_seqs, best.ref_id), genes)?,
                &resolve_gene(tx_name(&ref_seqs, second.ref_id), genes)?,
                genes,
            );
            if matches!(rel, GeneRelation::Unknown) {
                stats.gene_veto_disabled += 1;
                stats.same_gene_unknown += 1;
            }
        }

        let transcript_id = tx_name(&ref_seqs, best.ref_id).to_string();
        let gene_id = resolve_gene(&transcript_id, genes)?;

        let count = per_tx_count.entry(transcript_id.clone()).or_insert(0);
        if *count >= p.max_reads_per_sample {
            stats.filtered_cap += 1;
            stats.filtered_eligible_not_anchor += eligible_records - 1;
            stats.reads_over_cap += 1;
            continue;
        }
        *count += 1;

        let read_id = String::from_utf8_lossy(
            &name_arena[g.name_off as usize..(g.name_off + g.name_len) as usize],
        )
        .into_owned();
        anchors.push(AnchorRecord {
            sample_id: sample_id.to_string(),
            read_id,
            gene_id,
            transcript_id,
            poly_a_length: best.pt,
        });
        stats.passed += 1;
        stats.anchors_emitted += 1;

        stats.filtered_eligible_not_anchor += eligible_records - 1;
    }

    assert_eq!(
        stats.filtered_cap, stats.reads_over_cap,
        "filtered_cap counts rejected READS and must stay equal to reads_over_cap; \
         {} vs {} means one of them was incremented without the other",
        stats.filtered_cap, stats.reads_over_cap
    );

    assert_eq!(
        stats.reads_seen,
        stats.anchors_emitted
            + stats.rejected_tie
            + stats.rejected_usurpation
            + stats.rejected_gap
            + stats.rejected_same_gene
            + stats.rejected_cross_gene
            + stats.reads_no_eligible_candidate
            + stats.reads_over_cap,
        "every read that reached the per-read stage must land in exactly one \
         outcome. reads_seen={} anchors={} tie={} usurpation={} gap={} \
         same_gene={} cross_gene={} no_eligible={} over_cap={} \
         (gene_veto_disabled={} is NOT a destination: those reads cleared the \
         gap and were emitted)",
        stats.reads_seen,
        stats.anchors_emitted,
        stats.rejected_tie,
        stats.rejected_usurpation,
        stats.rejected_gap,
        stats.rejected_same_gene,
        stats.rejected_cross_gene,
        stats.reads_no_eligible_candidate,
        stats.reads_over_cap,
        stats.gene_veto_disabled,
    );

    Ok((anchors, stats))
}

fn tx_name(ref_seqs: &[String], ref_id: u32) -> &str {
    ref_seqs.get(ref_id as usize).map_or("", String::as_str)
}

fn relation(a: &str, b: &str, genes: &GtfGenes) -> GeneRelation {
    if genes.is_empty() {
        GeneRelation::Unknown
    } else if a == b {
        GeneRelation::Same
    } else {
        GeneRelation::Cross
    }
}

fn write_anchor_parquet<P: AsRef<Path>>(anchors: &[AnchorRecord], output: P) -> Result<()> {
    let mut sample_b = StringBuilder::new();
    let mut read_b = StringBuilder::new();
    let mut gene_b = StringBuilder::new();
    let mut tx_b = StringBuilder::new();
    let mut pal_b = Int32Builder::new();
    let mut w_b = Float64Builder::new();

    for a in anchors {
        sample_b.append_value(&a.sample_id);
        read_b.append_value(&a.read_id);
        gene_b.append_value(&a.gene_id);
        tx_b.append_value(&a.transcript_id);
        pal_b.append_value(a.poly_a_length);
        w_b.append_value(1.0_f64);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_id", DataType::Utf8, false),
        Field::new("read_id", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("transcript_id", DataType::Utf8, false),
        Field::new("polyA_length", DataType::Int32, false),
        Field::new("posterior_weight", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(sample_b.finish()),
            Arc::new(read_b.finish()),
            Arc::new(gene_b.finish()),
            Arc::new(tx_b.finish()),
            Arc::new(pal_b.finish()),
            Arc::new(w_b.finish()),
        ],
    )
    .context("Failed to build anchor RecordBatch")?;

    let out_file = File::create(&output)
        .with_context(|| format!("Failed to create anchor pool file: {:?}", output.as_ref()))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(out_file, schema, Some(props))
        .context("Failed to create ArrowWriter")?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[derive(Debug, Default, Clone)]
pub struct SampleStats {
    pub total_records: u64,
    pub parse_errors: u64,

    pub filtered_unique: u64,

    pub filtered_unmapped: u64,

    pub filtered_supplementary: u64,

    pub filtered_no_refid: u64,

    pub filtered_no_position: u64,

    pub filtered_no_read_id: u64,
    pub filtered_no_reflen: u64,
    pub filtered_3p_anchor: u64,
    pub filtered_coverage: u64,
    pub filtered_polya_invalid: u64,

    pub filtered_eligible_not_anchor: u64,

    pub filtered_cap: u64,

    pub passed: u64,

    pub reads_seen: u64,

    pub anchors_emitted: u64,

    pub rejected_tie: u64,

    pub rejected_usurpation: u64,

    pub rejected_gap: u64,

    pub rejected_same_gene: u64,

    pub rejected_cross_gene: u64,

    pub gene_veto_disabled: u64,

    pub same_gene_unknown: u64,

    pub reads_no_eligible_candidate: u64,

    pub reads_over_cap: u64,
}

#[derive(Debug, Default, Clone)]
pub struct AnchorStats {
    pub samples: usize,
    pub total_records: u64,

    pub parse_errors: u64,
    pub filtered_unique: u64,
    pub filtered_unmapped: u64,
    pub filtered_supplementary: u64,
    pub filtered_no_refid: u64,
    pub filtered_no_position: u64,
    pub filtered_no_read_id: u64,
    pub filtered_no_reflen: u64,
    pub filtered_3p_anchor: u64,
    pub filtered_coverage: u64,
    pub filtered_polya_invalid: u64,
    pub filtered_eligible_not_anchor: u64,
    pub filtered_cap: u64,
    pub passed: u64,

    pub reads_seen: u64,
    pub anchors_emitted: u64,
    pub rejected_tie: u64,
    pub rejected_usurpation: u64,
    pub rejected_gap: u64,
    pub rejected_same_gene: u64,
    pub rejected_cross_gene: u64,
    pub gene_veto_disabled: u64,
    pub same_gene_unknown: u64,
    pub reads_no_eligible_candidate: u64,
    pub reads_over_cap: u64,

    pub distinct_genes: usize,

    pub distinct_transcripts: usize,
}

impl AnchorStats {
    fn add(&mut self, s: &SampleStats) {
        self.samples += 1;
        self.total_records += s.total_records;
        self.parse_errors += s.parse_errors;
        self.filtered_unique += s.filtered_unique;
        self.filtered_unmapped += s.filtered_unmapped;
        self.filtered_supplementary += s.filtered_supplementary;
        self.filtered_no_refid += s.filtered_no_refid;
        self.filtered_no_position += s.filtered_no_position;
        self.filtered_no_read_id += s.filtered_no_read_id;
        self.filtered_no_reflen += s.filtered_no_reflen;
        self.filtered_3p_anchor += s.filtered_3p_anchor;
        self.filtered_coverage += s.filtered_coverage;
        self.filtered_polya_invalid += s.filtered_polya_invalid;
        self.filtered_eligible_not_anchor += s.filtered_eligible_not_anchor;
        self.filtered_cap += s.filtered_cap;
        self.passed += s.passed;
        self.reads_seen += s.reads_seen;
        self.anchors_emitted += s.anchors_emitted;
        self.rejected_tie += s.rejected_tie;
        self.rejected_usurpation += s.rejected_usurpation;
        self.rejected_gap += s.rejected_gap;
        self.rejected_same_gene += s.rejected_same_gene;
        self.rejected_cross_gene += s.rejected_cross_gene;
        self.gene_veto_disabled += s.gene_veto_disabled;
        self.same_gene_unknown += s.same_gene_unknown;
        self.reads_no_eligible_candidate += s.reads_no_eligible_candidate;
        self.reads_over_cap += s.reads_over_cap;
    }

    pub fn report(&self) -> String {
        format!(
            "samples={} total={} filtered(unique={} no_reflen={} 3p_anchor={} coverage={} \
             polya={} cap={}) passed={} unaccounted={} no_refid={} no_position={} no_read_id={} \
             unmapped={} supplementary={} not_anchor={} \
             reads(seen={} anchors={} tie={} gap={} same_gene={} cross_gene={} \
             veto_disabled={} same_gene_unknown={} no_eligible={} over_cap={} usurpation={})",
            self.samples,
            self.total_records,
            self.filtered_unique,
            self.filtered_no_reflen,
            self.filtered_3p_anchor,
            self.filtered_coverage,
            self.filtered_polya_invalid,
            self.filtered_cap,
            self.passed,
            self.unaccounted(),
            self.filtered_no_refid,
            self.filtered_no_position,
            self.filtered_no_read_id,
            self.filtered_unmapped,
            self.filtered_supplementary,
            self.filtered_eligible_not_anchor,
            self.reads_seen,
            self.anchors_emitted,
            self.rejected_tie,
            self.rejected_gap,
            self.rejected_same_gene,
            self.rejected_cross_gene,
            self.gene_veto_disabled,
            self.same_gene_unknown,
            self.reads_no_eligible_candidate,
            self.reads_over_cap,
            self.rejected_usurpation,
        )
    }

    pub fn unaccounted(&self) -> u64 {
        let bucket_sum = self.filtered_unique
            + self.filtered_unmapped
            + self.filtered_supplementary
            + self.filtered_no_refid
            + self.filtered_no_position
            + self.filtered_no_read_id
            + self.filtered_no_reflen
            + self.filtered_3p_anchor
            + self.filtered_coverage
            + self.filtered_polya_invalid
            + self.filtered_eligible_not_anchor
            + self.filtered_cap
            + self.passed;

        debug_assert!(
            bucket_sum <= self.total_records,
            "buckets sum to {} but total_records is {}: a record reached two \
             buckets, or `passed` was bumped without a push. saturating_sub \
             would report 0 for this and hide it.",
            bucket_sum,
            self.total_records
        );

        self.total_records
            .saturating_sub(self.filtered_unique)
            .saturating_sub(self.filtered_unmapped)
            .saturating_sub(self.filtered_supplementary)
            .saturating_sub(self.filtered_no_refid)
            .saturating_sub(self.filtered_no_position)
            .saturating_sub(self.filtered_no_read_id)
            .saturating_sub(self.filtered_no_reflen)
            .saturating_sub(self.filtered_3p_anchor)
            .saturating_sub(self.filtered_coverage)
            .saturating_sub(self.filtered_polya_invalid)
            .saturating_sub(self.filtered_eligible_not_anchor)
            .saturating_sub(self.filtered_cap)
            .saturating_sub(self.passed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::test_support::{
        synth_record, synth_record_no_position, synth_record_no_read_id, synth_record_no_refid,
        synth_record_with_flags, synth_record_with_gap, write_synthetic_bam,
    };

    fn params() -> AnchorParams {
        AnchorParams {
            tes_slack: 35,
            min_coverage: 0.80,
            max_reads_per_sample: 200,
            gap_abs: 0.0,
            gap_rel: 0.0,
        }
    }

    fn params_with_gap(gap_abs: f64, gap_rel: f64) -> AnchorParams {
        AnchorParams {
            gap_abs,
            gap_rel,
            ..params()
        }
    }

    #[test]
    fn test_anchor_stats_aggregation() {
        let s1 = SampleStats {
            total_records: 100,
            passed: 30,
            ..Default::default()
        };
        let s2 = SampleStats {
            total_records: 50,
            passed: 12,
            ..Default::default()
        };
        let mut agg = AnchorStats::default();
        agg.add(&s1);
        agg.add(&s2);
        assert_eq!(agg.samples, 2);
        assert_eq!(agg.total_records, 150);
        assert_eq!(agg.passed, 42);

        assert_eq!(agg.distinct_genes, 0);
        assert_eq!(agg.distinct_transcripts, 0);
    }

    fn three_tx_anchor_bam(tag: &str) -> std::path::PathBuf {
        let recs = vec![
            synth_record("a0", 0, None, 0, 60),
            synth_record("a1", 0, None, 0, 60),
            synth_record("b0", 1, None, 0, 60),
            synth_record("c0", 2, None, 0, 60),
        ];
        write_synthetic_bam(tag, recs)
    }

    #[test]
    fn test_degenerate_gene_pool_without_a_gtf_warns_and_names_the_consequence() {
        let bam = three_tx_anchor_bam("degen_pos");
        let out = std::env::temp_dir().join(format!(
            "telescope_anchor_degen_pos_{}.parquet",
            std::process::id()
        ));
        let stats = build_anchor_pool(&[(bam.clone(), "S1")], &GtfGenes::new(), &out, params())
            .expect("build_anchor_pool");
        let _ = std::fs::remove_file(&bam);
        let _ = std::fs::remove_file(&out);

        assert_eq!(
            stats.passed, 4,
            "the fixture must actually pass the anchor filters, or this test \
             would be asserting on an empty pool"
        );
        assert_eq!(stats.distinct_transcripts, 3, "tx0/tx1/tx2");
        assert_eq!(
            stats.distinct_genes, 3,
            "gene_id falls back to transcript_id, so the counts are equal"
        );

        let msg = degenerate_gene_pool_warning(stats.distinct_genes, stats.distinct_transcripts)
            .expect("a pool with gene_id == transcript_id on every row MUST warn");
        for (what, needle) in [
            ("the L2 tier's consequence", "L2"),
            ("gene-level wording", "gene-level"),
            (
                "the consequence for shared parameters",
                "no two transcripts ever share",
            ),
            ("the default thresholds it is scoped to", "tx_min_reads=10"),
            ("the remedy", "--gtf"),
        ] {
            assert!(msg.contains(needle), "warning must name {what}; got: {msg}");
        }

        assert!(
            !msg.contains("STRUCTURALLY UNREACHABLE"),
            "must not claim L2 is unconditionally unreachable; the thresholds \
             are build-emission flags and can make it fire; got: {msg}"
        );
    }

    #[test]
    fn test_pool_with_real_gene_structure_does_not_warn() {
        let bam = three_tx_anchor_bam("degen_neg");
        let out = std::env::temp_dir().join(format!(
            "telescope_anchor_degen_neg_{}.parquet",
            std::process::id()
        ));
        let gtf =
            std::env::temp_dir().join(format!("telescope_tx2gene_neg_{}.gtf", std::process::id()));
        std::fs::write(
            &gtf,
            "chr1\tTEST\ttranscript\t1\t100\t.\t+\t.\tgene_id \"GENE_A\"; transcript_id \"tx0\";\n\
             chr1\tTEST\ttranscript\t1\t100\t.\t+\t.\tgene_id \"GENE_A\"; transcript_id \"tx1\";\n\
             chr1\tTEST\ttranscript\t1\t100\t.\t+\t.\tgene_id \"GENE_A\"; transcript_id \"tx2\";\n",
        )
        .unwrap();
        let map = GtfGenes::from_gtf(&gtf).unwrap();
        assert_eq!(map.len(), 3);

        let stats = build_anchor_pool(&[(bam.clone(), "S1")], &map, &out, params())
            .expect("build_anchor_pool");
        let _ = std::fs::remove_file(&bam);
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&gtf);

        assert_eq!(stats.passed, 4);
        assert_eq!(stats.distinct_transcripts, 3);
        assert_eq!(
            stats.distinct_genes, 1,
            "three transcripts on one gene: the counts differ, so L2 is reachable"
        );
        assert!(
            degenerate_gene_pool_warning(stats.distinct_genes, stats.distinct_transcripts)
                .is_none(),
            "a pool with real gene structure must NOT warn"
        );
    }

    #[test]
    fn test_detector_is_exactly_equality_of_the_two_distinct_counts() {
        let row = |gene: &str, tx: &str| AnchorRecord {
            sample_id: "S1".into(),
            read_id: "r".into(),
            gene_id: gene.into(),
            transcript_id: tx.into(),
            poly_a_length: 100,
        };
        let one_tx_per_gene = vec![row("G1", "T1"), row("G1", "T1"), row("G2", "T2")];
        assert_eq!(pool_gene_tx_cardinality(&one_tx_per_gene), (2, 2));
        assert!(
            degenerate_gene_pool_warning(2, 2).is_some(),
            "one-transcript-per-gene is genuinely degenerate and must warn"
        );

        let degenerate = vec![
            row("T1", "T1"),
            row("T1", "T1"),
            row("T2", "T2"),
            row("T3", "T3"),
        ];
        assert_eq!(pool_gene_tx_cardinality(&degenerate), (3, 3));
        assert!(degenerate_gene_pool_warning(3, 3).is_some());

        assert!(degenerate_gene_pool_warning(1, 3).is_none());
        assert!(degenerate_gene_pool_warning(2, 3).is_none());

        assert_eq!(pool_gene_tx_cardinality(&[]), (0, 0));
        assert!(degenerate_gene_pool_warning(0, 0).is_none());
    }

    fn counted_buckets(s: &SampleStats) -> u64 {
        s.filtered_unique
            + s.filtered_unmapped
            + s.filtered_supplementary
            + s.filtered_no_refid
            + s.filtered_no_position
            + s.filtered_no_read_id
            + s.filtered_no_reflen
            + s.filtered_3p_anchor
            + s.filtered_coverage
            + s.filtered_polya_invalid
            + s.filtered_eligible_not_anchor
            + s.filtered_cap
    }

    #[test]
    fn test_anchor_record_conservation() {
        use noodles_sam::alignment::record::data::field::Tag;
        use noodles_sam::alignment::record_buf::data::field::Value;
        let nm: Tag = Tag::new(b'N', b'M');

        let mut nopt = synth_record("nopt", 0, None, 0, 60);
        nopt.data_mut().clear();
        nopt.data_mut().insert(nm, Value::from(1i32));

        let mut recs = vec![
            synth_record("ok0", 0, None, 0, 60),
            synth_record("ok1", 1, None, 0, 60),
            synth_record("lowq", 0, None, 0, 10),
            synth_record_with_gap("gap3p", 1, 100, true, None, 0, 60),
            synth_record_with_gap("shortcov", 1, 500, false, None, 0, 60),
            nopt,
        ];

        for i in 0..205 {
            recs.push(synth_record(&format!("cap{i}"), 2, None, 0, 60));
        }

        let bam = write_synthetic_bam("conserve", recs);
        let (anchors, s) =
            extract_anchors_from_bam(&bam, "conserve", &GtfGenes::new(), params()).unwrap();

        assert_eq!(
            s.total_records,
            counted_buckets(&s) + s.passed,
            "a record left the stream through an uncounted path: \
             total_records={} counted={} passed={}",
            s.total_records,
            counted_buckets(&s),
            s.passed
        );
        assert_eq!(
            anchors.len() as u64,
            s.passed,
            "passed counts records that never reached the pool"
        );

        assert_eq!(s.parse_errors, 0);

        assert_eq!(
            s.filtered_unique, 0,
            "`lowq` carries MAPQ 10 and there is no MAPQ gate any more, so the \
             retired slot stays at 0 and the record becomes an anchor"
        );
        assert_eq!(s.filtered_3p_anchor, 1, "fixture lost its 3'-anchor record");
        assert_eq!(s.filtered_coverage, 1, "fixture lost its coverage record");
        assert_eq!(
            s.filtered_polya_invalid, 1,
            "fixture lost its no-poly(A) record"
        );
        assert_eq!(s.filtered_cap, 5, "fixture lost its over-cap records");
        assert_eq!(s.passed, 203, "fixture lost its passing records");
        assert_eq!(
            s.total_records,
            1 + 1 + 1 + 5 + 203,
            "the fixture's record count changed; re-derive the expectations"
        );
        assert_eq!(s.filtered_no_refid, 0);
        assert_eq!(s.filtered_no_position, 0);
        assert_eq!(s.filtered_no_read_id, 0);
        assert_eq!(s.filtered_no_reflen, 0);

        let _ = std::fs::remove_file(&bam);
    }

    #[test]
    fn test_the_three_observable_drop_paths_really_execute() {
        let recs = vec![
            synth_record("ok0", 0, None, 0, 60),
            synth_record("ok1", 1, None, 0, 60),
            synth_record_no_refid("norid", None, 0, 60),
            synth_record_no_position(2, None, 0, 60),
            synth_record_no_read_id(0, None, 0, 60),
            synth_record_with_gap("gap3p", 1, 100, true, None, 0, 60),
        ];

        let bam = write_synthetic_bam("reach", recs);
        let (anchors, s) =
            extract_anchors_from_bam(&bam, "reach", &GtfGenes::new(), params()).unwrap();

        assert_eq!(
            s.total_records,
            counted_buckets(&s) + s.passed,
            "a record left the stream through an uncounted path: \
             total_records={} counted={} passed={}",
            s.total_records,
            counted_buckets(&s),
            s.passed
        );

        assert_eq!(s.total_records, 6);
        assert_eq!(s.passed, 2);
        assert_eq!(
            s.filtered_no_refid, 1,
            "a record with no tid must reach filtered_no_refid, not filtered_unique"
        );
        assert_eq!(
            s.filtered_no_position, 1,
            "a record with POS=0 must reach filtered_no_position"
        );
        assert_eq!(
            s.filtered_no_read_id, 1,
            "a record with no QNAME must reach filtered_no_read_id"
        );
        assert_eq!(s.filtered_3p_anchor, 1);
        assert_eq!(anchors.len() as u64, s.passed);

        assert_eq!(
            s.total_records as i64 - counted_buckets(&s) as i64 - s.passed as i64,
            0,
            "signed conservation: a negative value means a record was counted twice"
        );

        let _ = std::fs::remove_file(&bam);
    }

    #[test]
    fn test_unaccounted_is_zero_when_every_bucket_is_populated() {
        use noodles_sam::alignment::record::data::field::Tag;
        use noodles_sam::alignment::record_buf::data::field::Value;
        let _ = std::mem::size_of::<Tag>();
        let _ = std::mem::size_of::<Value>();

        let agg = AnchorStats {
            samples: 1,
            total_records: 100,
            parse_errors: 7,
            filtered_unique: 5,
            filtered_unmapped: 12,
            filtered_supplementary: 13,
            filtered_no_refid: 3,
            filtered_no_position: 4,
            filtered_no_read_id: 5,
            filtered_no_reflen: 6,
            filtered_3p_anchor: 7,
            filtered_coverage: 8,
            filtered_polya_invalid: 9,
            filtered_eligible_not_anchor: 10,
            filtered_cap: 11,
            passed: 7,

            reads_seen: 35,
            anchors_emitted: 7,
            rejected_tie: 9,
            rejected_usurpation: 6,
            rejected_gap: 1,
            rejected_same_gene: 3,
            rejected_cross_gene: 2,
            gene_veto_disabled: 1,
            same_gene_unknown: 1,
            reads_no_eligible_candidate: 4,
            reads_over_cap: 2,
            distinct_genes: 2,
            distinct_transcripts: 3,
        };
        assert_eq!(
            agg.unaccounted(),
            0,
            "twelve populated buckets summing to total_records must leave 0"
        );

        let leaking = AnchorStats {
            total_records: 101,
            ..agg.clone()
        };
        assert_eq!(
            leaking.unaccounted(),
            1,
            "unaccounted() must report the gap, not clamp it"
        );
    }

    fn tx2gene_gtf(tag: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "telescope_tx2gene_{tag}_{}.gtf",
            std::process::id()
        ));
        let mut out = String::new();
        for line in body.lines().filter(|l| !l.trim().is_empty()) {
            let mut parts = line.splitn(2, '\t');
            let tx = parts.next().unwrap_or("").trim();
            let gene = parts.next().unwrap_or("").trim();
            if tx.is_empty() || gene.is_empty() {
                continue;
            }
            out.push_str(&format!(
                "chr1\tTEST\ttranscript\t1\t100\t.\t+\t.\tgene_id \"{}\"; transcript_id \"{}\";\n",
                gene, tx
            ));
        }
        std::fs::write(&p, out).unwrap();
        p
    }

    fn run_extract(
        tag: &str,
        recs: Vec<noodles_sam::alignment::record_buf::RecordBuf>,
        map: &GtfGenes,
        p: AnchorParams,
    ) -> (Vec<AnchorRecord>, SampleStats) {
        let bam = write_synthetic_bam(tag, recs);
        let out = extract_anchors_from_bam(&bam, tag, map, p).expect("extract_anchors_from_bam");
        let _ = std::fs::remove_file(&bam);
        out
    }

    fn record_deficit(s: &SampleStats) -> u64 {
        s.total_records - counted_buckets(s) - s.passed
    }

    fn assert_record_ledger_closes(s: &SampleStats) {
        assert_eq!(
            s.total_records,
            counted_buckets(s) + s.passed,
            "a record left the stream through an uncounted path: total_records={} \
             counted={} passed={}",
            s.total_records,
            counted_buckets(s),
            s.passed
        );
    }

    #[test]
    fn test_a_same_gene_pair_with_a_clear_gap_emits_rather_than_being_vetoed() {
        let gtf = tx2gene_gtf("samegene", "tx0\tGA\ntx1\tGA\ntx2\tGA\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();
        assert_eq!(map.len(), 3);

        let (anchors, s) = run_extract(
            "samegene",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            1,
            "AS 100 vs 90 means the read aligned over sequence the runner-up does \
             not contain; on this data a single mismatch costs 6 AS, so 10 AS is \
             real isoform evidence and the read is an anchor"
        );
        assert_eq!(anchors[0].transcript_id, "tx0");
        assert_eq!(s.passed, 1);
        assert_eq!(
            s.rejected_same_gene, 0,
            "rejected_same_gene is a CLASSIFICATION of a gap failure, not a veto; \
             nothing failed the gap here"
        );
        assert_eq!(
            s.rejected_cross_gene, 0,
            "a same-gene pair must not be counted as cross-gene"
        );
        assert_eq!(s.rejected_tie, 0, "AS 100 vs 90 is not a tie");
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "two records were eligible, one became the anchor"
        );
        assert_eq!(record_deficit(&s), 0);
    }

    #[test]
    fn test_a_same_gene_pair_rejected_by_the_gap_is_classified_as_same_gene() {
        let gtf = tx2gene_gtf("samegenegap", "tx0\tGA\ntx1\tGA\ntx2\tGA\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let p = AnchorParams {
            gap_abs: 15.0,
            gap_rel: 0.0,
            ..params()
        };
        let (anchors, s) = run_extract(
            "samegenegap",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
            ],
            &map,
            p,
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 0, "10 AS does not clear a floor of 15");
        assert_eq!(s.rejected_same_gene, 1);
        assert_eq!(s.rejected_cross_gene, 0);
        assert_eq!(s.rejected_tie, 0);
        assert_eq!(
            s.filtered_eligible_not_anchor, 2,
            "both records were eligible and neither was emitted"
        );
        assert_eq!(record_deficit(&s), 0);
    }

    #[test]
    fn test_a_cross_gene_pair_rejected_by_the_gap_is_counted_separately() {
        let gtf = tx2gene_gtf("crossgene", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "crossgene",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
            ],
            &map,
            params_with_gap(15.0, 0.0),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 0);
        assert_eq!(s.rejected_cross_gene, 1, "tx0/GA vs tx1/GB is cross-gene");
        assert_eq!(
            s.rejected_same_gene, 0,
            "the two candidates are in DIFFERENT genes"
        );
        assert_eq!(s.rejected_gap, 0, "the gene relation was known here");
        assert_eq!(s.filtered_eligible_not_anchor, 2);
    }

    #[test]
    fn test_an_exact_argmax_tie_yields_no_anchor() {
        let gtf = tx2gene_gtf("argmaxtie", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "argmaxtie",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 0, "an exact AS tie has no argmax");
        assert_eq!(s.rejected_tie, 1);
        assert_eq!(
            s.rejected_same_gene, 0,
            "the tie is checked first, so the gene relation never decides this"
        );
        assert_eq!(s.rejected_cross_gene, 0);
        assert_eq!(s.passed, 0);
        assert_eq!(s.filtered_eligible_not_anchor, 2);

        let gtf2 = tx2gene_gtf("argmaxtie2", "tx0\tGA\ntx1\tGA\ntx2\tGA\n");
        let map2 = GtfGenes::from_gtf(&gtf2).unwrap();
        let (anchors2, s2) = run_extract(
            "argmaxtie2",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map2,
            params(),
        );
        let _ = std::fs::remove_file(&gtf2);
        assert_eq!(anchors2.len(), 0);
        assert_eq!(s2.rejected_tie, 1);
        assert_eq!(s2.rejected_same_gene, 0, "the tie fired first");
    }

    #[test]
    fn test_a_physically_disqualified_argmax_is_refused_rather_than_emitted() {
        let gtf = tx2gene_gtf("usurp", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "usurp",
            vec![
                synth_record_with_gap("r", 0, 100, true, Some(100), 0, 60),
                synth_record("r", 1, Some(50), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            0,
            "tx1 is the read's runner-up; attributing the read to it would be a \
             different and weaker claim than 'this is the read's transcript'"
        );
        assert_eq!(s.rejected_usurpation, 1);
        assert_eq!(s.passed, 0);
        assert_eq!(
            s.filtered_3p_anchor, 1,
            "the disqualified tx0 record went to its own bucket, not to the \
             read-level rejection"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "tx1 WAS physically eligible and is still not emitted, so its record \
             has to be absorbed here or the record ledger leaks"
        );
        assert_eq!(s.reads_seen, 1);
        assert_eq!(s.rejected_tie, 0, "100 vs 50 is not a tie");
        assert_eq!(s.rejected_cross_gene, 0, "the gap never got to decide");
        assert_eq!(s.rejected_same_gene, 0, "the gap never got to decide");
        assert_eq!(
            s.reads_seen,
            s.rejected_usurpation
                + s.anchors_emitted
                + s.rejected_tie
                + s.rejected_gap
                + s.rejected_same_gene
                + s.rejected_cross_gene
                + s.reads_no_eligible_candidate
                + s.reads_over_cap,
            "the read identity must close with usurpation as a destination"
        );
    }

    #[test]
    fn test_a_raw_top_two_tie_with_one_passing_member_is_a_tie_not_an_usurpation() {
        let gtf = tx2gene_gtf("rawtie", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "rawtie",
            vec![
                synth_record_with_gap("r", 0, 100, true, Some(100), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            0,
            "an exact AS tie has no argmax; emitting either member would invent \
             an isoform assignment the aligner did not make"
        );
        assert_eq!(s.rejected_tie, 1);
        assert_eq!(
            s.rejected_usurpation, 0,
            "THE BRANCH ORDER: a tie must be decided before the winner's \
             eligibility is inspected, or this read's verdict depends on which \
             of the two tied candidates happens to sort first"
        );
        assert_eq!(s.filtered_3p_anchor, 1);
        assert_eq!(s.filtered_eligible_not_anchor, 1);
        assert_eq!(s.passed, 0);
    }

    #[test]
    fn test_the_tie_fixture_differs_from_the_usurpation_fixture_by_one_as_point() {
        let gtf = tx2gene_gtf("tieadj", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (a1, s1) = run_extract(
            "tieadj",
            vec![
                synth_record_with_gap("r", 0, 100, true, Some(101), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        assert_record_ledger_closes(&s1);
        assert_eq!(a1.len(), 0);
        assert_eq!(
            s1.rejected_usurpation, 1,
            "one AS point of separation is the whole difference between 'tie' \
             and 'usurpation'"
        );
        assert_eq!(s1.rejected_tie, 0);

        let (a2, s2) = run_extract(
            "tieadj2",
            vec![
                synth_record_with_gap("r", 0, 100, true, Some(99), 0, 60),
                synth_record("r", 1, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&gtf);
        assert_record_ledger_closes(&s2);
        assert_eq!(a2.len(), 1, "the eligible candidate IS the argmax");
        assert_eq!(a2[0].transcript_id, "tx1");
        assert_eq!(s2.rejected_usurpation, 0);
        assert_eq!(s2.rejected_tie, 0);
    }

    #[test]
    fn test_a_read_with_one_raw_candidate_is_still_measured_against_the_gap_floor() {
        let gtf = tx2gene_gtf("solo_gap", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "solo_gap",
            vec![
                synth_record("under", 0, Some(11), 0, 60),
                synth_record("on", 1, Some(12), 0, 60),
            ],
            &map,
            params_with_gap(12.0, 0.0),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 1, "only the read exactly ON the floor emits");
        assert_eq!(anchors[0].read_id, "on");
        assert_eq!(
            s.rejected_gap, 1,
            "there is no second raw candidate, so there is no gene relation to \
             classify and the gap is the only thing left to name it by"
        );
        assert_eq!(s.rejected_usurpation, 0, "both argmaxes are eligible");
        assert_eq!(s.rejected_tie, 0, "one candidate each");
        assert_eq!(s.filtered_eligible_not_anchor, 1);
    }

    #[test]
    fn test_the_gap_is_measured_against_an_ineligible_raw_runner_up() {
        let gtf = tx2gene_gtf("rawrunnerup", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "rawrunnerup",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record_with_gap("r", 1, 900, false, Some(92), 0, 60),
            ],
            &map,
            params_with_gap(15.0, 0.0),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            0,
            "the read's second-best RAW alignment is 8 AS behind, which is \
             below the floor of 15; a candidate that fails the coverage filter \
             is disqualified, not deleted"
        );
        assert_eq!(s.rejected_cross_gene, 1, "tx0/GA vs tx1/GB");
        assert_eq!(
            s.filtered_coverage, 1,
            "the runner-up is also accounted as the record it is"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "tx0 was eligible and not emitted"
        );
    }

    #[test]
    fn test_the_gap_boundary_is_inclusive_and_exact() {
        let gtf = tx2gene_gtf("gapbound", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "gapbound",
            vec![
                synth_record("narrow", 0, Some(100), 0, 60),
                synth_record("narrow", 1, Some(90), 0, 60),
                synth_record("wide", 0, Some(100), 0, 60),
                synth_record("wide", 1, Some(80), 0, 60),
                synth_record("exact", 0, Some(100), 0, 60),
                synth_record("exact", 1, Some(85), 0, 60),
            ],
            &map,
            params_with_gap(15.0, 0.0),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            2,
            "gap 20 and gap 15 clear a floor of 15; gap 10 does not"
        );
        assert_eq!(
            s.rejected_cross_gene, 1,
            "exactly one cross-gene pair was rejected on the gap"
        );
        assert_eq!(s.passed, 2);
        let ids: Vec<&str> = anchors.iter().map(|a| a.read_id.as_str()).collect();
        assert!(
            ids.contains(&"wide") && ids.contains(&"exact"),
            "the two accepted reads must be the wide and exact ones; got {ids:?}"
        );
        assert!(!ids.contains(&"narrow"), "gap 10 is below the floor of 15");

        let (a2, s2) = run_extract(
            "gaprel",
            vec![
                synth_record("relnarrow", 0, Some(100), 0, 60),
                synth_record("relnarrow", 1, Some(96), 0, 60),
            ],
            &map,
            params_with_gap(0.0, 0.05),
        );
        assert_eq!(a2.len(), 0, "gap 4 is below 5% of AS_best = 100");
        assert_eq!(s2.rejected_cross_gene, 1);
        assert_record_ledger_closes(&s2);
    }

    #[test]
    fn test_two_records_to_the_same_reference_dedup_to_the_max_as() {
        let gtf = tx2gene_gtf("dedup", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "dedup",
            vec![
                synth_record_with_gap("dupe", 0, 100, true, Some(100), 0, 60),
                synth_record("dupe", 0, Some(50), 0, 60),
            ],
            &map,
            params(),
        );
        assert_record_ledger_closes(&s);
        assert_eq!(
            s.filtered_3p_anchor, 2,
            "the max-AS representative failed the 3' filter, and it stands for \
             BOTH collapsed records — one deduped representative, two records"
        );
        assert_eq!(anchors.len(), 0, "no surviving candidate");
        assert_eq!(
            s.filtered_eligible_not_anchor, 0,
            "the records were not ELIGIBLE — they died on a physical filter, so \
             counting them here would misfile them"
        );

        let (anchors2, s2) = run_extract(
            "dedup2",
            vec![
                synth_record("triple", 0, Some(100), 0, 60),
                synth_record("triple", 0, Some(100), 0, 60),
                synth_record("triple", 0, Some(100), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&gtf);
        assert_record_ledger_closes(&s2);
        assert_eq!(
            anchors2.len(),
            1,
            "three records to ONE reference are one candidate, not a tie"
        );
        assert_eq!(s2.rejected_tie, 0);
        assert_eq!(s2.passed, 1);
        assert_eq!(
            s2.filtered_eligible_not_anchor, 2,
            "the two collapsed duplicates were eligible and were not emitted"
        );
        assert_eq!(s2.reads_seen, 1, "three records, ONE read");
    }

    #[test]
    fn test_dedup_groups_records_that_are_not_adjacent_in_the_bam() {
        let gtf = tx2gene_gtf("dedupadj", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "dedupadj",
            vec![
                synth_record_with_gap("split", 0, 100, true, Some(100), 0, 60),
                synth_record("split", 1, Some(50), 0, 60),
                synth_record("split", 0, Some(50), 0, 60),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(
            s.filtered_3p_anchor, 2,
            "the tx0 representative fails the 3' filter and stands for BOTH of \
             the tx0 records, wherever they sit in the BAM"
        );
        assert_eq!(
            anchors.len(),
            0,
            "the read's raw argmax is the TRUNCATED tx0 record, which is not \
             physically eligible; emitting the lower-AS tx1 instead is exactly \
             the runner-up usurpation the C_raw rule exists to refuse"
        );
        assert_eq!(s.rejected_usurpation, 1);
        assert_eq!(
            s.rejected_tie, 0,
            "the two tx0 records are one candidate, so the raw top two are \
             tx0@100 and tx1@50, which is not a tie"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "the one physically-passing record of this read (tx1) is absorbed"
        );
        assert_eq!(s.passed, 0);
    }

    #[test]
    fn test_supplementary_and_unmapped_records_never_become_anchors_even_at_high_mapq() {
        use noodles_sam::alignment::record::Flags;

        let (anchors, s) = run_extract(
            "flagged",
            vec![
                synth_record_with_flags("prim", 0, Flags::empty(), Some(100)),
                synth_record_with_flags("supp", 0, Flags::SUPPLEMENTARY, Some(100)),
                synth_record_with_flags("unmapped", 1, Flags::UNMAPPED, Some(100)),
            ],
            &GtfGenes::new(),
            params(),
        );

        assert_record_ledger_closes(&s);
        assert_eq!(
            anchors.len(),
            1,
            "only the primary alignment may reach the pool"
        );
        assert_eq!(s.filtered_supplementary, 1);
        assert_eq!(s.filtered_unmapped, 1);
        assert_eq!(
            s.filtered_unique, 0,
            "every fixture record here has MAPQ 60; the two flagged records must \
             have been caught by their FLAG, not by the MAPQ gate"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 0,
            "a supplementary record is not 'eligible but not selected' — it was \
             never a molecule"
        );
        assert_eq!(s.passed, 1);
        assert_eq!(
            s.reads_seen, 1,
            "the two flagged records never formed a read"
        );
    }

    #[test]
    fn test_the_record_ledger_closes_while_secondaries_are_admitted() {
        use noodles_sam::alignment::record::Flags;

        let gtf = tx2gene_gtf("secondary", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();
        let (anchors, s) = run_extract(
            "secondary",
            vec![
                synth_record("p0", 0, Some(100), 0, 60),
                synth_record_with_flags("p0", 1, Flags::SECONDARY, Some(40)),
                synth_record_with_flags("s0", 1, Flags::SECONDARY, Some(100)),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(record_deficit(&s), 0);
        assert_eq!(s.total_records, 3);
        assert_eq!(
            anchors.len(),
            2,
            "p0 keeps tx0 on AS and s0 is anchored by a secondary alone"
        );
        assert_eq!(s.passed, 2);
        assert_eq!(
            s.filtered_eligible_not_anchor, 1,
            "the AS-40 secondary lost"
        );
        assert_eq!(
            s.filtered_unique, 0,
            "a secondary with MAPQ 60 must not be counted as low-MAPQ"
        );

        let mut tx: Vec<&str> = anchors.iter().map(|a| a.transcript_id.as_str()).collect();
        tx.sort_unstable();
        assert_eq!(tx, vec!["tx0", "tx1"]);
        let mut ids: Vec<&str> = anchors.iter().map(|a| a.read_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["p0", "s0"]);
    }

    #[test]
    fn test_anchors_emitted_never_exceed_reads_seen_and_the_read_block_closes() {
        use noodles_sam::alignment::record::Flags;
        let gtf = tx2gene_gtf("closure", "tx0\tGA\ntx1\tGB\ntx2\tGA\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();

        let (anchors, s) = run_extract(
            "closure",
            vec![
                synth_record("solo", 0, Some(100), 0, 60),
                synth_record("cross", 0, Some(100), 0, 60),
                synth_record("cross", 1, Some(80), 0, 60),
                synth_record("tie", 0, Some(100), 0, 60),
                synth_record("tie", 1, Some(100), 0, 60),
                synth_record("samegene", 0, Some(100), 0, 60),
                synth_record("samegene", 2, Some(70), 0, 60),
                synth_record_with_gap("dead", 1, 100, true, Some(100), 0, 60),
                synth_record_with_flags("gone", 0, Flags::UNMAPPED, Some(100)),
            ],
            &map,
            params(),
        );
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(
            s.reads_seen, 5,
            "the unmapped record never formed a group, so there are 5 reads"
        );
        assert_eq!(s.anchors_emitted, 3);
        assert_eq!(anchors.len() as u64, s.anchors_emitted);
        assert!(
            s.anchors_emitted <= s.reads_seen,
            "one anchor per read at most: {} anchors from {} reads",
            s.anchors_emitted,
            s.reads_seen
        );
        assert_eq!(
            s.passed, s.anchors_emitted,
            "`passed` means anchors emitted"
        );

        assert_eq!(s.rejected_tie, 1);

        assert_eq!(s.rejected_same_gene, 0);
        assert_eq!(s.rejected_cross_gene, 0);
        assert_eq!(s.gene_veto_disabled, 0);
        assert_eq!(s.reads_no_eligible_candidate, 1);

        assert_eq!(
            s.reads_seen,
            s.anchors_emitted
                + s.rejected_tie
                + s.rejected_usurpation
                + s.rejected_gap
                + s.rejected_same_gene
                + s.rejected_cross_gene
                + s.reads_no_eligible_candidate
                + s.reads_over_cap,
            "every read that reached the per-read stage must land in exactly one \
             outcome; reads_seen={} tie={} usurpation={} gap={} same_gene={} \
             cross_gene={} no_eligible={} over_cap={} anchors={} \
             (gene_veto_disabled={} is NOT a destination: those reads cleared \
             the gap and were emitted)",
            s.reads_seen,
            s.rejected_tie,
            s.rejected_usurpation,
            s.rejected_gap,
            s.rejected_same_gene,
            s.rejected_cross_gene,
            s.gene_veto_disabled,
            s.reads_no_eligible_candidate,
            s.reads_over_cap,
            s.anchors_emitted,
        );

        assert_eq!(
            s.reads_over_cap, 0,
            "no transcript reaches the cap of 200 here, so this fixture cannot \
             witness reads_over_cap -- do not read the identity above as \
             covering it"
        );
    }

    #[test]
    fn test_a_multi_record_read_over_the_cap_absorbs_every_record_it_contributed() {
        let gtf = tx2gene_gtf("capleak", "tx0\tGA\ntx1\tGB\ntx2\tGC\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();
        assert_eq!(map.len(), 3, "a cross-gene map, so the veto cannot fire");

        let mut recs = Vec::new();
        for i in 0..3 {
            recs.push(synth_record(&format!("sat{i}"), 2, Some(100), 0, 60));
            recs.push(synth_record(&format!("sat{i}"), 0, Some(50), 0, 60));
        }

        recs.push(synth_record("lowq", 1, Some(100), 0, 5));
        assert_eq!(recs.len(), 7);

        let mut p = params();
        p.max_reads_per_sample = 2;
        let (anchors, s) = run_extract("capleak", recs, &map, p);
        let _ = std::fs::remove_file(&gtf);

        assert_record_ledger_closes(&s);
        assert_eq!(record_deficit(&s), 0);

        assert_eq!(s.total_records, 7);
        assert_eq!(
            s.reads_seen, 4,
            "three saturating reads plus the low-MAPQ read"
        );

        assert_eq!(s.reads_over_cap, 1, "the third read on tx2 is over the cap");
        assert_eq!(
            s.reads_seen,
            s.anchors_emitted
                + s.rejected_tie
                + s.rejected_usurpation
                + s.rejected_gap
                + s.rejected_same_gene
                + s.rejected_cross_gene
                + s.gene_veto_disabled
                + s.reads_no_eligible_candidate
                + s.reads_over_cap,
            "the cap is a READ-level outcome and has to appear in the read identity"
        );
        assert_eq!(
            s.anchors_emitted, 3,
            "two cap winners plus the low-MAPQ read"
        );
        assert_eq!(s.passed, 3);
        assert_eq!(anchors.len(), 3);

        assert_eq!(
            s.filtered_cap, 1,
            "filtered_cap is READ-denominated: one rejected read"
        );
        assert_eq!(
            s.filtered_cap, s.reads_over_cap,
            "the two names must count the same population"
        );
        assert_eq!(
            s.filtered_eligible_not_anchor, 3,
            "the two emitted reads each lose one argmax loser (2), and the \
             rejected read contributes eligible_records - 1 == 1. Without that \
             last term this is 2 and the record ledger is short by exactly 1."
        );

        assert_eq!(
            s.filtered_unique, 0,
            "MAPQ 5 clears every filter now; the retired slot must stay at 0"
        );
        assert_eq!(s.filtered_unmapped, 0);
        assert_eq!(s.filtered_supplementary, 0);
        assert_eq!(s.filtered_3p_anchor, 0);
        assert_eq!(s.filtered_coverage, 0);
        assert_eq!(s.filtered_polya_invalid, 0);
        assert_eq!(s.rejected_tie, 0, "100 vs 50 is not a tie");
        assert_eq!(
            s.rejected_usurpation, 0,
            "every candidate here is physically eligible, so no argmax is \
             disqualified"
        );
        assert_eq!(s.rejected_same_gene, 0);
        assert_eq!(s.rejected_cross_gene, 0, "the gap floors are off");
        assert_eq!(s.gene_veto_disabled, 0);
        assert_eq!(s.reads_no_eligible_candidate, 0);

        let mut ids: Vec<&str> = anchors.iter().map(|a| a.read_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["lowq", "sat1", "sat2"]);
        let mut txs: Vec<&str> = anchors.iter().map(|a| a.transcript_id.as_str()).collect();
        txs.sort_unstable();
        assert_eq!(txs, vec!["tx1", "tx2", "tx2"]);
    }

    #[test]
    fn test_an_empty_genes_emits_but_records_the_unknown_relation() {
        let (anchors, s) = run_extract(
            "vetooff",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
                synth_record("solo", 2, Some(100), 0, 60),
            ],
            &GtfGenes::new(),
            params(),
        );

        assert_record_ledger_closes(&s);

        assert_eq!(
            anchors.len(),
            2,
            "the pair clears the gap, so it emits; the missing gene map is \
             recorded, not used as grounds for refusal"
        );
        assert_eq!(s.gene_veto_disabled, 1);
        assert_eq!(
            s.same_gene_unknown, 1,
            "the read must be recorded in the UNKNOWN class, not to cross_gene"
        );
        assert_eq!(
            s.rejected_cross_gene, 0,
            "with an empty map the gene relation is unknown, and 'unknown' is \
             not 'cross-gene' — this is the whole point of the counter"
        );
        assert_eq!(
            s.rejected_same_gene, 0,
            "nothing failed the gap, and 'unknown' must not be invented into \
             'same gene' either"
        );
        assert_eq!(s.filtered_eligible_not_anchor, 1);

        assert_eq!(anchors[0].gene_id, anchors[0].transcript_id);
    }

    #[test]
    fn test_an_empty_genes_with_an_unmet_gap_is_rejected_as_rejected_gap() {
        let p = AnchorParams {
            gap_abs: 15.0,
            gap_rel: 0.0,
            ..params()
        };
        let (anchors, s) = run_extract(
            "vetooffgap",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 1, Some(90), 0, 60),
            ],
            &GtfGenes::new(),
            p,
        );

        assert_record_ledger_closes(&s);
        assert_eq!(anchors.len(), 0, "10 AS does not clear a floor of 15");
        assert_eq!(s.rejected_gap, 1, "the Unknown arm of the gap block");
        assert_eq!(s.rejected_same_gene, 0, "no map means never Same");
        assert_eq!(s.rejected_cross_gene, 0, "no map means never Cross");
        assert_eq!(s.gene_veto_disabled, 0, "it never cleared the gap");
        assert_eq!(s.filtered_eligible_not_anchor, 2);
    }

    #[test]
    fn test_a_non_empty_gene_map_that_misses_a_transcript_is_an_error() {
        let gtf = tx2gene_gtf("partial", "tx0\tGA\ntx1\tGB\n");
        let map = GtfGenes::from_gtf(&gtf).unwrap();
        assert_eq!(map.len(), 2, "the map is NOT empty — that is the point");

        let bam = write_synthetic_bam(
            "partial",
            vec![
                synth_record("r", 0, Some(100), 0, 60),
                synth_record("r", 2, Some(90), 0, 60),
            ],
        );
        let err = extract_anchors_from_bam(&bam, "partial", &map, params())
            .expect_err("a non-empty map must not silently cover a missing transcript");
        let _ = std::fs::remove_file(&bam);
        let _ = std::fs::remove_file(&gtf);

        let msg = err.to_string();
        assert!(
            msg.contains("tx2"),
            "the error must name the transcript it could not resolve; got: {msg}"
        );
        assert!(
            msg.contains("Refusing"),
            "the error must say it refused rather than fell back; got: {msg}"
        );
    }

    #[test]
    fn test_report_keeps_the_pinned_prefix_and_appends_the_new_buckets() {
        let agg = AnchorStats {
            samples: 1,
            total_records: 100,
            filtered_unmapped: 12,
            filtered_supplementary: 13,
            filtered_eligible_not_anchor: 10,
            reads_seen: 27,
            anchors_emitted: 7,
            rejected_tie: 9,
            reads_over_cap: 4,
            rejected_usurpation: 6,
            ..Default::default()
        };
        let line = agg.report();

        assert!(
            line.starts_with("samples=1 total=100 filtered(unique=0 no_reflen=0 3p_anchor=0 coverage=0 polya=0 cap=0) passed=0 unaccounted=65 no_refid=0 no_position=0 no_read_id=0"),
            "the pinned prefix changed, which breaks ANCHOR_STATS_RE:\n{line}"
        );
        for token in [
            "unmapped=12",
            "supplementary=13",
            "not_anchor=10",
            "reads(seen=27 anchors=7 tie=9",
            "no_eligible=0 over_cap=4 usurpation=6)",
        ] {
            assert!(
                line.contains(token),
                "report must carry {token}; got: {line}"
            );
        }
        assert!(
            line.contains("reads("),
            "the read block must be labelled `reads` so its unit is not read as \
             the record ledger's unit; got: {line}"
        );
    }
}
