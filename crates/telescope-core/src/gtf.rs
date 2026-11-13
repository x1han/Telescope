use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneOf {
    pub gene_id: String,

    pub gene_name: String,
}

#[derive(Debug, Default, Clone)]
pub struct GtfGenes {
    map: HashMap<String, GeneOf>,
}

impl GtfGenes {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn from_gtf<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open the annotation GTF: {:?}", path))?;
        let reader = BufReader::new(file);
        let mut map: HashMap<String, GeneOf> = HashMap::new();
        let mut line_no = 0usize;
        for line in reader.lines() {
            line_no += 1;
            let line = line.with_context(|| {
                format!(
                    "Failed to read the annotation GTF: {:?} (line {})",
                    path, line_no
                )
            })?;
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 9 {
                bail!(
                    "the annotation GTF has a malformed line: {:?} (line {} has {} field(s), \
                     a GTF line has 9)",
                    line,
                    line_no,
                    fields.len()
                );
            }
            let attrs = parse_attributes(fields[8]);
            let transcript_id = match attrs.get("transcript_id") {
                Some(t) if !t.is_empty() => t.clone(),
                _ => continue,
            };
            let gene_id = match attrs.get("gene_id") {
                Some(g) if !g.is_empty() => g.clone(),

                _ => continue,
            };
            let gene_name = attrs
                .get("gene_name")
                .filter(|n| !n.is_empty())
                .cloned()
                .unwrap_or_else(|| gene_id.clone());
            match map.get(&transcript_id) {
                Some(seen) if seen.gene_id != gene_id => {
                    bail!(
                        "the annotation GTF gives the transcript {:?} two genes: {:?} and {:?}. \
                         A transcript belongs to one gene.",
                        transcript_id,
                        seen.gene_id,
                        gene_id
                    );
                }
                Some(_) => {}
                None => {
                    map.insert(transcript_id, GeneOf { gene_id, gene_name });
                }
            }
        }
        if map.is_empty() {
            bail!(
                "the annotation GTF carries no transcript with a gene id: {:?}. \
                 A map with no entry in it is not an annotation.",
                path
            );
        }
        Ok(Self { map })
    }

    pub fn gene_id(&self, transcript_id: &str) -> Option<&String> {
        self.map.get(transcript_id).map(|g| &g.gene_id)
    }

    pub fn gene_name(&self, transcript_id: &str) -> Option<&String> {
        self.map.get(transcript_id).map(|g| &g.gene_name)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn gene_name_by_gene(&self) -> BTreeMap<String, String> {
        let mut out: BTreeMap<String, String> = BTreeMap::new();
        for g in self.map.values() {
            out.entry(g.gene_id.clone())
                .or_insert_with(|| g.gene_name.clone());
        }
        out
    }

    pub fn gene_map(&self) -> HashMap<String, String> {
        self.map
            .iter()
            .map(|(tx, g)| (tx.clone(), g.gene_id.clone()))
            .collect()
    }

    pub fn gene_name_map(&self) -> HashMap<String, String> {
        self.map
            .iter()
            .map(|(tx, g)| (tx.clone(), g.gene_name.clone()))
            .collect()
    }
}

fn parse_attributes(field: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for part in field.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (key, value) = match part.find(char::is_whitespace) {
            Some(i) => (&part[..i], part[i..].trim()),
            None => (part, ""),
        };
        if key.is_empty() {
            continue;
        }
        let value = value.trim_matches('"').trim();
        out.insert(key.to_string(), value.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_gtf(body: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "telescope_gtf_test_{}_{}.gtf",
            std::process::id(),
            body.len()
        ));
        let mut f = File::create(&p).expect("create");
        f.write_all(body.as_bytes()).expect("write");
        p
    }

    #[test]
    fn the_transcripts_gene_and_name_are_read() {
        let p = write_gtf(
            "chr1\tHAVANA\tgene\t1\t100\t.\t+\t.\tgene_id \"G1\"; gene_name \"ALPHA\";\n\
             chr1\tHAVANA\ttranscript\t1\t100\t.\t+\t.\tgene_id \"G1\"; transcript_id \"T1\"; gene_name \"ALPHA\";\n\
             chr1\tHAVANA\texon\t1\t50\t.\t+\t.\tgene_id \"G1\"; transcript_id \"T1\"; gene_name \"ALPHA\"; exon_number 1;\n\
             chr1\tHAVANA\ttranscript\t1\t100\t.\t+\t.\tgene_id \"G2\"; transcript_id \"T2\"; gene_name \"BETA\";\n",
        );
        let genes = GtfGenes::from_gtf(&p).expect("read");
        assert_eq!(genes.len(), 2);
        assert_eq!(genes.gene_id("T1").map(|s| s.as_str()), Some("G1"));
        assert_eq!(genes.gene_name("T1").map(|s| s.as_str()), Some("ALPHA"));
        assert_eq!(genes.gene_id("T2").map(|s| s.as_str()), Some("G2"));
        assert_eq!(genes.gene_name("T2").map(|s| s.as_str()), Some("BETA"));
        assert!(genes.gene_id("T9").is_none());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn a_transcript_without_a_name_falls_back_to_its_gene_id() {
        let p = write_gtf(
            "chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G1\"; transcript_id \"T1\";\n",
        );
        let genes = GtfGenes::from_gtf(&p).expect("read");
        assert_eq!(genes.gene_name("T1").map(|s| s.as_str()), Some("G1"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn a_transcript_with_two_genes_is_an_error() {
        let p = write_gtf(
            "chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G1\"; transcript_id \"T1\";\n\
             chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G2\"; transcript_id \"T1\";\n",
        );
        let err = GtfGenes::from_gtf(&p).expect_err("two genes for one transcript");
        assert!(
            format!("{:#}", err).contains("two genes"),
            "the error names the contradiction: {:#}",
            err
        );
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn a_malformed_line_is_an_error_not_a_silent_skip() {
        let p = write_gtf("chr1\tX\ttranscript\t1\t10\t.\t+\t.\n");
        let err = GtfGenes::from_gtf(&p).expect_err("a line with 8 fields");
        assert!(
            format!("{:#}", err).contains("malformed"),
            "the error says the line is malformed: {:#}",
            err
        );
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn an_annotation_with_no_transcript_is_an_error() {
        let p = write_gtf("chr1\tX\tgene\t1\t10\t.\t+\t.\tgene_id \"G1\"; gene_name \"A\";\n");
        let err = GtfGenes::from_gtf(&p).expect_err("no transcript at all");
        assert!(
            format!("{:#}", err).contains("no transcript"),
            "the error says the annotation carries no transcript: {:#}",
            err
        );
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let p = write_gtf(
            "#a comment\n\
             \n\
             chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G1\"; transcript_id \"T1\";\n",
        );
        let genes = GtfGenes::from_gtf(&p).expect("read");
        assert_eq!(genes.len(), 1);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn a_bare_value_with_spaces_is_kept_whole() {
        let attrs = parse_attributes("gene_id \"G1\"; gene_name \"SECRETARY PROTEIN 1\"; x 1;");
        assert_eq!(
            attrs.get("gene_name").map(|s| s.as_str()),
            Some("SECRETARY PROTEIN 1")
        );
        assert_eq!(attrs.get("x").map(|s| s.as_str()), Some("1"));
    }

    #[test]
    fn the_map_can_be_taken_as_ids_or_as_names() {
        let p = write_gtf(
            "chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G1\"; transcript_id \"T1\"; gene_name \"ALPHA\";\n\
             chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G2\"; transcript_id \"T2\"; gene_name \"BETA\";\n",
        );
        let genes = GtfGenes::from_gtf(&p).expect("read");
        let ids = genes.gene_map();
        let names = genes.gene_name_map();
        assert_eq!(ids.get("T1").map(|s| s.as_str()), Some("G1"));
        assert_eq!(ids.get("T2").map(|s| s.as_str()), Some("G2"));
        assert_eq!(names.get("T1").map(|s| s.as_str()), Some("ALPHA"));
        assert_eq!(names.get("T2").map(|s| s.as_str()), Some("BETA"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn the_name_of_a_gene_is_found_through_its_transcripts() {
        let p = write_gtf(
            "chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G1\"; transcript_id \"T1\"; gene_name \"ALPHA\";\n\
             chr1\tX\ttranscript\t1\t10\t.\t+\t.\tgene_id \"G1\"; transcript_id \"T2\"; gene_name \"ALPHA\";\n",
        );
        let genes = GtfGenes::from_gtf(&p).expect("read");
        let by_gene = genes.gene_name_by_gene();
        assert_eq!(by_gene.get("G1").map(|s| s.as_str()), Some("ALPHA"));
        assert!(!by_gene.contains_key("G9"));
        std::fs::remove_file(&p).ok();
    }
}
