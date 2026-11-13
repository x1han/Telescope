#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use telescope_core::evidence::test_support::{synth_record, write_synthetic_bam_named};

pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn tiny_gtf() -> PathBuf {
    fixtures_dir().join("tiny.gtf")
}

pub fn scratch(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "telescope_cli_{}_{}.{}",
        tag,
        std::process::id(),
        ext
    ))
}

pub fn synthetic_bam(tag: &str) -> PathBuf {
    let recs = vec![
        synth_record("r1", 0, Some(100), 0, 60),
        synth_record("r2", 0, Some(100), 0, 60),
        synth_record("r3", 1, Some(100), 0, 60),
        synth_record("r4", 2, Some(100), 0, 60),
        synth_record("r4", 0, Some(95), 0, 60),
        synth_record("r5", 0, Some(100), 0, 60),
        synth_record("r5", 1, Some(80), 0, 60),
    ];
    write_synthetic_bam_named(tag, &["tx0", "tx1", "tx2"], recs)
}

pub fn synthetic_bam_second(tag: &str) -> PathBuf {
    let recs = vec![
        synth_record("q1", 1, Some(120), 0, 60),
        synth_record("q2", 2, Some(120), 0, 60),
        synth_record("q2", 0, Some(115), 0, 60),
        synth_record("q3", 0, Some(120), 0, 60),
        synth_record("q3", 1, Some(100), 0, 60),
    ];
    write_synthetic_bam_named(tag, &["tx0", "tx1", "tx2"], recs)
}

pub fn synthetic_bam_other_universe(tag: &str) -> PathBuf {
    let recs = vec![
        synth_record("u1", 0, Some(140), 0, 60),
        synth_record("u2", 1, Some(140), 0, 60),
        synth_record("u2", 0, Some(135), 0, 60),
    ];
    write_synthetic_bam_named(tag, &["tx0", "tx1", "tx3"], recs)
}

pub fn telescope(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args(args)
        .output()
        .expect("failed to spawn the telescope binary")
}

pub fn run_auto(
    dir: &Path,
    bams: &[PathBuf],
    sample_ids: &[&str],
    gtf: &Path,
    extra: &[&str],
) -> Output {
    std::fs::create_dir_all(dir).expect("mkdir");
    let mut args: Vec<String> = vec!["quant".to_string()];
    for b in bams {
        args.push("--bam".to_string());
        args.push(b.to_str().unwrap().to_string());
    }
    for s in sample_ids {
        args.push("--sample-id".to_string());
        args.push(s.to_string());
    }
    args.push("--gtf".to_string());
    args.push(gtf.to_str().unwrap().to_string());
    args.push("--output-dir".to_string());
    args.push(dir.to_str().unwrap().to_string());
    for e in extra {
        args.push((*e).to_string());
    }
    let argv: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    telescope(&argv)
}

pub fn temp_entries(prefix: &str) -> Vec<PathBuf> {
    let root = telescope::temp_root();
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&root) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with(prefix) {
                out.push(e.path());
            }
        }
    }
    out
}

pub fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

pub struct ManualChain {
    pub pool: PathBuf,
    pub emission: PathBuf,

    pub profiles: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn keys(&self) -> Vec<String> {
        match self {
            Json::Obj(m) => m.iter().map(|(k, _)| k.clone()).collect(),
            _ => Vec::new(),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_num(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

pub fn parse_json(s: &str) -> Result<Json, String> {
    let mut p = JsonParser {
        b: s.as_bytes(),
        i: 0,
    };
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing content at byte {}", p.i));
    }
    Ok(v)
}

pub fn read_json(path: &Path) -> Json {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    parse_json(&text).unwrap_or_else(|e| panic!("{} is NOT valid JSON: {e}", path.display()))
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.peek() {
            None => Err("unexpected end of input".to_string()),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => {
                self.lit("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.lit("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.lit("null")?;
                Ok(Json::Null)
            }
            Some(_) => self.number(),
        }
    }

    fn lit(&mut self, word: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(())
        } else {
            Err(format!("expected {word} at byte {}", self.i))
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1;
        let mut fields: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected a field name at byte {}", self.i));
            }
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at byte {}", self.i));
            }
            self.i += 1;
            let value = self.value()?;
            fields.push((key, value));
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(fields));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1;
        let mut items: Vec<Json> = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let v = self.value()?;
            items.push(v);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| "unterminated string".to_string())?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self
                        .peek()
                        .ok_or_else(|| "unterminated escape".to_string())?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = std::str::from_utf8(&self.b[self.i..self.i + 4])
                                .map_err(|_| "bad \\u escape".to_string())?;
                            let cp = u32::from_str_radix(hex, 16)
                                .map_err(|_| "bad \\u escape".to_string())?;
                            self.i += 4;
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(format!("unknown escape at byte {}", self.i)),
                    }
                }
                _ => {
                    let start = self.i - 1;
                    let mut end = self.i;
                    while end < self.b.len() && (self.b[end] & 0xC0) == 0x80 {
                        end += 1;
                    }
                    self.i = end;
                    out.push_str(
                        std::str::from_utf8(&self.b[start..end])
                            .map_err(|_| "invalid UTF-8 in a string".to_string())?,
                    );
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.i += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).unwrap_or("");
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("not a number: {text:?}"))
    }
}

pub fn run_manual_chain(dir: &Path, bam: &Path, gtf: &Path, sample_id: &str) -> ManualChain {
    std::fs::create_dir_all(dir).expect("mkdir");
    let pool = dir.join("anchor_pool.parquet");
    let emission = dir.join("telescope.emission.parquet");
    let profile = dir.join(format!("{}.profile.parquet", sample_id));

    let out = telescope(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        sample_id,
        "--output",
        pool.to_str().unwrap(),
        "--gtf",
        gtf.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "build-anchors failed: {}",
        stderr_of(&out)
    );

    let out = telescope(&[
        "build-emission",
        "--anchor-pool",
        pool.to_str().unwrap(),
        "--output",
        emission.to_str().unwrap(),
        "--gtf",
        gtf.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "build-emission failed: {}",
        stderr_of(&out)
    );

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--emission",
        emission.to_str().unwrap(),
        "--sample-id",
        sample_id,
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "quant failed: {}", stderr_of(&out));

    ManualChain {
        pool,
        emission,
        profiles: vec![profile],
    }
}

pub fn run_manual_chain_with_universe(
    dir: &Path,
    bam: &Path,
    gtf: &Path,
    sample_id: &str,
) -> ManualChain {
    std::fs::create_dir_all(dir).expect("mkdir");
    let pool = dir.join("anchor_pool.parquet");
    let emission = dir.join("telescope.emission.parquet");
    let profile = dir.join(format!("{}.profile.parquet", sample_id));

    let out = telescope(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        sample_id,
        "--output",
        pool.to_str().unwrap(),
        "--gtf",
        gtf.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "build-anchors failed: {}",
        stderr_of(&out)
    );

    let out = telescope(&[
        "build-emission",
        "--anchor-pool",
        pool.to_str().unwrap(),
        "--output",
        emission.to_str().unwrap(),
        "--gtf",
        gtf.to_str().unwrap(),
        "--reference-bam",
        bam.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "build-emission failed: {}",
        stderr_of(&out)
    );

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--emission",
        emission.to_str().unwrap(),
        "--sample-id",
        sample_id,
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "quant failed: {}", stderr_of(&out));

    ManualChain {
        pool,
        emission,
        profiles: vec![profile],
    }
}

pub fn run_manual_cohort_chain(
    dir: &Path,
    bams: &[PathBuf],
    sample_ids: &[&str],
    gtf: &Path,
) -> ManualChain {
    std::fs::create_dir_all(dir).expect("mkdir");
    let pool = dir.join("anchor_pool.parquet");
    let emission = dir.join("telescope.emission.parquet");

    let mut args: Vec<String> = vec!["build-anchors".to_string()];
    for b in bams {
        args.push("--bam".to_string());
        args.push(b.to_str().unwrap().to_string());
    }
    for s in sample_ids {
        args.push("--sample-id".to_string());
        args.push(s.to_string());
    }
    args.push("--output".to_string());
    args.push(pool.to_str().unwrap().to_string());
    args.push("--gtf".to_string());
    args.push(gtf.to_str().unwrap().to_string());
    let argv: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let out = telescope(&argv);
    assert!(
        out.status.success(),
        "build-anchors failed: {}",
        stderr_of(&out)
    );

    let out = telescope(&[
        "build-emission",
        "--anchor-pool",
        pool.to_str().unwrap(),
        "--output",
        emission.to_str().unwrap(),
        "--gtf",
        gtf.to_str().unwrap(),
        "--reference-bam",
        bams[0].to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "build-emission failed: {}",
        stderr_of(&out)
    );

    let mut profiles = Vec::new();
    for (bam, sid) in bams.iter().zip(sample_ids.iter()) {
        let profile = dir.join(format!("{sid}.profile.parquet"));
        let out = telescope(&[
            "quant",
            "--bam",
            bam.to_str().unwrap(),
            "--emission",
            emission.to_str().unwrap(),
            "--sample-id",
            sid,
            "--output",
            profile.to_str().unwrap(),
        ]);
        assert!(out.status.success(), "quant failed: {}", stderr_of(&out));
        profiles.push(profile);
    }

    ManualChain {
        pool,
        emission,
        profiles,
    }
}

pub fn n_rows_of(path: &Path) -> usize {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(f)
        .expect("the file is a parquet table")
        .build()
        .expect("build the parquet reader");
    reader
        .map(|b| b.expect("read a batch").num_rows())
        .sum::<usize>()
}
