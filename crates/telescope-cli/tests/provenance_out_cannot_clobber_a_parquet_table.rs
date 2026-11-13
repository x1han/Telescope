use std::path::{Path, PathBuf};
use std::process::Command;

use telescope_core::evidence::test_support::{synth_record, write_synthetic_bam_named};

fn anchor_bam(tag: &str) -> PathBuf {
    write_synthetic_bam_named(
        &format!("{tag}_pool"),
        &["tx0", "tx1"],
        vec![
            synth_record("a0", 0, Some(100), 0, 60),
            synth_record("a1", 0, Some(100), 0, 60),
            synth_record("b0", 1, Some(100), 0, 60),
        ],
    )
}

fn scratch(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("telescope_cli_{tag}_{}.{ext}", std::process::id()))
}

fn cleanup(paths: &[&Path]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

fn build_emission_with_prov(pool: &Path, out: &Path, prov: &Path) -> (bool, String) {
    let st = Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args([
            "build-emission",
            "--anchor-pool",
            pool.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--provenance-out",
            prov.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn the telescope binary");
    (
        st.status.success(),
        String::from_utf8_lossy(&st.stderr).into_owned(),
    )
}

fn prepared(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let bam = anchor_bam(tag);
    let pool = scratch(&format!("{tag}_pool"), "parquet");
    let emission = scratch(&format!("{tag}_emission"), "parquet");
    let tiers = emission.with_extension("provenance.parquet");
    let header = emission.with_extension("provenance.json");
    for p in [&pool, &emission, &tiers, &header] {
        let _ = std::fs::remove_file(p);
    }
    let st = Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args([
            "build-anchors",
            "--bam",
            bam.to_str().unwrap(),
            "--sample-id",
            "S1",
            "--output",
            pool.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn the telescope binary");
    assert!(
        st.status.success(),
        "build-anchors failed: {}\nstderr:\n{}",
        st.status,
        String::from_utf8_lossy(&st.stderr)
    );
    let st = Command::new(env!("CARGO_BIN_EXE_telescope"))
        .args([
            "build-emission",
            "--anchor-pool",
            pool.to_str().unwrap(),
            "--output",
            emission.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn the telescope binary");
    assert!(
        st.status.success(),
        "build-emission failed: {}\nstderr:\n{}",
        st.status,
        String::from_utf8_lossy(&st.stderr)
    );
    assert_eq!(
        magic(&emission),
        Some(*b"PAR1"),
        "CONTROL: the seed emission must be a real parquet file"
    );
    (bam, pool, emission)
}

fn magic(path: &Path) -> Option<[u8; 4]> {
    let b = std::fs::read(path).ok()?;
    Some([b[0], b[1], b[2], b[3]])
}

#[test]
fn a_provenance_out_ending_in_parquet_is_refused_and_writes_nothing() {
    let (bam, pool, emission) = prepared("provparquet");

    let other = scratch("provparquet_other", "parquet");
    let _ = std::fs::remove_file(&other);
    std::fs::copy(&emission, &other).expect("seed a parquet at the offending path");

    let (ok, stderr) = build_emission_with_prov(&pool, &emission, &other);

    let head = magic(&other);
    let len_after = std::fs::metadata(&other).map(|m| m.len()).unwrap_or(0);
    cleanup(&[&bam, &pool, &emission, &other]);

    assert!(
        !ok,
        "CONTROL-GUARD: `--provenance-out <...>.parquet` MUST fail. Exiting 0 is \
         how 654 bytes of JSON replaced a per-transcript tier table.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("ends in .parquet"),
        "the error must name the extension that caused it; a bare non-zero exit \
         would be indistinguishable from any other failure.\nstderr:\n{stderr}"
    );
    assert_eq!(
        head,
        Some(*b"PAR1"),
        "the offending path must be left as PARQUET: the guard refuses BEFORE \
         any write, it does not write then complain. len={len_after}"
    );
}

#[test]
fn a_provenance_out_equal_to_output_is_refused() {
    let (bam, pool, emission) = prepared("provsame");
    let bytes_before = std::fs::metadata(&emission).unwrap().len();

    let (ok, stderr) = build_emission_with_prov(&pool, &emission, &emission);

    let bytes_after = std::fs::metadata(&emission).map(|m| m.len()).unwrap_or(0);
    cleanup(&[&bam, &pool, &emission]);

    assert!(
        !ok,
        "CONTROL-GUARD: `--provenance-out` == `--output` MUST fail.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("same file as"),
        "the error must say the two paths collide.\nstderr:\n{stderr}"
    );
    assert_eq!(
        bytes_after, bytes_before,
        "the emission table must be left byte-identical"
    );
}

#[test]
fn a_provenance_out_named_at_the_tier_sidecar_is_refused_and_the_sidecar_survives() {
    let (bam, pool, emission) = prepared("provtiers");
    let tiers = emission.with_extension("provenance.parquet");

    let (ok, stderr) = build_emission_with_prov(&pool, &emission, &tiers);

    let head = magic(&tiers);
    cleanup(&[&bam, &pool, &emission, &tiers]);

    assert!(
        !ok,
        "CONTROL-GUARD: naming the per-transcript tier table as --provenance-out \
         MUST fail.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("--provenance-out"),
        "the error must name the offending flag, or a non-zero exit here is \
         indistinguishable from any other failure.\nstderr:\n{stderr}"
    );
    assert_eq!(
        head,
        Some(*b"PAR1"),
        "the per-transcript tier table must remain Parquet after the rejected write."
    );
}

#[cfg(unix)]
#[test]
fn a_provenance_out_behind_a_symlink_at_the_tier_sidecar_hits_the_collision_guard() {
    use std::os::unix::fs::symlink;

    let (bam, pool, emission) = prepared("provsym");
    let tiers = emission.with_extension("provenance.parquet");
    let header = emission.with_extension("provenance.json");
    let _ = std::fs::remove_file(&tiers);
    let _ = std::fs::remove_file(&header);
    symlink(&header, &tiers).expect("plant a symlink at the tier sidecar path");

    let (ok, stderr) = build_emission_with_prov(&pool, &emission, &header);

    let head = magic(&header);
    cleanup(&[&bam, &pool, &emission, &tiers, &header]);

    assert!(
        !ok,
        "CONTROL-GUARD: a --provenance-out that canonicalises onto the tier \
         sidecar MUST fail.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("same file as"),
        "the COLLISION branch must be the one that fires here, which is the \
         whole claim: a symlink at the sidecar path makes the second call in \
         `build-emission` reachable, and `refuse_a_provenance_path_that_\
         clobbers_a_table` checks collision before extension. If this says \
         \"ends in .parquet\" instead, the ordering changed and this fixture no \
         longer exercises what it says it does.\nstderr:\n{stderr}"
    );
    assert_eq!(
        head,
        Some(*b"PAR1"),
        "the JSON run header must never have been written over the file the \
         tier sidecar resolves to. A leading brace here would mean the guard \
         was bypassed."
    );
}

#[test]
fn a_provenance_json_path_is_accepted() {
    let (bam, pool, emission) = prepared("provok");
    let prov = scratch("provok_header", "json");
    let _ = std::fs::remove_file(&prov);

    let (ok, stderr) = build_emission_with_prov(&pool, &emission, &prov);

    let wrote = std::fs::metadata(&prov).map(|m| m.len()).unwrap_or(0);
    cleanup(&[
        &bam,
        &pool,
        &emission,
        &prov,
        &emission.with_extension("provenance.parquet"),
    ]);

    assert!(
        ok,
        "CONTROL: a `.json` --provenance-out is the documented use and must \
         still succeed.\nstderr:\n{stderr}"
    );
    assert!(
        wrote > 0,
        "the run header must actually be written: {wrote} bytes"
    );
}
