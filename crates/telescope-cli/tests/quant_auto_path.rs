mod common;

use common::{
    read_json, run_manual_chain_with_universe, scratch, stderr_of, synthetic_bam, telescope,
    tiny_gtf,
};
use std::path::{Path, PathBuf};

fn assert_same_bytes(a: &Path, b: &Path, what: &str) {
    let x = std::fs::read(a).unwrap_or_else(|e| panic!("read {}: {e}", a.display()));
    let y = std::fs::read(b).unwrap_or_else(|e| panic!("read {}: {e}", b.display()));
    assert_eq!(
        x.len(),
        y.len(),
        "{what}: {} is {} bytes, {} is {} bytes",
        a.display(),
        x.len(),
        b.display(),
        y.len()
    );
    assert!(
        x == y,
        "{what}: {} and {} are NOT byte-identical",
        a.display(),
        b.display()
    );
}

fn temp_dirs_named_in(stderr: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for line in stderr.lines() {
        if let Some(rest) = line
            .trim()
            .strip_prefix("[telescope]  temporary intermediates: ")
        {
            out.push(PathBuf::from(rest.trim()));
        }
    }
    out
}

#[test]
fn the_single_sample_path_is_the_manual_three_steps_byte_for_byte() {
    let bam = synthetic_bam("t4");
    let gtf = tiny_gtf();
    let out_dir = scratch("t4_auto", "dir");
    let manual_dir = scratch("t4_manual", "dir");
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&manual_dir);
    std::fs::create_dir_all(&out_dir).unwrap();
    let profile = out_dir.join("S1.profile.parquet");

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--gtf",
        gtf.to_str().unwrap(),
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "quant failed: {}", stderr_of(&out));

    let manual = run_manual_chain_with_universe(&manual_dir, &bam, &gtf, "S1");
    assert_eq!(
        common::n_rows_of(&profile),
        6,
        "quant must produce a real profile, not an empty one"
    );
    assert_same_bytes(
        &profile,
        &manual.profiles[0],
        "the profile of quant against the manual three steps",
    );

    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&manual_dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn conflicting_and_missing_quant_flags_fail_cleanly() {
    let bam = synthetic_bam("t7");
    let gtf = tiny_gtf();
    let dir = scratch("t7", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let profile = dir.join("S1.profile.parquet");

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--emission",
        "fake.emission.parquet",
        "--gtf",
        gtf.to_str().unwrap(),
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(stderr_of(&out).contains("mutually exclusive"));

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--gtf",
        gtf.to_str().unwrap(),
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(!out.status.success());

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(stderr_of(&out).contains("--gtf is required"));

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn the_temporary_intermediates_are_removed_after_a_run_and_after_a_failure() {
    let bam = synthetic_bam("t9");
    let gtf = tiny_gtf();
    let dir = scratch("t9", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let profile = dir.join("S1.profile.parquet");

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--gtf",
        gtf.to_str().unwrap(),
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let err = stderr_of(&out);
    for tmp in temp_dirs_named_in(&err) {
        assert!(
            !tmp.exists(),
            "the temporary directory {} must be gone after the run",
            tmp.display()
        );
    }

    let missing = dir.join("no_such.bam");
    let out = telescope(&[
        "quant",
        "--bam",
        missing.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--gtf",
        gtf.to_str().unwrap(),
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(!out.status.success(), "a run over a missing BAM must fail");
    let err = stderr_of(&out);
    for tmp in temp_dirs_named_in(&err) {
        assert!(
            !tmp.exists(),
            "the temporary directory {} must be gone after the FAILED run too",
            tmp.display()
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn the_single_sample_output_mode_writes_one_profile_and_nothing_else() {
    let bam = synthetic_bam("t13");
    let gtf = tiny_gtf();
    let dir = scratch("t13", "dir");
    let manual_dir = scratch("t13_manual", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&manual_dir);
    let profile = dir.join("nested").join("S1.profile.parquet");

    let ledger = dir.join("ledger.json");
    let assign = dir.join("assign.parquet");
    let params = dir.join("params.parquet");

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--gtf",
        gtf.to_str().unwrap(),
        "--output",
        profile.to_str().unwrap(),
        "--ledger",
        ledger.to_str().unwrap(),
        "--read-assignments",
        assign.to_str().unwrap(),
        "--dump-emission-params",
        params.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "the single-sample --output mode must run: {}",
        stderr_of(&out)
    );

    let manual = run_manual_chain_with_universe(&manual_dir, &bam, &gtf, "S1");
    assert!(
        common::n_rows_of(&profile) > 0,
        "the profile must not be empty"
    );
    assert_same_bytes(
        &profile,
        &manual.profiles[0],
        "the profile of the --output mode against the manual chain",
    );

    let mut entries: Vec<String> = std::fs::read_dir(dir.join("nested"))
        .expect("read output dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    assert_eq!(entries, vec!["S1.profile.parquet".to_string()]);
    assert!(ledger.exists(), "hidden --ledger still writes");
    assert!(assign.exists(), "hidden --read-assignments still writes");
    assert!(
        params.exists(),
        "hidden --dump-emission-params still writes"
    );

    let tmps = temp_dirs_named_in(&stderr_of(&out));
    assert!(!tmps.is_empty());
    for t in &tmps {
        assert!(!t.exists(), "{} must be removed", t.display());
    }

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&manual_dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn build_emission_receipt_keeps_expected_fields() {
    let bam = synthetic_bam("t12b");
    let gtf = tiny_gtf();
    let dir = scratch("t12b", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let pool = dir.join("pool.parquet");
    let emission = dir.join("sub.emission.parquet");
    let out = telescope(&[
        "build-anchors",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output",
        pool.to_str().unwrap(),
        "--gtf",
        gtf.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", stderr_of(&out));
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
    assert!(out.status.success(), "{}", stderr_of(&out));
    let sub = read_json(&dir.join("sub.emission.parquet.provenance.json"));
    assert_eq!(
        sub.get("prior_scope").and_then(|v| v.as_str()),
        Some("single_sample")
    );
    assert_eq!(
        sub.get("n_prior_samples").and_then(|v| v.as_num()),
        Some(1.0)
    );
    assert_eq!(
        sub.get("reference_count").and_then(|v| v.as_num()),
        Some(3.0)
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn the_temporary_root_uses_the_portable_precedence() {
    let expected = match std::env::var_os("TMPDIR") {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ if Path::new("/var/tmp").is_dir() => PathBuf::from("/var/tmp"),
        _ => std::env::temp_dir(),
    };
    assert_eq!(telescope::temp_root(), expected);
}

#[test]
fn successful_quant_echoes_the_actual_alignment_weight_once() {
    let bam = synthetic_bam("weight_echo");
    let gtf = tiny_gtf();
    let dir = scratch("weight_echo", "dir");
    let manual = common::run_manual_chain(&dir, &bam, &gtf, "S1");
    for mode in ["automatic", "emission"] {
        for (supplied, expected) in [(None, "20.0"), (Some("7.5"), "7.5"), (Some("7.25"), "7.25")] {
            let profile = dir.join(format!("echo-{mode}-{expected}.parquet"));
            let mut args = vec![
                "quant",
                "--bam",
                bam.to_str().unwrap(),
                "--sample-id",
                "S1",
                "--output",
                profile.to_str().unwrap(),
            ];
            if mode == "emission" {
                args.extend(["--emission", manual.emission.to_str().unwrap()]);
            } else {
                args.extend(["--gtf", gtf.to_str().unwrap()]);
            }
            if let Some(value) = supplied {
                args.extend(["--aln-weight-lambda", value]);
            }
            let out = telescope(&args);
            let err = stderr_of(&out);
            assert!(out.status.success(), "{err}");
            assert!(common::n_rows_of(&profile) > 0);
            let prefix = "[telescope]    aln_weight_lambda: ";
            assert_eq!(
                err.lines().filter(|line| line.starts_with(prefix)).count(),
                1,
                "{err}"
            );
            assert!(
                err.lines()
                    .any(|line| line == format!("{prefix}{expected}")),
                "{err}"
            );
            for forbidden in [
                ["FROZEN", "default"].join(" "),
                ["one-gene", "calibration"].join(" "),
                ["validated", "library-wide"].join(" "),
                ["old", "path"].join(" "),
            ] {
                assert!(
                    !err.to_lowercase().contains(&forbidden.to_lowercase()),
                    "{err}"
                );
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_file(&bam).unwrap();
}

#[test]
fn missing_output_names_the_flag_for_the_input_count() {
    for emission in [false, true] {
        let mut args = vec!["quant", "--bam", "missing.bam", "--sample-id", "S1"];
        if emission {
            args.extend(["--emission", "missing.parquet"]);
        } else {
            args.extend(["--gtf", "missing.gtf"]);
        }
        let out = telescope(&args);
        let err = stderr_of(&out);
        assert!(!out.status.success());
        assert!(err.contains("--output"), "{err}");
        assert!(!err.contains("--output-dir"), "{err}");
    }
    let out = telescope(&[
        "quant",
        "--bam",
        "one.bam",
        "--bam",
        "two.bam",
        "--sample-id",
        "S1",
        "--gtf",
        "missing.gtf",
        "--output",
        "out.parquet",
    ]);
    assert!(!out.status.success());
}

#[test]
fn temporary_roots_are_selected_per_process_and_cleaned_on_success_and_failure() {
    let bam = synthetic_bam("portable_tmp");
    let gtf = tiny_gtf();
    let dir = scratch("portable_tmp", "dir");
    let custom = dir.join("custom-root");
    std::fs::create_dir_all(&custom).unwrap();
    for value in [None, Some(""), Some(custom.to_str().unwrap())] {
        for fail in [false, true] {
            let output = dir.join("output.parquet");
            let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_telescope"));
            cmd.args([
                "quant",
                "--bam",
                bam.to_str().unwrap(),
                "--sample-id",
                "S1",
                "--gtf",
                if fail {
                    "missing.gtf"
                } else {
                    gtf.to_str().unwrap()
                },
                "--output",
                output.to_str().unwrap(),
            ]);
            match value {
                Some(v) => {
                    cmd.env("TMPDIR", v);
                }
                None => {
                    cmd.env_remove("TMPDIR");
                }
            }
            let out = cmd.output().unwrap();
            let err = stderr_of(&out);
            assert_eq!(out.status.success(), !fail, "{err}");
            let dirs = temp_dirs_named_in(&err);
            assert!(!dirs.is_empty(), "{err}");
            let expected = match value {
                Some(v) if !v.is_empty() => custom.clone(),
                _ if Path::new("/var/tmp").is_dir() => PathBuf::from("/var/tmp"),
                _ => std::env::temp_dir(),
            };
            for tmp in dirs {
                assert!(
                    tmp.starts_with(&expected),
                    "{} under {}",
                    tmp.display(),
                    expected.display()
                );
                assert!(!tmp.exists(), "{}", tmp.display());
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_file(&bam).unwrap();
}
