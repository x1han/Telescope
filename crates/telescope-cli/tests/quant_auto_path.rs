mod common;

use common::{
    read_json, run_auto, run_manual_chain_with_universe, run_manual_cohort_chain, scratch,
    stderr_of, synthetic_bam, synthetic_bam_other_universe, synthetic_bam_second, telescope,
    tiny_gtf, Json,
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

fn receipt_of(output_dir: &Path) -> Json {
    let p = output_dir
        .join("emission")
        .join("telescope.emission.parquet.provenance.json");
    read_json(&p)
}

#[test]
fn the_cohort_path_is_the_manual_three_steps_byte_for_byte() {
    let bam1 = synthetic_bam("t3_a");
    let bam2 = synthetic_bam_second("t3_b");
    let gtf = tiny_gtf();
    let auto_dir = scratch("t3_auto", "dir");
    let manual_dir = scratch("t3_manual", "dir");
    let _ = std::fs::remove_dir_all(&auto_dir);
    let _ = std::fs::remove_dir_all(&manual_dir);

    let out = run_auto(
        &auto_dir,
        &[bam1.clone(), bam2.clone()],
        &["S1", "S2"],
        &gtf,
        &[],
    );
    assert!(
        out.status.success(),
        "the automatic mode failed: {}",
        stderr_of(&out)
    );

    let manual = run_manual_cohort_chain(
        &manual_dir,
        &[bam1.clone(), bam2.clone()],
        &["S1", "S2"],
        &gtf,
    );

    for (sid, idx, n_rows) in [("S1", 0usize, 6usize), ("S2", 1usize, 4usize)] {
        let auto_profile = auto_dir.join(format!("{sid}.profile.parquet"));
        let manual_profile = &manual.profiles[idx];

        assert_eq!(
            common::n_rows_of(&auto_profile),
            n_rows,
            "the automatic mode must produce a real profile, not an empty one"
        );
        assert_same_bytes(
            &auto_profile,
            manual_profile,
            &format!("the {sid} profile of the cohort path"),
        );
    }

    let _ = std::fs::remove_dir_all(&auto_dir);
    let _ = std::fs::remove_dir_all(&manual_dir);
    let _ = std::fs::remove_file(&bam1);
    let _ = std::fs::remove_file(&bam2);
}

#[test]
fn the_single_sample_path_is_the_manual_three_steps_byte_for_byte() {
    let bam = synthetic_bam("t4");
    let gtf = tiny_gtf();
    let auto_dir = scratch("t4_auto", "dir");
    let manual_dir = scratch("t4_manual", "dir");
    let _ = std::fs::remove_dir_all(&auto_dir);
    let _ = std::fs::remove_dir_all(&manual_dir);

    let out = run_auto(&auto_dir, &[bam.clone()], &["S1"], &gtf, &[]);
    assert!(
        out.status.success(),
        "the automatic mode failed: {}",
        stderr_of(&out)
    );

    let manual = run_manual_chain_with_universe(&manual_dir, &bam, &gtf, "S1");
    let auto_profile = auto_dir.join("S1.profile.parquet");
    assert_eq!(
        common::n_rows_of(&auto_profile),
        6,
        "the automatic mode must produce a real profile, not an empty one"
    );
    assert_same_bytes(
        &auto_profile,
        &manual.profiles[0],
        "the profile of the single-sample path",
    );

    let _ = std::fs::remove_dir_all(&auto_dir);
    let _ = std::fs::remove_dir_all(&manual_dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn the_receipt_states_the_scope_and_the_samples_the_prior_used() {
    let bam1 = synthetic_bam("t5_a");
    let bam2 = synthetic_bam_second("t5_b");
    let gtf = tiny_gtf();

    let dir = scratch("t5_one", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    let out = run_auto(&dir, &[bam1.clone()], &["S1"], &gtf, &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let r = receipt_of(&dir);
    assert_eq!(
        r.get("prior_scope").and_then(|v| v.as_str()),
        Some("single_sample"),
        "one BAM under `auto` is a single-sample prior: {r:?}"
    );
    assert_eq!(
        r.get("n_prior_samples").and_then(|v| v.as_num()),
        Some(1.0),
        "the prior of one BAM used one sample: {r:?}"
    );

    let dir2 = scratch("t5_two", "dir");
    let _ = std::fs::remove_dir_all(&dir2);
    let out = run_auto(
        &dir2,
        &[bam1.clone(), bam2.clone()],
        &["S1", "S2"],
        &gtf,
        &[],
    );
    assert!(out.status.success(), "{}", stderr_of(&out));
    let r = receipt_of(&dir2);
    assert_eq!(
        r.get("prior_scope").and_then(|v| v.as_str()),
        Some("cohort"),
        "two BAMs under `auto` are a cohort prior: {r:?}"
    );
    assert_eq!(
        r.get("n_prior_samples").and_then(|v| v.as_num()),
        Some(2.0),
        "the cohort prior used BOTH samples: {r:?}"
    );
    let ids: Vec<String> = r
        .get("prior_sample_ids")
        .and_then(|v| v.as_arr())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(
        ids,
        vec!["S1".to_string(), "S2".to_string()],
        "the receipt must name the samples the prior used: {r:?}"
    );

    let dir3 = scratch("t5_cohort1", "dir");
    let _ = std::fs::remove_dir_all(&dir3);
    let out = run_auto(
        &dir3,
        &[bam1.clone()],
        &["S1"],
        &gtf,
        &["--prior-scope", "cohort"],
    );
    assert!(out.status.success(), "{}", stderr_of(&out));
    let r = receipt_of(&dir3);
    assert_eq!(
        r.get("prior_scope").and_then(|v| v.as_str()),
        Some("cohort"),
        "--prior-scope cohort must be honoured even for one BAM: {r:?}"
    );
    assert_eq!(
        r.get("n_prior_samples").and_then(|v| v.as_num()),
        Some(1.0),
        "a cohort prior of ONE BAM used one sample: {r:?}"
    );

    let dir4 = scratch("t5_sample2", "dir");
    let _ = std::fs::remove_dir_all(&dir4);
    let out = run_auto(
        &dir4,
        &[bam1.clone(), bam2.clone()],
        &["S1", "S2"],
        &gtf,
        &["--prior-scope", "sample"],
    );
    assert!(out.status.success(), "{}", stderr_of(&out));
    for sid in ["S1", "S2"] {
        let p = dir4
            .join("emission")
            .join(sid)
            .join("telescope.emission.parquet.provenance.json");
        let r = read_json(&p);
        assert_eq!(
            r.get("prior_scope").and_then(|v| v.as_str()),
            Some("single_sample"),
            "--prior-scope sample gives every sample its own prior: {r:?}"
        );
        assert_eq!(
            r.get("n_prior_samples").and_then(|v| v.as_num()),
            Some(1.0),
            "a per-sample prior used ONE sample: {r:?}"
        );
    }

    for d in [&dir, &dir2, &dir3, &dir4] {
        let _ = std::fs::remove_dir_all(d);
    }
    let _ = std::fs::remove_file(&bam1);
    let _ = std::fs::remove_file(&bam2);
}

#[test]
fn an_unequal_number_of_bams_and_sample_ids_is_an_error() {
    let bam1 = synthetic_bam("t6_a");
    let bam2 = synthetic_bam_second("t6_b");
    let gtf = tiny_gtf();
    let dir = scratch("t6", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let out = run_auto(&dir, &[bam1.clone(), bam2.clone()], &["S1"], &gtf, &[]);
    assert!(
        !out.status.success(),
        "two BAMs and one sample id must FAIL, not run on the shorter list"
    );
    let err = stderr_of(&out);
    assert!(
        err.contains("must have equal length"),
        "the error must say what is wrong: {err}"
    );

    let out = run_auto(&dir, &[bam1.clone()], &["S1", "S2"], &gtf, &[]);
    assert!(
        !out.status.success(),
        "one BAM and two sample ids must FAIL too"
    );

    assert!(
        !dir.join("S1.profile.parquet").exists(),
        "a refused run must not write a profile"
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam1);
    let _ = std::fs::remove_file(&bam2);
}

#[test]
fn the_impossible_flag_combinations_are_refused() {
    let bam = synthetic_bam("t7");
    let gtf = tiny_gtf();
    let dir = scratch("t7", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let emission = dir.join("telescope.emission.parquet");
    let profile = dir.join("S1.profile.parquet");

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--emission",
        emission.to_str().unwrap(),
        "--output",
        profile.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "the --emission mode must refuse --output-dir"
    );
    assert!(
        stderr_of(&out).contains("--output-dir"),
        "the error must name the offending flag: {}",
        stderr_of(&out)
    );

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--gtf",
        gtf.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "the automatic mode must refuse a run that names no output at all"
    );
    let err = stderr_of(&out);
    assert!(
        err.contains("--output <FILE> is required") && !err.contains("--output-dir"),
        "the error must name the single-sample output flag: {}",
        err
    );

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--emission",
        emission.to_str().unwrap(),
        "--gtf",
        gtf.to_str().unwrap(),
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "--emission and --gtf must be mutually exclusive"
    );
    assert!(
        stderr_of(&out).contains("mutually exclusive"),
        "the error must say that the two flags exclude each other: {}",
        stderr_of(&out)
    );

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
        "--output-dir",
        dir.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "the automatic mode must refuse --output together with --output-dir"
    );
    assert!(
        stderr_of(&out).contains("mutually exclusive"),
        "the error must say that the two flags exclude each other: {}",
        stderr_of(&out)
    );

    let bam2 = synthetic_bam_second("t7_b");
    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--bam",
        bam2.to_str().unwrap(),
        "--sample-id",
        "S2",
        "--gtf",
        gtf.to_str().unwrap(),
        "--output",
        profile.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "--output must refuse a run over several samples"
    );
    assert!(
        stderr_of(&out).contains("--output-dir"),
        "the error must point at the flag that does fit: {}",
        stderr_of(&out)
    );
    let _ = std::fs::remove_file(&bam2);

    let out = telescope(&[
        "quant",
        "--bam",
        bam.to_str().unwrap(),
        "--sample-id",
        "S1",
        "--output-dir",
        dir.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "the automatic mode must refuse a run without --gtf"
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn the_temporary_intermediates_are_removed_after_a_run_and_after_a_failure() {
    let bam = synthetic_bam("t9");
    let gtf = tiny_gtf();
    let dir = scratch("t9", "dir");
    let _ = std::fs::remove_dir_all(&dir);

    let out = run_auto(&dir, &[bam.clone()], &["S1"], &gtf, &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let err = stderr_of(&out);
    let tmp = temp_dir_named_in(&err);
    assert!(
        err.contains("anchor_pool"),
        "the run must have used the temporary directory it named: {err}"
    );
    assert!(
        !tmp.exists(),
        "the temporary directory {} must be gone after the run",
        tmp.display()
    );

    let missing = dir.join("no_such.bam");
    let out = run_auto(&dir, &[missing], &["S1"], &gtf, &[]);
    assert!(!out.status.success(), "a run over a missing BAM must fail");
    let err = stderr_of(&out);
    let tmp = temp_dir_named_in(&err);
    assert!(
        err.contains("anchor_pool"),
        "the failed run must have reached the anchor stage: {err}"
    );
    assert!(
        !tmp.exists(),
        "the temporary directory {} must be gone after the FAILED run too",
        tmp.display()
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam);
}

fn temp_dir_named_in(stderr: &str) -> PathBuf {
    for line in stderr.lines() {
        if let Some(rest) = line
            .trim()
            .strip_prefix("[telescope]  temporary intermediates: ")
        {
            return PathBuf::from(rest.trim());
        }
    }
    panic!("the run did not name its temporary directory:\n{stderr}");
}

#[test]
fn the_scope_is_read_out_of_the_receipt() {
    let bam1 = synthetic_bam("t10_a");
    let bam2 = synthetic_bam_second("t10_b");
    let gtf = tiny_gtf();

    let dir = scratch("t10", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    let out = run_auto(
        &dir,
        &[bam1.clone(), bam2.clone()],
        &["S1", "S2"],
        &gtf,
        &[],
    );
    assert!(out.status.success(), "{}", stderr_of(&out));

    let r = receipt_of(&dir);
    assert_eq!(
        r.get("prior_scope").and_then(|v| v.as_str()),
        Some("cohort"),
        "the receipt is the authority on the scope: {r:?}"
    );

    let raw = std::fs::read_to_string(
        dir.join("emission")
            .join("telescope.emission.parquet.provenance.json"),
    )
    .unwrap();
    assert!(
        !raw.contains("\"prior_scope\": \"auto\""),
        "the receipt must state the scope that was USED, never `auto`:\n{raw}"
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam1);
    let _ = std::fs::remove_file(&bam2);
}

#[test]
fn several_samples_write_several_files_and_not_one() {
    let bam1 = synthetic_bam("t11_a");
    let bam2 = synthetic_bam_second("t11_b");
    let gtf = tiny_gtf();
    let dir = scratch("t11", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let ledger = dir.join("out.json");
    let assignments = dir.join("assign.parquet");
    let params = dir.join("params.parquet");
    let out = run_auto(
        &dir,
        &[bam1.clone(), bam2.clone()],
        &["S1", "S2"],
        &gtf,
        &[
            "--ledger",
            ledger.to_str().unwrap(),
            "--read-assignments",
            assignments.to_str().unwrap(),
            "--dump-emission-params",
            params.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{}", stderr_of(&out));

    for (stem, ext) in [
        ("out", "json"),
        ("assign", "parquet"),
        ("params", "parquet"),
    ] {
        for sid in ["S1", "S2"] {
            let p = dir.join(format!("{stem}.{sid}.{ext}"));
            assert!(
                p.exists(),
                "the automatic mode must write {} for every sample",
                p.display()
            );
        }
        assert!(
            !dir.join(format!("{stem}.{ext}")).exists(),
            "the unsuffixed name must NOT be written: the samples would overwrite \
             each other"
        );
    }

    let a = std::fs::read(dir.join("out.S1.json")).unwrap();
    let b = std::fs::read(dir.join("out.S2.json")).unwrap();
    assert!(
        a != b,
        "the two ledgers must be the two samples' own ledgers, not one of them twice"
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam1);
    let _ = std::fs::remove_file(&bam2);
}

#[test]
fn the_receipt_is_valid_json_and_keeps_every_old_field() {
    let bam = synthetic_bam("t12");
    let gtf = tiny_gtf();
    let dir = scratch("t12", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let out = run_auto(&dir, &[bam.clone()], &["S1"], &gtf, &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let auto_receipt = dir
        .join("emission")
        .join("telescope.emission.parquet.provenance.json");
    let r = read_json(&auto_receipt);

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
    let sub_receipt = dir.join("sub.emission.parquet.provenance.json");
    let rs = read_json(&sub_receipt);

    let old_keys = [
        "reference_source",
        "reference_count",
        "reference_checksum",
        "reference_checksum_note",
        "imputation_rule",
        "imputed_marker",
        "schema_columns",
        "n_anchored",
        "n_imputed",
        "n_total",
        "n_anchored_not_in_reference",
    ];
    for (name, r) in [("automatic", &r), ("subcommand", &rs)] {
        let keys = r.keys();
        assert_eq!(
            &keys[..old_keys.len()],
            &old_keys[..],
            "the {name} receipt must keep the old fields, in the old order"
        );
        assert_eq!(
            &keys[old_keys.len()..],
            &[
                "prior_scope".to_string(),
                "n_prior_samples".to_string(),
                "prior_sample_ids".to_string()
            ],
            "the {name} receipt must append exactly the three prior fields"
        );
        assert_eq!(
            r.get("imputation_rule").and_then(|v| v.as_str()),
            Some("cohort_prior"),
            "the {name} receipt must keep the old values"
        );
        assert_eq!(
            r.get("imputed_marker").and_then(|v| v.as_str()),
            Some("n_cohort_flnc == 0"),
            "the {name} receipt must keep the old values"
        );
        assert_eq!(
            r.get("schema_columns").and_then(|v| v.as_num()),
            Some(7.0),
            "the {name} receipt must keep the old values"
        );
        assert_eq!(
            r.get("reference_checksum"),
            Some(&Json::Null),
            "the {name} receipt must keep the old values"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn the_two_receipts_agree_on_the_old_fields() {
    let bam = synthetic_bam("t12b");
    let gtf = tiny_gtf();
    let dir = scratch("t12b", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let out = run_auto(&dir, &[bam.clone()], &["S1"], &gtf, &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let auto = read_json(
        &dir.join("emission")
            .join("telescope.emission.parquet.provenance.json"),
    );

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

    for key in [
        "reference_count",
        "imputation_rule",
        "imputed_marker",
        "schema_columns",
        "n_anchored",
        "n_imputed",
        "n_total",
        "n_anchored_not_in_reference",
        "prior_scope",
        "n_prior_samples",
    ] {
        assert_eq!(
            auto.get(key),
            sub.get(key),
            "the two receipts must agree on `{key}`"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn the_output_directory_has_the_documented_layout() {
    let bam1 = synthetic_bam("t12c_a");
    let bam2 = synthetic_bam_second("t12c_b");
    let gtf = tiny_gtf();
    let dir = scratch("t12c", "dir");
    let _ = std::fs::remove_dir_all(&dir);

    let out = run_auto(
        &dir,
        &[bam1.clone(), bam2.clone()],
        &["S1", "S2"],
        &gtf,
        &[],
    );
    assert!(out.status.success(), "{}", stderr_of(&out));

    for p in [
        "emission/telescope.emission.parquet",
        "emission/telescope.emission.parquet.provenance.json",
        "emission/telescope.emission.provenance.parquet",
        "S1.profile.parquet",
        "S2.profile.parquet",
    ] {
        let f = dir.join(p);
        assert!(f.exists(), "the output directory must hold {p}");
    }

    assert!(
        !dir.join("anchor_pool.parquet").exists(),
        "the anchor pool is a temporary intermediate and must not be left in \
         the output directory"
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&bam1);
    let _ = std::fs::remove_file(&bam2);
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
fn the_single_sample_output_mode_writes_one_profile_and_nothing_else() {
    let bam = synthetic_bam("t13");
    let gtf = tiny_gtf();
    let dir = scratch("t13", "dir");
    let auto_dir = scratch("t13_auto", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&auto_dir);
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
    assert!(
        out.status.success(),
        "the single-sample --output mode must run: {}",
        stderr_of(&out)
    );

    let out2 = run_auto(&auto_dir, std::slice::from_ref(&bam), &["S1"], &gtf, &[]);
    assert!(out2.status.success(), "{}", stderr_of(&out2));
    assert!(
        common::n_rows_of(&profile) > 0,
        "the profile must not be empty"
    );
    assert_same_bytes(
        &profile,
        &auto_dir.join("S1.profile.parquet"),
        "the profile of the --output mode against the --output-dir mode",
    );

    let mut entries: Vec<String> = std::fs::read_dir(&dir)
        .expect("read the output directory")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    assert_eq!(
        entries,
        vec!["S1.profile.parquet".to_string()],
        "the --output mode must leave nothing but the profile behind: {:?}",
        entries
    );

    let err = stderr_of(&out);
    let tmps = temp_dirs_named_in(&err);
    assert!(
        !tmps.is_empty(),
        "the run must have named its temporary directory:\n{err}"
    );
    for t in &tmps {
        assert!(
            !t.exists(),
            "the temporary directory {} must be gone after the run",
            t.display()
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&auto_dir);
    let _ = std::fs::remove_file(&bam);
}

#[test]
fn the_reference_universe_falls_back_to_the_input_bams() {
    let bam = synthetic_bam("t14");
    let gtf = tiny_gtf();
    let dir = scratch("t14", "dir");
    let named = scratch("t14_named", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&named);

    let out = run_auto(&dir, std::slice::from_ref(&bam), &["S1"], &gtf, &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));

    let out2 = run_auto(
        &named,
        std::slice::from_ref(&bam),
        &["S1"],
        &gtf,
        &["--reference-bam", bam.to_str().unwrap()],
    );
    assert!(out2.status.success(), "{}", stderr_of(&out2));
    assert_same_bytes(
        &dir.join("emission").join("telescope.emission.parquet"),
        &named.join("emission").join("telescope.emission.parquet"),
        "the emission model without --reference-bam against the one with it",
    );
    assert_same_bytes(
        &dir.join("S1.profile.parquet"),
        &named.join("S1.profile.parquet"),
        "the profile without --reference-bam against the one with it",
    );

    let r = receipt_of(&dir);
    match r.get("reference_count") {
        Some(common::Json::Num(n)) => assert_eq!(
            *n, 3.0,
            "the receipt must count the universe that was read: {n}"
        ),
        other => panic!("the receipt must carry reference_count: {other:?}"),
    }
    let rn = receipt_of(&named);
    assert_eq!(
        r.get("reference_count"),
        rn.get("reference_count"),
        "the two runs must report the same universe size"
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&named);
    let _ = std::fs::remove_file(&bam);
}

fn gtf_with_tx3(tag: &str) -> PathBuf {
    let p = scratch(tag, "gtf");
    let body = "chr1\ttX\ttranscript\t1\t2000\t.\t+\t.\tgene_id \"G1\"; transcript_id \"tx0\"; gene_name \"GA\";\n\
chr1\ttX\ttranscript\t1\t2000\t.\t+\t.\tgene_id \"G1\"; transcript_id \"tx1\"; gene_name \"GA\";\n\
chr1\ttX\ttranscript\t1\t2000\t.\t+\t.\tgene_id \"G2\"; transcript_id \"tx2\"; gene_name \"GB\";\n\
chr1\ttX\ttranscript\t1\t2000\t.\t+\t.\tgene_id \"G3\"; transcript_id \"tx3\"; gene_name \"GC\";\n";
    std::fs::write(&p, body).expect("write the gtf");
    p
}

#[test]
fn the_reference_universe_of_a_cohort_is_the_union_of_its_bams() {
    let bam1 = synthetic_bam("t15_a");
    let bam2 = synthetic_bam_other_universe("t15_b");
    let gtf = gtf_with_tx3("t15");
    let dir = scratch("t15", "dir");
    let named = scratch("t15_named", "dir");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&named);

    let out = run_auto(
        &dir,
        &[bam1.clone(), bam2.clone()],
        &["S1", "S2"],
        &gtf,
        &[],
    );
    assert!(out.status.success(), "{}", stderr_of(&out));

    let r = receipt_of(&dir);
    let count = |j: &Json| match j.get("reference_count") {
        Some(Json::Num(n)) => *n,
        other => panic!("the receipt must carry reference_count: {other:?}"),
    };
    assert_eq!(
        count(&r),
        4.0,
        "the universe must be the union of the two headers: {}",
        count(&r)
    );

    let out = run_auto(
        &named,
        &[bam1.clone(), bam2.clone()],
        &["S1", "S2"],
        &gtf,
        &["--reference-bam", bam1.to_str().unwrap()],
    );
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(
        count(&receipt_of(&named)),
        3.0,
        "naming one BAM must give that BAM's own header"
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&named);
    let _ = std::fs::remove_file(&bam1);
    let _ = std::fs::remove_file(&bam2);
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
        assert!(err.contains("--output <FILE> is required"), "{err}");
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
        "--sample-id",
        "S2",
        "--gtf",
        "missing.gtf",
    ]);
    assert!(!out.status.success());
    assert!(stderr_of(&out).contains("--output-dir <DIR> is required"));
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
