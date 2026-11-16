mod common;

use common::{stderr_of, telescope};
use telescope::{
    ANCHOR_GAP_ABS_DEFAULT, ANCHOR_GAP_REL_DEFAULT, ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT,
    ANCHOR_MIN_COVERAGE_DEFAULT, ANCHOR_TES_SLACK_DEFAULT, EMISSION_COHORT_MIN_READS_DEFAULT,
    EMISSION_GENE_MIN_READS_DEFAULT, EMISSION_TX_MIN_READS_DEFAULT,
};

#[test]
fn the_help_shows_only_quant_and_diff() {
    let out = telescope(&["--help"]);
    assert!(out.status.success(), "--help must succeed");
    let help = String::from_utf8_lossy(&out.stdout).into_owned();

    for visible in ["quant", "diff"] {
        assert!(
            help.contains(visible),
            "`{visible}` must be visible in the help:\n{help}"
        );
    }
    for hidden in ["build-anchors", "build-emission", "summarize"] {
        assert!(
            !help.contains(hidden),
            "`{hidden}` must be HIDDEN from the help:\n{help}"
        );
    }
}

#[test]
fn the_hidden_subcommands_still_work() {
    for cmd in ["build-anchors", "build-emission", "summarize"] {
        let out = telescope(&[cmd, "--help"]);
        assert!(
            out.status.success(),
            "`{cmd} --help` must exit 0, it exited {:?}: {}",
            out.status.code(),
            stderr_of(&out)
        );
        let help = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(
            help.contains(&format!("Usage: telescope {cmd}")),
            "`{cmd} --help` must describe itself:\n{help}"
        );
    }
}

#[test]
fn the_frozen_defaults_have_one_home() {
    assert_eq!(ANCHOR_TES_SLACK_DEFAULT, 35);
    assert_eq!(ANCHOR_MIN_COVERAGE_DEFAULT, 0.80);
    assert_eq!(ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT, 200);
    assert_eq!(ANCHOR_GAP_ABS_DEFAULT, 12.0);
    assert_eq!(ANCHOR_GAP_REL_DEFAULT, 0.0);
    assert_eq!(EMISSION_TX_MIN_READS_DEFAULT, 10);
    assert_eq!(EMISSION_GENE_MIN_READS_DEFAULT, 30);
    assert_eq!(EMISSION_COHORT_MIN_READS_DEFAULT, 50);
}

#[test]
fn the_build_anchors_defaults_are_the_constants() {
    let help =
        String::from_utf8_lossy(&telescope(&["build-anchors", "--help"]).stdout).into_owned();
    assert!(
        help.contains(&format!("[default: {}]", ANCHOR_TES_SLACK_DEFAULT)),
        "the rendered --tes-slack default must be the constant:\n{help}"
    );
    assert!(
        help.contains(&format!(
            "[default: {}]",
            ANCHOR_MAX_READS_PER_SAMPLE_DEFAULT
        )),
        "the rendered --max-reads-per-sample default must be the constant:\n{help}"
    );
    assert!(
        help.contains(&format!("[default: {}]", ANCHOR_GAP_ABS_DEFAULT)),
        "the rendered --gap-abs default must be the constant:\n{help}"
    );
}

#[test]
fn the_build_emission_defaults_are_the_constants() {
    let help =
        String::from_utf8_lossy(&telescope(&["build-emission", "--help"]).stdout).into_owned();
    assert!(
        help.contains(&format!("[default: {}]", EMISSION_TX_MIN_READS_DEFAULT)),
        "the rendered --tx-min-reads default must be the constant:\n{help}"
    );
    assert!(
        help.contains(&format!("[default: {}]", EMISSION_GENE_MIN_READS_DEFAULT)),
        "the rendered --gene-min-reads default must be the constant:\n{help}"
    );
    assert!(
        help.contains(&format!("[default: {}]", EMISSION_COHORT_MIN_READS_DEFAULT)),
        "the rendered --cohort-min-reads default must be the constant:\n{help}"
    );
}
