//! Telescope core library
//!
//! Phase v0.2.0 scaffold. Provides:
//!   Module 1: Emission Model Builder (`emission_model`)
//!   Module 2: Transcriptome Evidence Extractor (`evidence`)
//!   Module 3: TEL-EM Solver (`em`)
//!   Step 2:   Profile Summary Engine (`summary`)
//! Module 4 R + Step 3 (Universal Mode) follow per `specs/telescope-v0.2.0-rc1.md`.

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod em;
pub mod emission_model;
pub mod evidence;
pub mod summary;
