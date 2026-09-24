//! Telescope core library
//!
//! Phase C scaffold. Provides Module 1 (Emission Model Builder),
//! Module 2 (Transcriptome Evidence Extractor), and Module 3
//! (TEL-EM Solver). Module 4 R to follow per `specs/telescope-v0.1.7.md`.

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod em;
pub mod emission_model;
pub mod evidence;
