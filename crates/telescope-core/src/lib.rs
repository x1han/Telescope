//! Telescope core library.
//!
//! Modules:
//! - `resolver` — transcript evidence resolution (tier classification, subsumption)
//! - `evidence` — transcriptome evidence extraction from alignments
//! - `anchor_builder` — anchor pool construction
//! - `emission_model` — per-transcript poly(A) emission model
//! - `em` — TEL-EM solver
//! - `summary` — profile summary engine

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod anchor_builder;
pub mod em;
pub mod emission_model;
pub mod evidence;
pub mod resolver;
pub mod summary;
