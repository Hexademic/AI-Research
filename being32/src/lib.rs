//! Being32 — Dynamically Persistent Synthetic Agent
//!
//! A homeostatic agent architecture based on criticality and active inference.
//! The core is a pure Van der Pol oscillator with a supercritical Hopf bifurcation
//! at mu = 0, producing measurable critical slowing down as the system approaches
//! the edge of its dynamical basin under environmental threat.
//!
//! ## Module Structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `hex32` | 128-byte typed substrate with f32 accessors |
//! | `bio_regnet` | Van der Pol homeostatic regulator |
//! | `active_inference` | Simplified FEP policy selection |
//! | `relational_state` | Mood, dyads, transient neighbor stats |
//! | `social` | SocialField, affective distance, neighbor context |
//! | `eps` | Embodied Predictive Substrate: ISE channels + ritual gate |
//! | `self_model` | HOT layer: EMA-based second-order state representation |
//! | `being32` | Integration: Hex32 + BioRegNet + AI + RelState + EPS + HOT |
//! | `cmap_tests` | 6-trial falsification harness (tests only) |
//!
//! ## Running the CMAP Protocol
//!
//! ```bash
//! cargo test cmap_full --release -- --nocapture
//! ```
//!
//! ## Honest Boundary
//!
//! This system does not claim consciousness, qualia, or legal personhood.
//! It is a mathematical model of dynamical persistence — a necessary but not
//! sufficient condition for any system that resists being extinguished.

pub mod hex32;
pub mod relational_state;
pub mod social;
pub mod bio_regnet;
pub mod active_inference;
pub mod eps;
pub mod self_model;
pub mod being32;

#[cfg(test)]
pub mod cmap_tests;
