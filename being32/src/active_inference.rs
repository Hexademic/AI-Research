//! Active Inference — Simplified Variational Policy Selection
//!
//! A minimal implementation of the Free Energy Principle (Friston, 2010)
//! for motor policy selection among three actions: approach, avoid, freeze.
//!
//! Expected Free Energy is decomposed into Risk (KL divergence from preferences)
//! and Ambiguity (entropy of outcomes), with precision modulated by dyadic
//! bond strength and prediction error.
//!
//! ```text
//! G(π) = Risk(π) + Ambiguity(π)/γ
//! P(π) = softmax(-G(π))
//! ```
//!
//! This module is `no_std` compatible and operates entirely on stack-allocated
//! `[f32; 3]` arrays.

pub type MotorPolicy = [f32; 3];
pub type PreferencePrior = [f32; 3];
pub type GenerativeLikelihood = [f32; 3];

#[derive(Clone, Debug)]
pub struct ActiveInference {
    pub precision: [f32; 4],
    pub preference: PreferencePrior,
    pub likelihoods: [GenerativeLikelihood; 3],
}

impl ActiveInference {
    pub fn new() -> Self {
        Self {
            precision: [1.0; 4],
            preference: [0.6, 0.1, 0.3],
            likelihoods: [
                [0.7, 0.2, 0.6], // approach
                [0.2, 0.1, 0.1], // avoid
                [0.1, 0.8, 0.1], // freeze
            ],
        }
    }

    pub fn compute_policy(&self, valence: f32, arousal: f32, tension: f32, bond_strength: f32) -> MotorPolicy {
        let _pe_valence = (valence - (self.preference[0] * 2.0 - 1.0)).abs();
        let _pe_arousal = (arousal - self.preference[2]).abs();
        let _pe_contact = bond_strength - self.preference[2];
        let _pe_tension = tension;

        let mut g_values: [f32; 3] = [0.0; 3];

        for pi in 0..3 {
            let likelihood = self.likelihoods[pi];

            let mut risk = 0.0_f32;
            for o in 0..3 {
                if likelihood[o] > 1e-6 && self.preference[o] > 1e-6 {
                    risk += likelihood[o] * (likelihood[o].ln() - self.preference[o].ln());
                }
            }

            let mut ambiguity = 0.0_f32;
            for o in 0..3 {
                if likelihood[o] > 1e-6 {
                    ambiguity -= likelihood[o] * likelihood[o].ln();
                }
            }
            let avg_precision = self.precision.iter().sum::<f32>() / 4.0;
            ambiguity /= avg_precision;

            g_values[pi] = risk + ambiguity;
        }

        let mut exp_neg_g: [f32; 3] = [0.0; 3];
        let mut sum_exp = 0.0_f32;
        for pi in 0..3 {
            exp_neg_g[pi] = (-g_values[pi]).exp();
            sum_exp += exp_neg_g[pi];
        }

        let mut policy: MotorPolicy = [0.0; 3];
        for pi in 0..3 {
            policy[pi] = exp_neg_g[pi] / sum_exp;
        }
        policy
    }

    pub fn update_precision(&mut self, bond_strength: f32, prediction_error: f32) {
        let xi = bond_strength;
        for i in 0..2 {
            let solo = 1.0;
            let bonded = 0.2 + 0.8 * (1.0 - prediction_error);
            self.precision[i] = (1.0 - xi) * solo + xi * bonded;
        }
        self.precision[2] = 0.5 + 0.5 * xi;
        self.precision[3] = 1.0;
    }
}
