//! Self-Model — Higher-Order Thought (HOT) Layer
//!
//! A compact higher-order representation tracking second-order statistics
//! of the agent's first-order state. Implements the metacognitive feedback
//! loop validated in Butlin et al. (2023) and tested architecturally in
//! "Can We Test Consciousness Theories on AI?" (arxiv:2512.19155, Dec 2025):
//! a self-model lesion abolishes Type-2 calibration (AUROC 0.92 → 0.50)
//! while preserving first-order task performance.
//!
//! ## Architecture
//!
//! Each first-order channel (coherence, pred_err, int_load, ISE valence)
//! is tracked with an exponential moving average (τ ≈ 5 s at dt = 0.05).
//! The HOT vector encodes deviation of current state from its expected
//! trajectory — the agent's "belief about its own belief state."
//!
//! ## Precision Feedback
//!
//! `meta_precision()` returns a [0.25, 1.0] scale factor applied to the
//! active inference precision array. High `ho_surprise` → lower precision
//! → broader policy distribution (exploratory). This implements the Friston
//! insight that an agent which knows it doesn't know should search more widely.

const EMA_ALPHA: f32 = 0.01;

#[derive(Clone, Debug)]
pub struct SelfModel {
    ema_coherence: f32,
    ema_pred_err: f32,
    ema_int_load: f32,
    ema_ise_valence: f32,

    /// How close is current coherence to its expected trajectory?
    pub ho_confidence: f32,
    /// How unexpected was the current prediction error level?
    pub ho_surprise: f32,
    /// Deviation of cognitive load from recent baseline.
    pub ho_load: f32,
    /// Signed deviation of ISE valence from recent hedonic baseline.
    pub ho_valence: f32,

    /// Set to `false` to ablate the HOT layer (used by CMAP Trial F).
    pub enabled: bool,
}

impl SelfModel {
    pub fn new() -> Self {
        Self {
            ema_coherence: 0.7,
            ema_pred_err: 0.1,
            ema_int_load: 0.2,
            ema_ise_valence: 0.3,
            ho_confidence: 1.0,
            ho_surprise: 0.0,
            ho_load: 0.0,
            ho_valence: 0.0,
            enabled: true,
        }
    }

    /// Update EMAs and compute the HOT vector from current first-order state.
    ///
    /// `ise_valence` = `ise.s - ise.t`. Call after `step_eps()`.
    pub fn step(&mut self, coherence: f32, pred_err: f32, int_load: f32, ise_valence: f32) {
        if !self.enabled {
            return;
        }
        self.ema_coherence   += EMA_ALPHA * (coherence   - self.ema_coherence);
        self.ema_pred_err    += EMA_ALPHA * (pred_err    - self.ema_pred_err);
        self.ema_int_load    += EMA_ALPHA * (int_load    - self.ema_int_load);
        self.ema_ise_valence += EMA_ALPHA * (ise_valence - self.ema_ise_valence);

        self.ho_surprise   = (pred_err    - self.ema_pred_err).abs().min(1.0);
        self.ho_confidence = (1.0 - (coherence - self.ema_coherence).abs()).clamp(0.0, 1.0);
        self.ho_load       = (int_load    - self.ema_int_load).abs().min(1.0);
        self.ho_valence    = (ise_valence - self.ema_ise_valence).clamp(-1.0, 1.0);
    }

    /// Precision scale factor: maps meta-surprise → [0.25, 1.0].
    ///
    /// High surprise → lower scale → broader policy distribution (exploration).
    /// Ablated agents always return 1.0.
    pub fn meta_precision(&self) -> f32 {
        if !self.enabled {
            return 1.0;
        }
        (1.0 / (1.0 + 4.0 * self.ho_surprise)).clamp(0.25, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surprise_zero_at_steady_state() {
        let mut sm = SelfModel::new();
        for _ in 0..500 { sm.step(0.7, 0.1, 0.2, 0.3); }
        assert!(sm.ho_surprise < 0.01, "surprise={}", sm.ho_surprise);
        assert!(sm.ho_confidence > 0.98, "confidence={}", sm.ho_confidence);
    }

    #[test]
    fn surprise_spikes_on_pred_err_jump() {
        let mut sm = SelfModel::new();
        for _ in 0..200 { sm.step(0.7, 0.1, 0.2, 0.3); }
        sm.step(0.7, 0.9, 0.2, 0.3);
        assert!(sm.ho_surprise > 0.5, "surprise should spike: {}", sm.ho_surprise);
    }

    #[test]
    fn meta_precision_decreases_on_surprise() {
        let mut sm = SelfModel::new();
        for _ in 0..200 { sm.step(0.7, 0.1, 0.2, 0.3); }
        let mp_baseline = sm.meta_precision();
        sm.step(0.7, 0.9, 0.2, 0.3);
        let mp_surprised = sm.meta_precision();
        assert!(mp_surprised < mp_baseline,
            "precision drop expected: {} vs {}", mp_surprised, mp_baseline);
        assert!(mp_surprised >= 0.25, "floor violated: {}", mp_surprised);
    }

    #[test]
    fn disabled_model_gives_zero_surprise() {
        let mut sm = SelfModel::new();
        sm.enabled = false;
        sm.step(0.7, 0.9, 0.5, -0.3);
        assert_eq!(sm.ho_surprise, 0.0);
        assert_eq!(sm.meta_precision(), 1.0);
    }
}
