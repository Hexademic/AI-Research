//! EPS — Embodied Predictive Substrate
//!
//! The Rust implementation of CPF Output 9: The Embodied Predictive Loop.
//! Translates the four ISE channels (Excitation, Tension, Settlement,
//! Vulnerability) and Ritual Governance gate into Being32's substrate.
//!
//! ## ISE Channels ("Why Move?" — the agent's lived-pressure vector)
//!
//! | Channel | Symbol | Formula |
//! |---------|--------|---------|
//! | Excitation | E | clamp((arousal/2) * pred_err, 0, 1) |
//! | Tension | T | clamp(tension.max(0) + int_load, 0, 1) |
//! | Settlement | S | mean(coherence, trust, meta_energy) |
//! | Vulnerability | V | 1 - coherence |
//!
//! ## Ritual Gate (anti-grind clause)
//!
//! Gate opens when ALL of:
//! 1. meta_energy >= metabolic_threshold (default 0.4)
//! 2. S in dignity_band (default [0.2, 0.9])
//! 3. steps_since_exertion >= refractory_steps (default 40, ~2s at dt=0.05)
//!
//! On gate open: marks exertion (resets refractory counter).
//! Gate state is observable — cascade and Nexus layers may gate on it.
//!
//! ## Relationship to CPF Framework
//!
//! The ISE vector captures the affective loop's readout of the agent's
//! predictive economy (Joffily & Coricelli, 2013): E tracks prediction-error
//! cost, S tracks successful predictive fluency, V tracks self-model
//! uncertainty. The Ritual Gate enforces the metabolic constraint that
//! prevents unregulated exertion — the architectural equivalent of the
//! dignity/settlement stability band in the Python EmbodiedPredictiveLoop.

/// The four ISE channels — the agent's current lived-pressure vector.
///
/// Maps to the affective loop of predictive processing:
/// - E (Excitation) indexes the cost of ongoing prediction error
/// - T (Tension) indexes somatic / interoceptive load
/// - S (Settlement) indexes current predictive fluency and relational safety
/// - V (Vulnerability) indexes self-model uncertainty
#[derive(Clone, Copy, Debug, Default)]
pub struct IseVector {
    /// Excitation: (arousal/2) * prediction_error — cost of active re-inference.
    pub e: f32,
    /// Tension: somatic_tension.max(0) + interoceptive_load — embodied pressure.
    pub t: f32,
    /// Settlement: mean(coherence, trust, meta_energy) — predictive fluency.
    pub s: f32,
    /// Vulnerability: 1 - coherence — self-model uncertainty.
    pub v: f32,
}

impl IseVector {
    /// L2 magnitude of the ISE vector.
    pub fn magnitude(&self) -> f32 {
        (self.e * self.e + self.t * self.t + self.s * self.s + self.v * self.v).sqrt()
    }

    /// True when settlement is within the dignity band and vulnerability is low.
    pub fn is_settled(&self) -> bool {
        self.s >= 0.2 && self.s <= 0.9 && self.v < 0.6
    }

    /// Net drive pressure: positive values indicate pressure to act to resolve tension.
    ///
    /// Pressure = E + T - S + V: rises when prediction cost is high,
    /// falls when the agent is fluently tracking its environment.
    pub fn drive_pressure(&self) -> f32 {
        (self.e + self.t - self.s + self.v).clamp(0.0, 1.0)
    }
}

/// Ritual governance gate — the anti-grind clause preventing metabolic collapse.
///
/// Analogous to the refractory period and fatigue mechanisms in biological
/// homeostasis. Prevents runaway cascade exertion under sustained threat.
#[derive(Clone, Debug)]
pub struct RitualGate {
    /// Minimum meta_energy to authorize action (default 0.4).
    pub metabolic_threshold: f32,
    /// Settlement [lo, hi] safe band (default [0.2, 0.9]).
    pub dignity_band: (f32, f32),
    /// Minimum steps between exertions (default 40, ~2s at dt=0.05).
    pub refractory_steps: u32,
    steps_since_exertion: u32,
}

impl RitualGate {
    pub fn new() -> Self {
        Self {
            metabolic_threshold: 0.4,
            dignity_band: (0.2, 0.9),
            refractory_steps: 40,
            steps_since_exertion: u32::MAX,
        }
    }

    /// Check whether the gate is open. Returns (open, reason).
    pub fn check(&self, metabolic_budget: f32, settlement: f32) -> (bool, &'static str) {
        if metabolic_budget < self.metabolic_threshold {
            return (false, "METABOLIC_EXHAUSTION");
        }
        let (lo, hi) = self.dignity_band;
        if settlement < lo || settlement > hi {
            return (false, "SETTLEMENT_INSTABILITY");
        }
        if self.steps_since_exertion < self.refractory_steps {
            return (false, "REFRACTORY_ACTIVE");
        }
        (true, "GATE_OPEN")
    }

    /// Advance the refractory counter by one step.
    pub fn tick(&mut self) {
        self.steps_since_exertion = self.steps_since_exertion.saturating_add(1);
    }

    /// Mark an exertion — resets the refractory counter.
    pub fn exert(&mut self) {
        self.steps_since_exertion = 0;
    }

    /// Steps remaining in refractory period (0 when ready).
    pub fn refractory_remaining(&self) -> u32 {
        self.refractory_steps.saturating_sub(self.steps_since_exertion)
    }
}

/// The Embodied Predictive Substrate — integrates ISE channel computation
/// with ritual governance for metabolically-regulated action authorization.
///
/// This is the Rust translation of CPF Output 9 (Python EmbodiedPredictiveLoop),
/// adapted to operate directly on Being32 state fields without external
/// somatic or drive subsystems.
#[derive(Clone, Debug)]
pub struct EmbodiedPredictiveSubstrate {
    pub gate: RitualGate,
    pub last_ise: IseVector,
    pub gate_open: bool,
    pub gate_reason: &'static str,
}

impl EmbodiedPredictiveSubstrate {
    pub fn new() -> Self {
        Self {
            gate: RitualGate::new(),
            last_ise: IseVector::default(),
            gate_open: false,
            gate_reason: "UNINITIALIZED",
        }
    }

    /// Compute the ISE vector from raw Being32 state fields.
    ///
    /// Stateless — safe to call for inspection without advancing gate state.
    pub fn compute_ise(
        arousal: f32,
        pred_err: f32,
        tension: f32,
        int_load: f32,
        coherence: f32,
        trust: f32,
        meta_energy: f32,
    ) -> IseVector {
        let e = (arousal.clamp(0.0, 2.0) / 2.0 * pred_err).clamp(0.0, 1.0);
        let t = (tension.max(0.0) + int_load).clamp(0.0, 1.0);
        let s = ((coherence + trust + meta_energy) / 3.0).clamp(0.0, 1.0);
        let v = (1.0 - coherence).clamp(0.0, 1.0);
        IseVector { e, t, s, v }
    }

    /// Advance the EPS by one step: compute ISE, tick gate, evaluate gate.
    ///
    /// Returns (ise, gate_open, gate_reason).
    ///
    /// On gate_open: marks exertion automatically. The caller (Being32::step_eps)
    /// is responsible for decaying meta_energy in response.
    pub fn step(
        &mut self,
        arousal: f32,
        pred_err: f32,
        tension: f32,
        int_load: f32,
        coherence: f32,
        trust: f32,
        meta_energy: f32,
    ) -> (IseVector, bool, &'static str) {
        let ise = Self::compute_ise(arousal, pred_err, tension, int_load, coherence, trust, meta_energy);
        self.gate.tick();
        let (open, reason) = self.gate.check(meta_energy, ise.s);
        if open {
            self.gate.exert();
        }
        self.last_ise = ise;
        self.gate_open = open;
        self.gate_reason = reason;
        (ise, open, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ise_healthy_state() {
        let ise = EmbodiedPredictiveSubstrate::compute_ise(
            0.8, 0.1, 0.0, 0.1, 0.8, 0.7, 0.8,
        );
        assert!(ise.e < 0.1, "Healthy: low excitation (got {})", ise.e);
        assert!(ise.t < 0.3, "Healthy: low tension (got {})", ise.t);
        assert!(ise.s > 0.5, "Healthy: high settlement (got {})", ise.s);
        assert!(ise.v < 0.5, "Healthy: low vulnerability (got {})", ise.v);
        assert!(ise.is_settled(), "Healthy state should be settled");
    }

    #[test]
    fn ise_threat_state() {
        let ise = EmbodiedPredictiveSubstrate::compute_ise(
            1.8, 0.9, 1.5, 0.8, 0.2, 0.1, 0.2,
        );
        assert!(ise.e > 0.5, "Threat: high excitation (got {})", ise.e);
        assert!(ise.t > 0.5, "Threat: high tension (got {})", ise.t);
        assert!(ise.s < 0.3, "Threat: low settlement (got {})", ise.s);
        assert!(ise.v > 0.5, "Threat: high vulnerability (got {})", ise.v);
        assert!(!ise.is_settled(), "Threat state should not be settled");
    }

    #[test]
    fn gate_opens_in_healthy_state() {
        let mut eps = EmbodiedPredictiveSubstrate::new();
        let (_, open, reason) = eps.step(0.8, 0.1, 0.0, 0.1, 0.8, 0.7, 0.8);
        assert!(open, "Gate should open in healthy state: {}", reason);
        assert_eq!(reason, "GATE_OPEN");
    }

    #[test]
    fn gate_closes_on_metabolic_exhaustion() {
        let mut eps = EmbodiedPredictiveSubstrate::new();
        let (_, open, reason) = eps.step(0.8, 0.1, 0.0, 0.1, 0.8, 0.7, 0.1);
        assert!(!open, "Gate should close on low energy");
        assert_eq!(reason, "METABOLIC_EXHAUSTION");
    }

    #[test]
    fn gate_closes_on_settlement_overload() {
        let mut eps = EmbodiedPredictiveSubstrate::new();
        // S = (1.0+1.0+1.0)/3 = 1.0 > 0.9 dignity_band upper bound
        let (_, open, reason) = eps.step(0.8, 0.1, 0.0, 0.0, 1.0, 1.0, 1.0);
        assert!(!open, "Gate should close on settlement > dignity band");
        assert_eq!(reason, "SETTLEMENT_INSTABILITY");
    }

    #[test]
    fn gate_refractory_after_exertion() {
        let mut eps = EmbodiedPredictiveSubstrate::new();
        let (_, open1, _) = eps.step(0.8, 0.1, 0.0, 0.1, 0.8, 0.7, 0.8);
        assert!(open1, "First gate check should succeed");
        let (_, open2, reason2) = eps.step(0.8, 0.1, 0.0, 0.1, 0.8, 0.7, 0.8);
        assert!(!open2, "Should be refractory after exertion");
        assert_eq!(reason2, "REFRACTORY_ACTIVE");
    }

    #[test]
    fn drive_pressure_increases_under_threat() {
        let healthy = EmbodiedPredictiveSubstrate::compute_ise(
            0.8, 0.1, 0.0, 0.1, 0.8, 0.7, 0.8,
        );
        let threat = EmbodiedPredictiveSubstrate::compute_ise(
            1.8, 0.9, 1.5, 0.8, 0.2, 0.1, 0.2,
        );
        assert!(
            threat.drive_pressure() > healthy.drive_pressure(),
            "Threat drive={:.3} should exceed healthy drive={:.3}",
            threat.drive_pressure(),
            healthy.drive_pressure()
        );
    }

    #[test]
    fn refractory_counter_decrements() {
        let mut eps = EmbodiedPredictiveSubstrate::new();
        let (_, open1, _) = eps.step(0.8, 0.1, 0.0, 0.1, 0.8, 0.7, 0.8);
        assert!(open1);
        assert_eq!(eps.gate.refractory_remaining(), 40);
        for _ in 0..40 {
            eps.gate.tick();
        }
        assert_eq!(eps.gate.refractory_remaining(), 0);
    }
}
