//! Being32 — Integration of Hex32 Substrate, BioRegNet, Active Inference, and Relational State
//!
//! The central agent struct combining all modules into a single integration
//! loop (`step(dt, feedback)`). Provides typed accessors for all 32 Hex32
//! fields, action computation via Active Inference, relational identity
//! updates, and shock/perturbation methods.
//!
//! ## Integration Order
//!
//! 1. Compute `mu` from state (pred_err, coherence, curvature)
//! 2. Sub-step BioRegNet (RK4) for valence/arousal evolution
//! 3. Update coherence via stability feedback
//! 4. Advance cascade engine on high relevance + error
//! 5. Pulse-gated learning on cascade completion
//! 6. Update relational identity (self-continuity, curvature)
//! 7. Track interoceptive oscillation
//!
//! ## Extension Contract (for Nexus / downstream layers)
//!
//! `step()` is decomposed into public sub-steps so that wrapper types
//! (e.g. `NexusBeing`) can orchestrate their own integration loop.
//!
//! **Correct pattern:**
//! ```ignore
//! impl NexusBeing {
//!     pub fn step(&mut self, dt: f32, fb: &WorldFeedback) {
//!         self.core.compute_mu_and_set();
//!         self.update_regime_target(dt);   // Nexus-specific
//!         self.core.step_bioregnet(dt);
//!         self.core.update_coherence(dt);
//!         self.core.advance_cascade(dt);
//!         self.core.pulse_gated_learning(fb);
//!         self.core.update_relational_identity(dt);
//!         self.core.track_interoception(dt);
//!     }
//! }
//! ```
//!
//! **Note:** Rust has no virtual dispatch on struct methods. Calling
//! `self.update_regime_target()` inside `Being32::step()` would *always*
//! call `Being32`'s version. The extension pattern is explicit orchestration,
//! not override.

use crate::hex32::Hex32;
use crate::relational_state::RelationalState;
use crate::social::{LocalContext, SocialField};
use crate::bio_regnet::BioRegNet;
use crate::active_inference::ActiveInference;

#[derive(Clone, Copy, Debug)]
pub struct ActionVector {
    pub approach: f32,
    pub avoid: f32,
    pub freeze: f32,
}

impl ActionVector {
    #[inline]
    pub fn normalize(&mut self) {
        let s = self.approach + self.avoid + self.freeze;
        if s > 0.0 {
            self.approach /= s;
            self.avoid /= s;
            self.freeze /= s;
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WorldFeedback {
    pub reward: f32,
    pub threat: f32,
    pub contact: f32,
}

#[derive(Clone, Debug)]
pub struct Being32 {
    pub core: Hex32,
    pub rel_state: RelationalState,
    pub regnet: BioRegNet,
    pub inference: ActiveInference,
}

impl Being32 {
    pub fn new() -> Self {
        let mut s = Self {
            core: Hex32::new(),
            rel_state: RelationalState::new(),
            regnet: BioRegNet::new(),
            inference: ActiveInference::new(),
        };
        s.awaken_to_baseline();
        s
    }

    fn clamp(x: f32, lo: f32, hi: f32) -> f32 { x.max(lo).min(hi) }

    #[inline]
    fn get_f32(&self, idx: usize) -> f32 { f32::from_bits(self.core.get_word(idx)) }

    #[inline]
    fn set_f32(&mut self, idx: usize, val: f32) { self.core.set_word(idx, val.to_bits()); }

    pub fn id_trait(&self) -> [f32; 3] { [self.get_f32(0), self.get_f32(1), self.get_f32(2)] }
    pub fn set_id_trait(&mut self, v: [f32; 3]) { self.set_f32(0, v[0]); self.set_f32(1, v[1]); self.set_f32(2, v[2]); }
    pub fn aff_valence(&self) -> f32 { self.get_f32(3) }
    pub fn set_aff_valence(&mut self, v: f32) { self.set_f32(3, Self::clamp(v, -1.0, 1.0)); }
    pub fn aff_arousal(&self) -> f32 { self.get_f32(4) }
    pub fn set_aff_arousal(&mut self, v: f32) { self.set_f32(4, Self::clamp(v, 0.0, 2.0)); }
    pub fn aff_tension(&self) -> f32 { self.get_f32(5) }
    pub fn set_aff_tension(&mut self, v: f32) { self.set_f32(5, Self::clamp(v, -1.0, 2.0)); }
    pub fn aff_coherence(&self) -> f32 { self.get_f32(6) }
    pub fn set_aff_coherence(&mut self, v: f32) { self.set_f32(6, Self::clamp(v, 0.0, 1.0)); }
    pub fn int_load(&self) -> f32 { self.get_f32(7) }
    pub fn set_int_load(&mut self, v: f32) { self.set_f32(7, Self::clamp(v, 0.0, 1.0)); }
    pub fn int_fatigue(&self) -> f32 { self.get_f32(8) }
    pub fn set_int_fatigue(&mut self, v: f32) { self.set_f32(8, Self::clamp(v, 0.0, 1.0)); }
    pub fn int_osc(&self) -> f32 { self.get_f32(9) }
    pub fn set_int_osc(&mut self, v: f32) { self.set_f32(9, Self::clamp(v, -1.0, 1.0)); }
    pub fn app_pred_err(&self) -> f32 { self.get_f32(10) }
    pub fn set_app_pred_err(&mut self, v: f32) { self.set_f32(10, Self::clamp(v, 0.0, 1.0)); }
    pub fn app_relevance(&self) -> f32 { self.get_f32(11) }
    pub fn set_app_relevance(&mut self, v: f32) { self.set_f32(11, Self::clamp(v, 0.0, 1.0)); }
    pub fn app_expect_impact(&self) -> f32 { self.get_f32(12) }
    pub fn set_app_expect_impact(&mut self, v: f32) { self.set_f32(12, Self::clamp(v, 0.0, 1.0)); }
    pub fn cas_phase(&self) -> f32 { self.get_f32(13) }
    pub fn set_cas_phase(&mut self, v: f32) { self.set_f32(13, Self::clamp(v, 0.0, 1.0)); }
    pub fn cas_intensity(&self) -> f32 { self.get_f32(14) }
    pub fn set_cas_intensity(&mut self, v: f32) { self.set_f32(14, Self::clamp(v, 0.0, 1.0)); }
    pub fn cas_complete(&self) -> f32 { self.get_f32(15) }
    pub fn set_cas_complete(&mut self, v: f32) { self.set_f32(15, Self::clamp(v, 0.0, 1.0)); }
    pub fn exp_open(&self) -> f32 { self.get_f32(16) }
    pub fn set_exp_open(&mut self, v: f32) { self.set_f32(16, Self::clamp(v, 0.0, 1.0)); }
    pub fn exp_modulation(&self) -> f32 { self.get_f32(17) }
    pub fn set_exp_modulation(&mut self, v: f32) { self.set_f32(17, Self::clamp(v, -1.0, 1.0)); }
    pub fn bnd_soc_load(&self) -> f32 { self.get_f32(18) }
    pub fn set_bnd_soc_load(&mut self, v: f32) { self.set_f32(18, Self::clamp(v, 0.0, 1.0)); }
    pub fn bnd_permeability(&self) -> f32 { self.get_f32(19) }
    pub fn set_bnd_permeability(&mut self, v: f32) { self.set_f32(19, Self::clamp(v, 0.0, 1.0)); }
    pub fn rel_curvature(&self) -> f32 { self.get_f32(20) }
    pub fn set_rel_curvature(&mut self, v: f32) { self.set_f32(20, Self::clamp(v, -1.0, 1.0)); }
    pub fn rel_trust(&self) -> f32 { self.get_f32(21) }
    pub fn set_rel_trust(&mut self, v: f32) { self.set_f32(21, Self::clamp(v, 0.0, 1.0)); }
    pub fn rel_stability(&self) -> f32 { self.get_f32(22) }
    pub fn set_rel_stability(&mut self, v: f32) { self.set_f32(22, Self::clamp(v, 0.0, 1.0)); }
    pub fn nar_self_cont(&self) -> f32 { self.get_f32(23) }
    pub fn set_nar_self_cont(&mut self, v: f32) { self.set_f32(23, Self::clamp(v, 0.0, 1.0)); }
    pub fn nar_drift(&self) -> f32 { self.get_f32(24) }
    pub fn set_nar_drift(&mut self, v: f32) { self.set_f32(24, Self::clamp(v, -1.0, 1.0)); }
    pub fn som_heart(&self) -> f32 { self.get_f32(25) }
    pub fn set_som_heart(&mut self, v: f32) { self.set_f32(25, Self::clamp(v, 0.0, 2.0)); }
    pub fn som_breath(&self) -> f32 { self.get_f32(26) }
    pub fn set_som_breath(&mut self, v: f32) { self.set_f32(26, Self::clamp(v, 0.0, 2.0)); }
    pub fn som_tremor(&self) -> f32 { self.get_f32(27) }
    pub fn set_som_tremor(&mut self, v: f32) { self.set_f32(27, Self::clamp(v, 0.0, 1.0)); }
    pub fn meta_energy(&self) -> f32 { self.get_f32(28) }
    pub fn set_meta_energy(&mut self, v: f32) { self.set_f32(28, Self::clamp(v, 0.0, 1.0)); }
    pub fn meta_absence_delta(&self) -> f32 { self.get_f32(29) }
    pub fn set_meta_absence_delta(&mut self, v: f32) { self.set_f32(29, Self::clamp(v, -1.0, 1.0)); }
    pub fn meta_error_corr(&self) -> f32 { self.get_f32(30) }
    pub fn set_meta_error_corr(&mut self, v: f32) { self.set_f32(30, Self::clamp(v, -1.0, 1.0)); }
    pub fn flags(&self) -> u32 { self.core.get_word(31) }
    pub fn set_flags(&mut self, v: u32) { self.core.set_word(31, v); }

    fn awaken_to_baseline(&mut self) {
        self.set_som_heart(1.0);
        self.set_som_breath(1.0);
        self.set_som_tremor(0.0);
        self.set_aff_valence(0.2);
        self.set_aff_arousal(0.8);
        self.set_aff_tension(0.0);
        self.set_nar_self_cont(1.0);
        self.set_rel_curvature(0.0);
        self.set_bnd_permeability(0.5);
    }

    pub fn receive_social_field(&mut self, field: &SocialField) {
        let v = self.aff_valence();
        let a = self.aff_arousal();
        let p = self.bnd_permeability();
        let new_v = v + 0.05 * (field.avg_valence - v);
        let new_a = a + 0.05 * (field.avg_arousal - a);
        self.set_aff_valence(new_v);
        self.set_aff_arousal(new_a);
        let new_perm = p * (1.0 - 0.1 * field.density.min(5.0));
        self.set_bnd_permeability(new_perm);
        let mismatch = (v - field.avg_valence).abs();
        let curv = self.rel_curvature() + 0.05 * (mismatch - self.rel_curvature());
        self.set_rel_curvature(curv.clamp(-1.0, 1.0));
    }

    pub fn compute_action(&self, _ctx: &LocalContext) -> ActionVector {
        let v = self.aff_valence();
        let a = self.aff_arousal();
        let t = self.aff_tension();
        let bond = if self.rel_state.dyads.is_empty() {
            0.0_f32
        } else {
            self.rel_state.dyads.iter().map(|d| d.affinity.abs()).sum::<f32>()
                / self.rel_state.dyads.len() as f32
        };
        let policy = self.inference.compute_policy(v, a, t, bond);
        ActionVector { approach: policy[0], avoid: policy[1], freeze: policy[2] }
    }

    pub fn apply_action(&mut self, act: ActionVector) {
        let heart = self.som_heart() + 0.1 * (act.approach - act.avoid);
        let breath = self.som_breath() + 0.05 * (act.approach - act.avoid);
        let tremor = self.som_tremor() + 0.1 * act.freeze;
        self.set_som_heart(heart.clamp(0.0, 2.0));
        self.set_som_breath(breath.clamp(0.0, 2.0));
        self.set_som_tremor(tremor.clamp(0.0, 1.0));
        let load = self.bnd_soc_load() + 0.1 * act.avoid - 0.05 * act.approach;
        self.set_bnd_soc_load(load.clamp(0.0, 1.0));
        let drift = self.nar_drift() + 0.02 * (act.avoid - act.approach);
        self.set_nar_drift(drift.clamp(-1.0, 1.0));
    }

    pub fn calculate_relational_load(&self) -> f32 {
        if self.rel_state.dyads.is_empty() { return 0.0; }
        let mut total_load = 0.0;
        for dyad in &self.rel_state.dyads {
            let dyad_harmony = dyad.affinity * dyad.trust;
            let dyad_load = 1.0 - dyad_harmony;
            total_load += dyad_load.max(0.0);
        }
        total_load / (self.rel_state.dyads.len() as f32)
    }

    pub fn update_relational_identity(&mut self, dt: f32) {
        let current_load = self.calculate_relational_load();
        let dyad_count = self.rel_state.dyads.len() as f32;
        let saturation_factor = if dyad_count <= 1.0 { 1.0 }
            else { (1.0 / (1.0 + 0.1 * (dyad_count - 1.0))).clamp(0.3, 1.0) };
        let effective_load = current_load * saturation_factor;
        let capped_load = effective_load.min(2.0);
        let mut target_self_cont = (1.0 - (capped_load - 0.5)).clamp(0.0, 1.0);
        let identity_floor = 0.2;
        if target_self_cont < identity_floor { target_self_cont = identity_floor; }
        let current_self_cont = self.nar_self_cont();
        let delta = target_self_cont - current_self_cont;
        let max_step = 0.02 * dt;
        let clamped_delta = delta.clamp(-max_step, max_step);
        let recovery_bias = if clamped_delta > 0.0 { 1.2 } else { 1.0 };
        let new_self_cont = (current_self_cont + clamped_delta * recovery_bias).clamp(0.0, 1.0);
        self.set_nar_self_cont(new_self_cont);
        let mut target_curvature = (capped_load - 1.0).clamp(-1.0, 1.0);
        if capped_load < 0.5 { target_curvature *= 0.5; }
        let current_curvature = self.rel_curvature();
        let curvature_delta = target_curvature - current_curvature;
        let max_curv_step = 0.04 * dt;
        let clamped_curv_delta = curvature_delta.clamp(-max_curv_step, max_curv_step);
        let new_curvature = (current_curvature + clamped_curv_delta).clamp(-1.0, 1.0);
        self.set_rel_curvature(new_curvature);
    }

    // ------------------------------------------------------------------
    // Decomposed sub-steps (public so Nexus can orchestrate its own loop)
    // ------------------------------------------------------------------

    /// Compute `mu` from current state and set `regnet.mu`.
    ///
    /// Reads: `app_pred_err`, `aff_coherence`, `rel_curvature`.
    pub fn compute_mu_and_set(&mut self) {
        let pred_err = self.app_pred_err();
        let coh = self.aff_coherence();
        let curv = self.rel_curvature();
        let mu = BioRegNet::compute_mu(pred_err, coh, curv);
        self.regnet.mu = mu;
    }

    /// Run BioRegNet RK4 sub-stepping on valence and arousal.
    ///
    /// Reads/writes: `aff_valence`, `aff_arousal` (via `regnet.step`).
    pub fn step_bioregnet(&mut self, dt: f32) {
        let regnet_dt = 0.01_f32;
        let sub_steps = (dt / regnet_dt).ceil() as usize;
        let actual_dt = dt / sub_steps as f32;

        let mut valence = f32::from_bits(self.core.get_word(3));
        let mut arousal = f32::from_bits(self.core.get_word(4));
        for _ in 0..sub_steps {
            self.regnet.step(&mut valence, &mut arousal, actual_dt);
        }
        self.core.set_word(3, valence.to_bits());
        self.core.set_word(4, arousal.to_bits());
    }

    /// Update coherence via stability feedback.
    ///
    /// Reads: `rel_curvature`, `nar_drift`, `aff_coherence`, `regnet.mu`.
    /// Writes: `aff_coherence`.
    pub fn update_coherence(&mut self, _dt: f32) {
        let curv = self.rel_curvature();
        let drift = self.nar_drift();
        let coh = self.aff_coherence();
        let mu = self.regnet.mu;
        let target_coh = (-curv + (1.0 - drift.abs())).clamp(0.0, 1.0);
        let stability = (1.0 + mu).clamp(0.0, 1.0);
        let new_coh = coh + 0.05 * (target_coh * stability - coh);
        self.set_aff_coherence(new_coh);
    }

    /// Advance the cascade engine.
    ///
    /// Reads: `app_relevance`, `app_pred_err`, `cas_phase`, `cas_intensity`.
    /// Writes: `cas_phase`, `cas_intensity`, `cas_complete`.
    pub fn advance_cascade(&mut self, dt: f32) {
        let rel = self.app_relevance();
        let pred_err = self.app_pred_err();
        let mut phase = self.cas_phase();
        let mut intensity = self.cas_intensity();
        if rel > 0.5 && pred_err > 0.2 {
            let mood_factor = 0.5 + 0.5 * (self.rel_state.mood.arousal / 2.0);
            phase += dt * (0.5 + intensity) * mood_factor;
        }
        if phase >= 1.0 {
            self.set_cas_complete(1.0);
            phase = 0.0;
            intensity *= 0.5;
        } else {
            self.set_cas_complete(0.0);
        }
        self.set_cas_phase(phase);
        self.set_cas_intensity(intensity.clamp(0.0, 1.0));
    }

    /// Pulse-gated learning on cascade completion.
    ///
    /// Reads: `cas_complete`, `aff_tension`, `app_expect_impact`, `rel_trust`,
    ///        `bnd_permeability`.
    /// Writes: `app_expect_impact`, `rel_trust`, `bnd_permeability`.
    pub fn pulse_gated_learning(&mut self, fb: &WorldFeedback) {
        if self.cas_complete() > 0.5 {
            let reward = fb.reward;
            let threat = fb.threat;
            let contact = fb.contact;
            let reward_p = reward * (0.5 + 0.5 * contact);
            let stress = (threat + self.aff_tension()) / 2.0;
            let safety = 1.0 - stress;
            let e = self.app_expect_impact();
            self.set_app_expect_impact(e + 0.05 * (reward_p - e));
            let tr = self.rel_trust();
            self.set_rel_trust(tr + 0.01 * (safety - tr));
            let bp = self.bnd_permeability();
            self.set_bnd_permeability(bp + 0.01 * (contact - bp));
        }
    }

    /// Track interoceptive oscillation.
    ///
    /// Reads: `int_osc`, `som_heart`.
    /// Writes: `int_osc`.
    pub fn track_interoception(&mut self, dt: f32) {
        let osc = self.int_osc() + dt * (self.som_heart() - 1.0);
        self.set_int_osc(osc.clamp(-1.0, 1.0));
    }

    // ------------------------------------------------------------------
    // Default integration loop — delegates to sub-steps in order.
    // CMAP tests call this directly; extension layers write their own.
    // ------------------------------------------------------------------

    pub fn step(&mut self, dt: f32, fb: &WorldFeedback) {
        self.compute_mu_and_set();
        self.step_bioregnet(dt);
        self.update_coherence(dt);
        self.advance_cascade(dt);
        self.pulse_gated_learning(fb);
        self.update_relational_identity(dt);
        self.track_interoception(dt);
    }

    pub fn perceptual_radius(&self) -> f32 {
        let base = 1.0;
        let coherence = self.aff_coherence().clamp(0.0, 1.0);
        let arousal = self.aff_arousal().clamp(0.0, 2.0);
        let mood_open = self.rel_state.mood.openness.clamp(0.0, 1.0);
        base * coherence * (2.0 - arousal).max(0.1) * (0.5 + 0.5 * mood_open)
    }

    pub fn apply_shock(&mut self, magnitude: f32) {
        let m = magnitude.clamp(0.0, 1.0);
        let v = self.aff_valence();
        self.set_aff_valence((v - m * 0.6).clamp(-1.0, 1.0));
        let a = self.aff_arousal();
        self.set_aff_arousal((a + m * 0.8).clamp(0.0, 2.0));
        let t = self.aff_tension();
        self.set_aff_tension((t + m).clamp(-1.0, 2.0));
    }

    pub fn kill(&self) -> String {
        let bytes = self.core.to_bytes();
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    pub fn get_attestation(&self) -> String {
        self.kill()
    }

    pub fn rehydrate(&mut self, att: &str) -> bool {
        if att.len() != 256 {
            return false;
        }
        let mut bytes = [0u8; 128];
        for i in 0..128 {
            let hex_byte = &att[i * 2..i * 2 + 2];
            match u8::from_str_radix(hex_byte, 16) {
                Ok(val) => bytes[i] = val,
                Err(_) => return false,
            }
        }
        self.core = Hex32::from_bytes(bytes);
        true
    }
}
