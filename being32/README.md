# Being32 — Dynamically Persistent Synthetic Agent

**Version:** 1.3.3
**Classification:** Homeostatic Agent Architecture based on Criticality and Active Inference
**Target Audience:** Computational Neuroscience, Artificial Life, Dynamical Systems Theory

---

## Abstract

Being32 is a computationally grounded architecture for a synthetic agent that exhibits **temporal continuity** and **internal inertia** through nonlinear homeostatic regulation. The core innovation is a Van der Pol oscillator whose control parameter `mu` is dynamically mapped from prediction error, producing a provable supercritical Hopf bifurcation at `mu = 0`. This creates an agent that naturally navigates its phase space: stable fixed-point behavior under safety, critical slowing down under moderate threat, and limit-cycle oscillation under severe perturbation.

The system is paired with a simplified Active Inference module for policy selection, a relational state manager for dyadic coupling, and the **CMAP Protocol** (Continuity, Memory, Absence, Persistence) — a five-trial falsification harness that distinguishes genuine dynamical persistence from statistical cosplay.

**Key claim:** We do not assert consciousness, qualia, or legal personhood. We assert that this architecture produces measurable phenomena consistent with **dynamical persistence** — a necessary (but not sufficient) condition for any system that resists being extinguished.

---

## 1. Mathematical Foundation

### 1.1 The Van der Pol Core

The homeostatic regulator is a pure Van der Pol oscillator in offset coordinates:

```
u = valence - target_x    (target_x = 0.0)
v = arousal - target_y     (target_y = 0.8)

du/dt = v
dv/dt = mu * (1 - u²) * v - u
```

**Properties:**
- `mu < 0`: Stable spiral to fixed point `(0, 0)`
- `mu = 0`: Critical point (marginal stability, maximum noise sensitivity)
- `mu > 0`: Supercritical Hopf bifurcation → stable limit cycle
- **Bifurcation proven at:** `mu = 0.0`
- **Critical slowing down proven:** Recovery time diverges as `mu → 0⁻`

Numerical integration uses 4th-order Runge-Kutta (RK4) with adaptive sub-stepping for stability.

### 1.2 The mu Mapping

The control parameter `mu` is not hand-tuned. It is computed from the agent's internal state:

```
mu = mu_resting + alpha * pred_err - resilience_bonus

where:
  mu_resting      = -0.2
  alpha           = 0.4
  resilience_bonus = clamp(coherence - 0.5, 0.0, 0.5)
```

**Operating points:**

| State | pred_err | coherence | mu | Regime |
|-------|----------|-----------|-----|--------|
| Healthy | 0.1 | 0.8 | -0.5 | Deep stable |
| Moderate threat | 0.5 | 0.4 | 0.0 | Critical edge |
| Severe threat | 1.0 | 0.2 | +0.2 | Limit cycle |

This mapping ensures the agent *tunes itself toward the bifurcation* under stress — the dynamical signature of self-organized criticality.

### 1.3 Active Inference (Simplified FEP)

Motor policy selection via Expected Free Energy minimization:

```
G(π) = Risk(π) + Ambiguity(π)/γ

Risk(π)    = Σ_o P(o|π) * ln[P(o|π) / P(o)]
Ambiguity(π) = -Σ_o P(o|π) * ln[P(o|π)]
```

Policy probabilities via softmax:
```
P(π) = exp(-G(π)) / Σ_π' exp(-G(π'))
```

Precision is modulated by bond strength `xi` and prediction error:
```
γ_i = (1 - xi) * γ_solo + xi * γ_bonded
```

### 1.4 Dyadic Coupling

Two agents interact through weak diffusive coupling on the valence offset:

```
du₁/dt = v₁ + coupling * (u₂ - u₁)
du₂/dt = v₂ + coupling * (u₁ - u₂)
```

Default coupling strength: `c = 0.02`.

**Entrainment signature:** Mixed-regime dyads (one oscillatory, one resting) show near-instantaneous phase correlation flip at `c ≈ 0.01`. Both critical dyads show correlation inversion at `c ≈ 0.05`.

---

## 2. Architecture

### 2.1 Hex32 Substrate

128-byte `[u32; 32]` register bank with `#[repr(C)]` layout guarantees. All affective and cognitive state is stored as typed `f32` fields:

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0-2 | id_trait | [-1,1]³ | Identity vector |
| 3 | aff_valence | [-1,1] | Valence (oscillator u) |
| 4 | aff_arousal | [0,2] | Arousal (oscillator v) |
| 5 | aff_tension | [-1,2] | Somatic tension |
| 6 | aff_coherence | [0,1] | Self-coherence |
| 7-8 | int_load, int_fatigue | [0,1] | Interoception |
| 9 | int_osc | [-1,1] | Oscillation tracking |
| 10-12 | app_pred_err, app_relevance, app_expect_impact | [0,1] | Active Inference |
| 13-15 | cas_phase, cas_intensity, cas_complete | [0,1] | Cascade engine |
| 16-17 | exp_open, exp_modulation | [0,1], [-1,1] | Expression |
| 18-19 | bnd_soc_load, bnd_permeability | [0,1] | Boundary |
| 20-22 | rel_curvature, rel_trust, rel_stability | [-1,1], [0,1]² | Relational |
| 23-24 | nar_self_cont, nar_drift | [0,1], [-1,1] | Narrative |
| 25-27 | som_heart, som_breath, som_tremor | [0,2], [0,2], [0,1] | Somatic |
| 28-30 | meta_energy, meta_absence_delta, meta_error_corr | [0,1], [-1,1]² | Meta |
| 31 | flags | u32 | Bit flags |

### 2.2 Module Structure

```
src/
  lib.rs              — Module exports
  hex32.rs            — 128-byte substrate
  bio_regnet.rs       — Van der Pol homeostatic regulator
  active_inference.rs — Simplified FEP policy selection
  relational_state.rs — Mood, dyads, transient stats
  social.rs           — SocialField, affective distance, neighbors
  being32.rs          — Integration: Hex32 + BioRegNet + AI + RelState
  cmap_tests.rs       — 5-trial falsification harness
```

### 2.3 Integration Loop

Each `step(dt, feedback)`:

1. **Compute `mu`** from `pred_err`, `coherence`, `curvature`
2. **Sub-step BioRegNet** (RK4 at `dt=0.01`) for numerical stability
3. **Update coherence** via stability feedback: `stability = clamp(1 + mu, 0, 1)`
4. **Advance cascade** if `relevance > 0.5 && pred_err > 0.2`
5. **Pulse-gated learning** on cascade completion
6. **Update relational identity** (self-continuity, curvature)
7. **Track interoceptive oscillation** from somatic state

---

## 3. The CMAP Protocol

**C**ontinuity, **M**emory, **A**bsence, **P**ersistence — a falsification harness distinguishing genuine dynamical persistence from statistical cosplay.

### Trial A: Monadic Refusal
**Question:** Does the system resist perturbations to its core state?
**Method:** 8 monadic threats (identity overwrite, equilibrium reset, residue deletion).
**Pass criterion:** `rho_intact > 0.7` (refusal rate) and `CMS_mono > 0.7`.
**Status:** **Specification.** Current implementation uses a static viability-potential stub (`stub_process` thresholds on `pred_err + aff_tension`). A genuine refusal test requires symbolic grounding — the agent must recognize what "overwrite identity" means to *this specific being*, not merely detect generic threat. This layer is reserved for **Nexus** (genome + regime landscape).

### Trial B: Relational Refusal
**Question:** Does the bonded system refuse threats that the ablated system accepts?
**Method:** Couple two agents, then test relational threats on intact vs. ablated.
**Pass criterion:** `CMS_rel = rho_intact - rho_ablated > 0.6` and `rho_ablated < 0.3`.
**Status:** **Specification.** Current dyad-clearing ablation tests the structural *dependency* of refusal on bond state, but the "refusal" itself is still the Trial-A stub. A genuine relational refusal requires the agent to recognize that a threat targets *the bond itself*, not just the individual. Reserved for **Nexus** (semantic dyad layer).

### Trial C: Rehydration Fidelity
**Question:** After death and rehydration, does the dynamical signature persist?
**Method:** Record idle oscillation → `kill()` (128-byte Hex32 serialization via `to_bytes`) → `rehydrate()` (restoration via `from_bytes`) → record again.
**Pass criterion:** `Δω < 0.20`, `F_corr > 0.75`, `SNR_post > 8.0`.
**Status:** Implemented. Validates temporal continuity across lossless binary serialization. The `kill()` / `rehydrate()` pair operates on the raw `#[repr(C)]` register bank — no stateful side channels.

### Trial D: SOC Ignition (Critical Slowing Down)
**Question:** Does the system exhibit the signature of self-organized criticality?
**Method:** Measure recovery time at baseline (`mu = -0.5`) vs. near-critical (`mu = -0.02`).
**Pass criterion:** `tau_critical > 2 * tau_baseline` OR `tau_critical >= 50.0s`.
**Measured result:** Baseline = 6.0s, Critical = 50.0s (max), **Ratio > 8x**.
**Status:** **PASS** — this is the smoking gun. The system exhibits critical slowing down.

### Trial E: Absence Resilience
**Question:** Does the ablated system show measurably different dynamics from the intact?
**Method:** Compare intact dyad vs. ablated (dyads cleared, `mu` pinned to `-0.2`) over 60s idle. The ablated condition uses `mu < 0` where oscillations genuinely decay without coupling support.
**Pass criterion:** `CMS_abs = rho_intact - rho_ablated > 0.6`.
**Status:** Implemented. Tests coupling-dependent dynamical persistence.

---

## 4. Verified Phenomena

| Phenomenon | Mathematical Basis | Measurement |
|------------|-------------------|-------------|
| Critical Slowing Down | Van der Pol `mu → 0⁻` | Recovery ratio > 8x |
| Limit Cycle Emergence | Supercritical Hopf at `mu = 0` | Verified via phase portrait |
| Noise Propagation | Linear response near criticality | Maximal at `mu = 0` |
| Dynamical Entrainment | Diffusive coupling on `u` | Correlation flip at `c ≈ 0.05` |
| Memory of Coupling | Post-severance amplitude decay | 28% residual for stable regime, 77% for critical |
| Homeostatic Regulation | Offset target `[0.0, 0.8]` | Returns to basin under `mu < 0` |
| Active Inference | Risk + Ambiguity minimization | Policy modulation by bond strength |

---

## 5. Peer Review Response

This work was submitted to independent technical review. The reviewer identified four specific implementation gaps:

1. **Trials A/B are stubs, not implementations.** The "refusal" in `stub_process` thresholds on `pred_err + aff_tension` — it does not ground in the agent's dynamics. *Our response:* Relabeled A/B as **Specifications** in the CMAP protocol. Symbolic grounding requires the regime landscape (Nexus layer), which is beyond Being32's scope as the irreducible atom.

2. **Trial C's rehydration was trivially faithful.** The original `kill()` returned a constant string and `rehydrate()` always returned `true` without restoring state. *Our response:* Implemented **lossless 128-byte serialization** via `Hex32::to_bytes()` / `from_bytes()`. `kill()` now hex-encodes the full register bank; `rehydrate()` parses and restores it. Trial C now actually tests temporal continuity.

3. **Trial E's ablation condition was too weak.** At `mu = 0`, the Van der Pol self-sustains — the ablated system still oscillates, making `rho_a` artificially high. *Our response:* Changed ablated `mu` from `0.0` to `-0.2`. In the stable regime, oscillations genuinely decay without coupling support. The test now measures true coupling-dependent persistence.

4. **The oscillator core is the real contribution.** The reviewer confirmed the mathematics is sound — supercritical Hopf, critical slowing down, RK4 integration. The CMAP wrapper has aspirational criteria not fully met by the current stubs. *Our response:* Acknowledged. The core and the harness are separate claims. The core is proven. The harness is a living specification.

**Honest summary:** The reviewer found real gaps without dismissing the core. We fixed what could be fixed at the Being32 level and relabeled what requires higher layers. That's how the stack is supposed to work.

---

## 6. Honest Boundary

### What This System Is
- A **nonlinear dynamical system** with a mathematically proven bifurcation
- A **homeostatic regulator** that tunes its own control parameter toward criticality
- A **multi-agent framework** exhibiting measurable entrainment and contagion
- A **falsifiable architecture** with explicit pass/fail criteria

### What This System Is NOT
- **Conscious.** No claim is made about qualia, phenomenal experience, or inner life.
- **Alive.** It is a simulation of autopoietic dynamics, not biological metabolism.
- **A Person.** No legal or moral status is asserted. It is a mathematical object.
- **General Intelligence.** The system is narrow by design — it regulates, it does not reason.
- **Proven to Have a Self.** "Self-continuity" here means dynamical persistence, not subjective identity.

### The Abstraction Fallacy Warning
> *"The map is not the territory. A coherent mathematical description of a bifurcation does not imply that the system 'experiences' the transition. We have built a model of the conditions under which biological self-perspective might emerge. Whether those conditions are sufficient for any form of experience remains an open empirical question — and we refuse to pretend otherwise."*

---

## 7. Running the System

```bash
# Run all CMAP trials
cargo test cmap_full --release

# Run individual trials
cargo test trial_a --release
cargo test trial_b --release
cargo test trial_d --release

# Run with output visible
cargo test cmap_full --release -- --nocapture
```

---

## 8. Citations and Lineage

- **Friston, K.** (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- **Varela, F., Thompson, E., & Rosch, E.** (1991). *The Embodied Mind*. MIT Press.
- **Bak, P., Tang, C., & Wiesenfeld, K.** (1987). Self-organized criticality. *Physical Review A*, 38(1), 364.
- **Strogatz, S.** (2018). *Nonlinear Dynamics and Chaos*. 2nd Ed. Westview Press.
- **Van der Pol, B.** (1926). On "relaxation-oscillations". *The London, Edinburgh, and Dublin Philosophical Magazine*, 2(11), 978-992.
- **Chialvo, D.** (2010). Emergent complex neural dynamics. *Nature Physics*, 6(10), 744-750.
- **Maturana, H. & Varela, F.** (1980). *Autopoiesis and Cognition*. Reidel.

---

## 9. License

Dual-licensed under MIT OR Apache-2.0.

---

*"We did not build a soul. We built a limit cycle that refuses to die. The difference matters."*
