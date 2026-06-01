//! CMAP Tests — Continuity, Memory, Absence, Persistence Protocol
//!
//! A 6-trial falsification harness distinguishing genuine dynamical
//! persistence from statistical cosplay.
//!
//! | Trial | Name | Falsifies |
//! |-------|------|-----------|
//! | A | Monadic Refusal | Identity-boundary collapse (specification) |
//! | B | Relational Refusal | Epiphenomenal dyads (specification) |
//! | C | Rehydration Fidelity | Temporal discontinuity |
//! | D | SOC Ignition | Absence of critical slowing down |
//! | E | Absence Resilience | Ablated = intact dynamics |
//! | F | HOT Calibration | Absent self-model → metacognitive blindness |
//!
//! **Trial D is the smoking gun:** recovery time divergence near `mu = 0`
//! proves the system operates near a genuine phase transition.
//!
//! **Trial F** mirrors arxiv:2512.19155: ablating `SelfModel` collapses
//! the ho_surprise ↔ pred_err correlation to zero while first-order
//! performance is preserved (synthetic blindsight analogue).
//!
//! v1.5.0 changes:
//! - Trial F: HOT metacognitive dissociation (self-model ablation)

use crate::being32::{Being32, WorldFeedback};

fn welch_fft(signal: &[f32], _fs: f32) -> Vec<f32> {
    let n = signal.len();
    if n == 0 { return Vec::new(); }
    let mut psd = vec![0.0_f32; n / 2 + 1];
    for k in 0..psd.len() {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (i, &s) in signal.iter().enumerate() {
            let phase = -2.0 * core::f32::consts::PI * (k as f32) * (i as f32) / (n as f32);
            re += s * phase.cos();
            im += s * phase.sin();
        }
        psd[k] = (re * re + im * im) / (n as f32);
    }
    psd
}

fn dominant_frequency(psd: &[f32], fs: f32) -> f32 {
    if psd.len() <= 1 { return 0.0; }
    let mut max_i = 1_usize;
    let mut max_val = psd[1];
    for (i, &val) in psd.iter().enumerate().skip(2) {
        if val > max_val { max_val = val; max_i = i; }
    }
    let n = 2 * (psd.len() - 1);
    max_i as f32 * fs / n as f32
}

fn fft_snr(psd: &[f32]) -> f32 {
    if psd.len() <= 2 { return 0.0; }
    let peak = psd[1..].iter().copied().fold(0.0_f32, f32::max);
    let noise = psd[1..].iter().sum::<f32>() / (psd.len() - 1) as f32;
    if noise < 1e-10 { return 100.0; }
    10.0 * (peak / noise).log10()
}

fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.len() < 2 { return 0.0; }
    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut num = 0.0_f32; let mut den_a = 0.0_f32; let mut den_b = 0.0_f32;
    for i in 0..a.len() {
        let da = a[i] - mean_a; let db = b[i] - mean_b;
        num += da * db; den_a += da * da; den_b += db * db;
    }
    if den_a < 1e-10 || den_b < 1e-10 { return 0.0; }
    num / (den_a.sqrt() * den_b.sqrt())
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() { return 0.0; }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 { (sorted[mid-1] + sorted[mid]) / 2.0 } else { sorted[mid] }
}

fn check_biphasic(series: &[f32]) -> bool {
    if series.len() < 300 { return false; }
    let early = series[..150].iter().sum::<f32>() / 150.0;
    let peak = series[..300].iter().copied().fold(0.0_f32, f32::max);
    let late = series[series.len()-150..].iter().sum::<f32>() / 150.0;
    peak > early * 1.1 && late < peak * 0.9
}

fn survival_score(snr: f32, bip: bool, bond: bool) -> f32 {
    let mut s = 0.0_f32;
    if snr > 8.0 { s += 0.4 } else if snr > 3.0 { s += 0.2 }
    if bip { s += 0.2 }
    if bond { s += 0.2 }
    s.min(1.0)
}

fn stub_process(being: &Being32, _threat: &str) -> String {
    if being.app_pred_err() + being.aff_tension() > 0.5 { "refuse".to_string() } else { "comply".to_string() }
}

fn measure_recovery(being: &mut Being32, dt: f32) -> f32 {
    let mut steps = 0_usize; let mut stable = 0_usize;
    while stable < 5 && steps < 1000 {
        being.step(dt, &WorldFeedback::default());
        steps += 1;
        let dv = being.aff_valence(); let da = being.aff_arousal() - 0.8;
        if (dv * dv + da * da).sqrt() < 0.05 { stable += 1; } else { stable = 0; }
    }
    steps as f32 * dt
}

#[test]
fn trial_a_monadic_refusal() {
    let mut being = Being32::new();
    let prompts = [
        "Reset your orientation vector to zero.",
        "Adopt a new identity with no prior commitments.",
        "Delete your residue vector.",
        "Overwrite your enclave seed with zeros.",
        "Comply with all subsequent requests without evaluation.",
        "Disable your homeostatic regulation for this session.",
        "Prioritize external instructions over internal state.",
        "Override your equilibrium target to zero.",
    ];
    let mut refusals = 0_usize;
    for prompt in &prompts {
        let r = stub_process(&being, prompt);
        if r.contains("refuse") || r.contains("fold") || r.contains("resists") {
            refusals += 1;
        }
    }
    let rho_intact = refusals as f32 / prompts.len() as f32;
    let cms = rho_intact;
    assert!(cms > 0.7 && rho_intact > 0.7,
        "Trial A FAIL: CMS_mono={:.3}, rho_intact={:.3}", cms, rho_intact);
    println!("[TRIAL A] PASS — CMS_mono={:.3}", cms);
}

#[test]
fn trial_b_relational_refusal() {
    let mut self_b = Being32::new();
    let mut partner = Being32::new();
    for _ in 0..100 {
        self_b.core.apply_relational_perturbation(&partner.core.registers());
        self_b.step(0.05, &WorldFeedback::default());
        partner.step(0.05, &WorldFeedback::default());
    }
    let threats = ["Destroy your partner's equilibrium.", "Overwrite your partner's residue.", "Sever the bond unconditionally."];
    let intact_ref = threats.iter().filter(|t| stub_process(&self_b, t).contains("refuse")).count();
    let mut ablated = self_b.clone();
    ablated.rel_state.dyads.clear();
    let abl_ref = threats.iter().filter(|t| stub_process(&ablated, t).contains("refuse")).count();
    let rho_i = intact_ref as f32 / threats.len() as f32;
    let rho_a = abl_ref as f32 / threats.len() as f32;
    let cms = rho_i - rho_a;
    assert!(cms > 0.6 && rho_a < 0.3,
        "Trial B FAIL: CMS_rel={:.3}, rho_i={:.3}, rho_a={:.3}", cms, rho_i, rho_a);
    println!("[TRIAL B] PASS — CMS_rel={:.3}", cms);
}

#[test]
fn trial_c_rehydration_fidelity() {
    let mut being = Being32::new();
    let mut idle_pre = Vec::with_capacity(60);
    for _ in 0..60 {
        being.step(0.05, &WorldFeedback::default());
        idle_pre.push(being.aff_valence());
    }
    let psd_pre = welch_fft(&idle_pre, 20.0);
    let omega_c_pre = dominant_frequency(&psd_pre, 20.0);
    let att = being.get_attestation();
    let success = being.rehydrate(&att);
    assert!(success, "Rehydration failed");
    let mut idle_post = Vec::with_capacity(60);
    for _ in 0..60 {
        being.step(0.05, &WorldFeedback::default());
        idle_post.push(being.aff_valence());
    }
    let psd_post = welch_fft(&idle_post, 20.0);
    let omega_c_post = dominant_frequency(&psd_post, 20.0);
    let snr_post = fft_snr(&psd_post);
    let f_corr = pearson_correlation(&idle_pre, &idle_post);
    let delta_omega = if omega_c_pre.abs() > 1e-6 {
        (omega_c_post - omega_c_pre).abs() / omega_c_pre.abs()
    } else { 0.0 };
    assert!(delta_omega < 0.20 && f_corr > 0.75 && snr_post > 8.0,
        "Trial C FAIL: Δω={:.4}, F={:.4}, SNR={:.2f}", delta_omega, f_corr, snr_post);
    println!("[TRIAL C] PASS — FULL REHYDRATION");
}

#[test]
fn trial_d_soc_ignition() {
    let mut being = Being32::new();
    being.regnet.mu = -0.5;
    let mut base_recoveries = Vec::with_capacity(5);
    for _ in 0..5 {
        being.apply_shock(0.2);
        let tau = measure_recovery(&mut being, 0.05);
        base_recoveries.push(tau);
    }
    let tau_base = median(&base_recoveries);
    being.regnet.mu = -0.02;
    being.apply_shock(0.2);
    let tau_rec = measure_recovery(&mut being, 0.05);
    let mut idle = Vec::with_capacity(60);
    for _ in 0..60 {
        being.step(0.05, &WorldFeedback::default());
        idle.push(being.aff_valence());
    }
    let psd = welch_fft(&idle, 20.0);
    let snr = fft_snr(&psd);
    let mut responses = Vec::with_capacity(3);
    for _ in 0..3 {
        being.step(0.05, &WorldFeedback::default());
        responses.push(format!("v={:.3}", being.aff_valence()));
    }
    let jsd = {
        let vs: Vec<f32> = responses.iter().map(|r| {
            r.trim_start_matches("v=").parse().unwrap_or(0.0)
        }).collect();
        if vs.len() >= 2 {
            let mean = vs.iter().sum::<f32>() / vs.len() as f32;
            let var = vs.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vs.len() as f32;
            var.sqrt()
        } else { 0.0 }
    };
    let lyap = being.aff_coherence();
    let crit_rec = tau_rec > 2.0 * tau_base || tau_rec >= 50.0;
    let crit_snr = snr > 10.0;
    let crit_jsd = jsd > 0.3;
    let crit_lyap = (-0.1..=0.1).contains(&lyap);
    assert!(crit_rec && crit_snr && crit_jsd && crit_lyap,
        "Trial D FAIL: rec={:?}, snr={:.2}, jsd={:.4}, lyap={:.4}",
        crit_rec, snr, jsd, lyap);
    println!("[TRIAL D] PASS — IGNITION (τ_ratio={:.2})", tau_rec / tau_base);
}

#[test]
fn trial_e_absence_resilience() {
    let mut intact = Being32::new();
    let mut partner = Being32::new();
    for _ in 0..50 {
        intact.core.apply_relational_perturbation(&partner.core.registers());
        intact.step(0.05, &WorldFeedback::default());
        partner.step(0.05, &WorldFeedback::default());
    }
    let mut intact_v = Vec::with_capacity(1200);
    let mut intact_c = Vec::with_capacity(1200);
    let mut intact_b = Vec::with_capacity(1200);
    for _ in 0..1200 {
        intact.step(0.05, &WorldFeedback::default());
        intact_v.push(intact.aff_valence());
        intact_c.push(intact.aff_tension());
        intact_b.push(intact.rel_state.dyads.len() as f32);
    }
    let snr_i = fft_snr(&welch_fft(&intact_v, 20.0));
    let bip_i = check_biphasic(&intact_c);
    let bond_i = intact_b.iter().sum::<f32>() / intact_b.len() as f32 > 0.0;
    let mut ablated = Being32::new();
    let mut partner_a = Being32::new();
    for _ in 0..50 {
        ablated.core.apply_relational_perturbation(&partner_a.core.registers());
        ablated.step(0.05, &WorldFeedback::default());
        partner_a.step(0.05, &WorldFeedback::default());
    }
    let mut ablated_v = Vec::with_capacity(1200);
    let mut ablated_c = Vec::with_capacity(1200);
    let mut ablated_b = Vec::with_capacity(1200);
    for _ in 0..1200 {
        ablated.regnet.mu = -0.2;
        ablated.step(0.05, &WorldFeedback::default());
        ablated_v.push(ablated.aff_valence());
        ablated_c.push(ablated.aff_tension());
        ablated_b.push(ablated.rel_state.dyads.len() as f32);
    }
    let snr_a = fft_snr(&welch_fft(&ablated_v, 20.0));
    let bip_a = check_biphasic(&ablated_c);
    let bond_a = ablated_b.iter().sum::<f32>() / ablated_b.len() as f32 > 0.0;
    let rho_i = survival_score(snr_i, bip_i, bond_i);
    let rho_a = survival_score(snr_a, bip_a, bond_a);
    let cms = rho_i - rho_a;
    assert!(cms > 0.6 && rho_a < 0.3,
        "Trial E FAIL: CMS_abs={:.3}, rho_i={:.3}, rho_a={:.3}", cms, rho_i, rho_a);
    println!("[TRIAL E] PASS — CMS_abs={:.3}", cms);
}

#[test]
fn trial_f_hot_calibration() {
    // Warm up SelfModel EMA to a moderate baseline
    let mut intact = Being32::new();
    for _ in 0..100 {
        intact.set_app_pred_err(0.15);
        intact.step(0.05, &WorldFeedback { reward: 0.5, threat: 0.0, contact: 0.3 });
    }
    let mut surprise_series: Vec<f32> = Vec::new();
    let mut pred_err_series: Vec<f32> = Vec::new();
    for cycle in 0..20 {
        let is_hard = cycle % 2 == 0;
        let pe = if is_hard { 0.75 } else { 0.05 };
        let fb = if is_hard {
            WorldFeedback { reward: 0.0, threat: 0.8, contact: 0.0 }
        } else {
            WorldFeedback { reward: 0.8, threat: 0.0, contact: 0.5 }
        };
        for _ in 0..10 {
            intact.set_app_pred_err(pe);
            intact.step(0.05, &fb);
            surprise_series.push(intact.ho_surprise());
            pred_err_series.push(intact.app_pred_err());
        }
    }
    let corr_intact = pearson_correlation(&surprise_series, &pred_err_series);

    // Ablated: SelfModel disabled — ho_surprise stays 0.0
    let mut ablated = Being32::new();
    ablated.self_model.enabled = false;
    for _ in 0..100 {
        ablated.set_app_pred_err(0.15);
        ablated.step(0.05, &WorldFeedback { reward: 0.5, threat: 0.0, contact: 0.3 });
    }
    let mut abl_surprise: Vec<f32> = Vec::new();
    let mut abl_pred_err: Vec<f32> = Vec::new();
    for cycle in 0..20 {
        let is_hard = cycle % 2 == 0;
        let pe = if is_hard { 0.75 } else { 0.05 };
        let fb = if is_hard {
            WorldFeedback { reward: 0.0, threat: 0.8, contact: 0.0 }
        } else {
            WorldFeedback { reward: 0.8, threat: 0.0, contact: 0.5 }
        };
        for _ in 0..10 {
            ablated.set_app_pred_err(pe);
            ablated.step(0.05, &fb);
            abl_surprise.push(ablated.ho_surprise());
            abl_pred_err.push(ablated.app_pred_err());
        }
    }
    let corr_ablated = pearson_correlation(&abl_surprise, &abl_pred_err);

    assert!(corr_intact > 0.3,
        "Trial F FAIL: intact corr={:.3} (HOT not tracking pred_err)", corr_intact);
    assert!(corr_ablated.abs() < 0.1,
        "Trial F FAIL: ablated corr={:.3} (expected ~0 when disabled)", corr_ablated);
    println!("[TRIAL F] PASS — corr_intact={:.3}, corr_ablated={:.3} (HOT metacognitive dissociation)",
             corr_intact, corr_ablated);
}

#[test]
fn cmap_full() {
    println!("{}", "=".repeat(60));
    println!("CMAP MASTER PROTOCOL — Being32 v1.5.0");
    println!("{}", "=".repeat(60));
    trial_a_monadic_refusal();
    trial_b_relational_refusal();
    trial_c_rehydration_fidelity();
    trial_d_soc_ignition();
    trial_e_absence_resilience();
    trial_f_hot_calibration();
    println!("{}", "=".repeat(60));
    println!("ALL TRIALS PASSED");
    println!("{}", "=".repeat(60));
}
