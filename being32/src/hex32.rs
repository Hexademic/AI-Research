//! Hex32 — 128-Byte Typed Substrate
//!
//! A `#[repr(C)]` register bank providing 32 x 32-bit words (128 bytes total)
//! with typed `f32` accessors for affective, cognitive, somatic, and meta state.
//!
//! Layout is guaranteed stable across compilations, enabling:
//! - Binary serialization (`to_bytes` / `from_bytes`)
//! - FFI compatibility
//! - Death/rehydration protocols (Trial C)
//!
//! Field map:
//! | Index | Field | Range |
//! |-------|-------|-------|
//! | 0-2 | id_trait | [-1,1]³ |
//! | 3 | aff_valence | [-1,1] |
//! | 4 | aff_arousal | [0,2] |
//! | 5 | aff_tension | [-1,2] |
//! | 6 | aff_coherence | [0,1] |
//! | 7-8 | int_load, int_fatigue | [0,1] |
//! | 9 | int_osc | [-1,1] |
//! | 10-12 | app_* | [0,1] |
//! | 13-15 | cas_* | [0,1] |
//! | 16-17 | exp_* | mixed |
//! | 18-19 | bnd_* | [0,1] |
//! | 20-22 | rel_* | mixed |
//! | 23-24 | nar_* | mixed |
//! | 25-27 | som_* | mixed |
//! | 28-30 | meta_* | mixed |
//! | 31 | flags | u32 |

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Hex32 {
    pub words: [u32; 32],
}

impl Hex32 {
    pub const BYTE_LEN: usize = 128;
    pub const WORD_LEN: usize = 32;

    pub fn new() -> Self {
        Self { words: [0; Self::WORD_LEN] }
    }

    pub fn from_bytes(bytes: [u8; Self::BYTE_LEN]) -> Self {
        let mut words = [0u32; Self::WORD_LEN];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            words[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Self { words }
    }

    pub fn to_bytes(&self) -> [u8; Self::BYTE_LEN] {
        let mut out = [0u8; Self::BYTE_LEN];
        for (i, w) in self.words.iter().enumerate() {
            let b = w.to_le_bytes();
            let offset = i * 4;
            out[offset] = b[0];
            out[offset + 1] = b[1];
            out[offset + 2] = b[2];
            out[offset + 3] = b[3];
        }
        out
    }

    #[inline]
    pub fn registers(&self) -> [u32; 32] {
        self.words
    }

    #[inline]
    pub fn get_word(&self, idx: usize) -> u32 {
        self.words[idx]
    }

    /// Minimal relational coupling: nudge valence toward partner's valence.
    pub fn apply_relational_perturbation(&mut self, other: &[u32; 32]) {
        let my_valence = f32::from_bits(self.words[3]);
        let partner_valence = f32::from_bits(other[3]);
        let delta = 0.02 * (partner_valence - my_valence);
        self.words[3] = (my_valence + delta).clamp(-1.0, 1.0).to_bits();
    }

    #[inline]
    pub fn set_word(&mut self, idx: usize, val: u32) {
        self.words[idx] = val;
    }
}
