//! plato-sonar-text — Acoustic scene analysis with text output for agents.
//!
//! This is the text interface layer that sits on top of sonar-vision-rs's
//! signal processing, providing high-level acoustic scene descriptions,
//! sound event detection, voice activity detection, and alerting.

use std::fmt;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Text description of an acoustic environment.
#[derive(Debug, Clone)]
pub struct AcousticScene {
    /// Human-readable summary of the acoustic environment.
    pub summary: String,
    /// Overall sound level in decibels.
    pub sound_level_db: f64,
    /// Dominant frequency in Hz.
    pub dominant_frequency: f64,
    /// Detected sound events.
    pub events: Vec<SoundEvent>,
    /// Whether voice activity is present.
    pub voice_active: bool,
}

/// A detected sound event.
#[derive(Debug, Clone)]
pub struct SoundEvent {
    /// Label (e.g. "sine", "noise", "silence", "speech-like").
    pub label: String,
    /// Detection confidence [0, 1].
    pub confidence: f64,
    /// Start time in seconds relative to analysis start.
    pub start_time: f64,
    /// Duration in seconds.
    pub duration: f64,
    /// Estimated direction in degrees (0 = front, 90 = right, etc.).
    pub direction: f64,
    /// Frequency range as (low_hz, high_hz).
    pub frequency_range: (f64, f64),
}

/// Voice activity detection result.
#[derive(Debug, Clone)]
pub struct VoiceActivity {
    /// Whether speech was detected.
    pub is_speech: bool,
    /// Duration of the voice segment in seconds.
    pub duration: f64,
    /// Energy level [0, 1].
    pub energy_level: f64,
    /// Estimated direction in degrees.
    pub direction: f64,
}

/// Baseline ambient noise characteristics.
#[derive(Debug, Clone)]
pub struct AmbientProfile {
    /// Average sound level in dB.
    pub avg_db: f64,
    /// Peak sound level in dB.
    pub peak_db: f64,
    /// Summary of the frequency spectrum.
    pub frequency_spectrum_summary: String,
    /// When the baseline was last updated (epoch seconds).
    pub baseline_updated: u64,
}

/// A significant sound event worth alerting the agent.
#[derive(Debug, Clone)]
pub struct AcousticAlert {
    /// Alert severity (0 = info, 1 = warning, 2 = critical).
    pub severity: u8,
    /// Human-readable description.
    pub description: String,
    /// Type of sound that triggered the alert.
    pub sound_type: String,
    /// Estimated direction in degrees.
    pub direction: f64,
}

impl fmt::Display for AcousticAlert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sev = match self.severity {
            0 => "INFO",
            1 => "WARNING",
            _ => "CRITICAL",
        };
        write!(
            f,
            "[{}] {} (type: {}, dir: {:.0}°)",
            sev, self.description, self.sound_type, self.direction
        )
    }
}

// ---------------------------------------------------------------------------
// DirectionEstimator — helper for estimating direction from stereo samples
// ---------------------------------------------------------------------------

/// Estimates direction of sound sources from inter-channel differences.
pub struct DirectionEstimator;

impl DirectionEstimator {
    /// Estimate direction from left/right channel energy difference.
    ///
    /// `left` and `right` are mono sample slices for the same time window.
    /// Returns an angle in degrees: 0 = front, positive = right, negative = left.
    pub fn estimate(left: &[f64], right: &[f64]) -> f64 {
        let energy_left: f64 = left.iter().map(|s| s * s).sum::<f64>() / left.len().max(1) as f64;
        let energy_right: f64 = right.iter().map(|s| s * s).sum::<f64>() / right.len().max(1) as f64;
        let total = energy_left + energy_right;
        if total < 1e-12 {
            return 0.0;
        }
        // Map difference to angle: fully right → +90°, fully left → -90°
        let balance = (energy_right - energy_left) / total;
        balance * 90.0
    }
}

// ---------------------------------------------------------------------------
// SonarText — the main interface
// ---------------------------------------------------------------------------

/// The main acoustic-to-text processor.
pub struct SonarText {
    /// Accumulated ambient baseline.
    ambient: Option<AmbientProfile>,
    /// Frame counter for timing.
    frame_count: u64,
    /// Epoch offset for start_time calculations.
    start_instant: Instant,
    /// Accumulated frames for baseline building.
    baseline_frames: Vec<Vec<f64>>,
}

impl SonarText {
    /// Create a new SonarText processor.
    pub fn new() -> Self {
        Self {
            ambient: None,
            frame_count: 0,
            start_instant: Instant::now(),
            baseline_frames: Vec::new(),
        }
    }

    fn elapsed_secs(&self) -> f64 {
        self.start_instant.elapsed().as_secs_f64()
    }

    // -- Internal DSP helpers ------------------------------------------------

    /// Compute RMS of samples.
    fn rms(samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f64).sqrt()
    }

    /// Convert RMS to approximate dB (relative to full scale).
    fn rms_to_db(rms: f64) -> f64 {
        if rms < 1e-12 {
            return -96.0;
        }
        20.0 * rms.log10()
    }

    /// Simple zero-crossing rate frequency estimate.
    fn estimate_frequency(samples: &[f64], sample_rate: f64) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0u32;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        // Each period has 2 zero crossings
        (crossings as f64 / 2.0) * sample_rate / samples.len() as f64
    }

    /// Classify a frame into a sound label based on features.
    fn classify(rms: f64, frequency: f64, zcr: f64, spectral_flatness: f64) -> (String, f64) {
        let db = Self::rms_to_db(rms);

        // Silence threshold
        if db < -50.0 {
            return ("silence".into(), 0.95);
        }

        // Sine-like: low zero-crossing rate variation (clean periodic signal)
        // Pure sine has very regular zero crossings → low spectral flatness
        if zcr < 0.15 && frequency > 20.0 && rms > 0.01 {
            return ("sine".into(), 0.85);
        }

        // Noise-like: high spectral flatness or high zero-crossing rate
        if spectral_flatness > 0.3 || zcr > 0.3 {
            return ("noise".into(), 0.80);
        }

        // Speech-like: moderate frequency, moderate energy, not already classified
        if rms > 0.005 && frequency > 50.0 && frequency < 600.0 {
            return ("speech-like".into(), 0.7);
        }

        ("unknown".into(), 0.5)
    }

    /// Compute a rough spectral flatness from samples (Geometric mean / Arithmetic mean of energy).
    fn spectral_flatness(samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let n = samples.len();
        // Use magnitude squares
        let vals: Vec<f64> = samples.iter().map(|s| s * s + 1e-20).collect();
        let arithmetic_mean: f64 = vals.iter().sum::<f64>() / n as f64;
        if arithmetic_mean < 1e-20 {
            return 0.0;
        }
        let log_sum: f64 = vals.iter().map(|v| v.ln()).sum();
        let geometric_mean = (log_sum / n as f64).exp();
        (geometric_mean / arithmetic_mean).min(1.0)
    }

    /// Zero-crossing rate.
    fn zero_crossing_rate(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0u32;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f64 / (samples.len() - 1) as f64
    }

    // -- Public API ----------------------------------------------------------

    /// Process an audio frame and return a full acoustic scene description.
    ///
    /// `audio_samples` is interleaved stereo: [L0, R0, L1, R1, ...].
    pub fn process_frame(&mut self, audio_samples: &[f64]) -> AcousticScene {
        let events = self.detect_events(audio_samples);
        let voice = self.detect_voice(audio_samples);

        let rms = Self::rms(audio_samples);
        let db = Self::rms_to_db(rms);
        let sample_rate = 44100.0; // assumed
        let freq = Self::estimate_frequency(audio_samples, sample_rate);

        let summary = self.build_summary(db, &events, voice.is_speech);

        self.frame_count += 1;

        AcousticScene {
            summary,
            sound_level_db: db,
            dominant_frequency: freq,
            events,
            voice_active: voice.is_speech,
        }
    }

    /// Detect sound events in a frame.
    pub fn detect_events(&self, audio_samples: &[f64]) -> Vec<SoundEvent> {
        if audio_samples.is_empty() {
            return vec![];
        }

        // Split into mono for analysis
        let mono: Vec<f64> = audio_samples.iter().step_by(2).copied().collect();
        let rms = Self::rms(&mono);
        let freq = Self::estimate_frequency(&mono, 44100.0);
        let zcr = Self::zero_crossing_rate(&mono);
        let sf = Self::spectral_flatness(&mono);

        let (label, confidence) = Self::classify(rms, freq, zcr, sf);

        // Estimate direction from stereo
        let left: Vec<f64> = audio_samples.iter().step_by(2).copied().collect();
        let right: Vec<f64> = audio_samples.iter().skip(1).step_by(2).copied().collect();
        let direction = DirectionEstimator::estimate(&left, &right);

        let duration = mono.len() as f64 / 44100.0;

        vec![SoundEvent {
            label,
            confidence,
            start_time: self.elapsed_secs(),
            duration,
            direction,
            frequency_range: (freq * 0.8, freq * 1.2),
        }]
    }

    /// Detect voice activity.
    pub fn detect_voice(&self, audio_samples: &[f64]) -> VoiceActivity {
        let mono: Vec<f64> = audio_samples.iter().step_by(2).copied().collect();
        let rms = Self::rms(&mono);
        let freq = Self::estimate_frequency(&mono, 44100.0);
        let zcr = Self::zero_crossing_rate(&mono);

        // Simple heuristic: voice typically 50-600 Hz, not silence
        // Pure tones have very low ZCR; voice has richer harmonics
        let is_speech = rms > 0.01
            && (
                (freq > 50.0 && freq < 600.0 && zcr > 0.005)
                    || (freq > 80.0 && freq < 400.0 && rms > 0.05)
            );

        let left: Vec<f64> = audio_samples.iter().step_by(2).copied().collect();
        let right: Vec<f64> = audio_samples.iter().skip(1).step_by(2).copied().collect();
        let direction = DirectionEstimator::estimate(&left, &right);
        let duration = mono.len() as f64 / 44100.0;

        VoiceActivity {
            is_speech,
            duration,
            energy_level: (rms * 10.0).min(1.0),
            direction,
        }
    }

    /// Update the ambient baseline from collected frames.
    pub fn update_baseline(&mut self, audio_samples: &[f64]) -> AmbientProfile {
        self.baseline_frames.push(audio_samples.to_vec());

        // Accumulate over up to 32 frames
        let all_mono: Vec<f64> = self
            .baseline_frames
            .iter()
            .flat_map(|f| f.iter().step_by(2).copied())
            .collect();

        let avg_rms = Self::rms(&all_mono);
        let peak = all_mono
            .iter()
            .map(|s| s.abs())
            .fold(0.0f64, |a, b| a.max(b));
        let avg_db = Self::rms_to_db(avg_rms);
        let peak_db = if peak < 1e-12 { -96.0 } else { 20.0 * peak.log10() };

        let freq = Self::estimate_frequency(&all_mono, 44100.0);
        let sf = Self::spectral_flatness(&all_mono);

        let spectrum_summary = if avg_db < -50.0 {
            "very quiet, minimal spectral content".into()
        } else if sf > 0.7 {
            format!("noisy, broadband (flatness: {:.2}), ~{:.0} Hz center", sf, freq)
        } else if freq > 0.0 {
            format!(
                "tonal, center ~{:.0} Hz, flatness {:.2}",
                freq, sf
            )
        } else {
            "flat, no dominant frequency".into()
        };

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let profile = AmbientProfile {
            avg_db,
            peak_db,
            frequency_spectrum_summary: spectrum_summary,
            baseline_updated: ts,
        };

        self.ambient = Some(profile.clone());
        profile
    }

    /// Check for alerts from detected events.
    pub fn check_alerts(&self, events: &[SoundEvent]) -> Vec<AcousticAlert> {
        let mut alerts = Vec::new();

        for ev in events {
            // Loud sounds
            if ev.label == "noise" && ev.confidence > 0.8 {
                alerts.push(AcousticAlert {
                    severity: 1,
                    description: format!("Loud noise detected: {}", ev.label),
                    sound_type: ev.label.clone(),
                    direction: ev.direction,
                });
            }

            // Unusual / unknown sounds with high energy
            if ev.label == "unknown" && ev.confidence > 0.4 {
                alerts.push(AcousticAlert {
                    severity: 0,
                    description: "Unusual sound pattern detected".into(),
                    sound_type: "unknown".into(),
                    direction: ev.direction,
                });
            }

            // Sudden loud events (if baseline exists, compare)
            if let Some(ref ambient) = self.ambient {
                let event_db = Self::rms_to_db((ev.confidence * 0.1).sqrt());
                if event_db > ambient.avg_db + 20.0 {
                    alerts.push(AcousticAlert {
                        severity: 2,
                        description: format!(
                            "Sudden loud event: {} ({} dB above baseline)",
                            ev.label,
                            event_db - ambient.avg_db
                        ),
                        sound_type: ev.label.clone(),
                        direction: ev.direction,
                    });
                }
            }
        }

        alerts
    }

    /// Estimate directions for a list of events using stereo analysis.
    /// Returns (label, angle_degrees) pairs.
    pub fn estimate_direction(&self, events: &[SoundEvent]) -> Vec<(String, f64)> {
        events.iter().map(|ev| (ev.label.clone(), ev.direction)).collect()
    }

    // -- Helpers -------------------------------------------------------------

    fn build_summary(&self, db: f64, events: &[SoundEvent], voice_active: bool) -> String {
        if db < -50.0 {
            return "Quiet environment".into();
        }

        let mut parts = Vec::new();

        if voice_active {
            parts.push("voice detected".into());
        }

        if let Some(event) = events.first() {
            parts.push(format!("dominant sound: {}", event.label));
        }

        let level_desc = if db < -30.0 {
            "low"
        } else if db < -10.0 {
            "moderate"
        } else {
            "high"
        };
        parts.push(format!("sound level: {} ({:.1} dB)", level_desc, db));

        parts.join(", ")
    }
}

impl Default for SonarText {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq: f64, duration_secs: f64, sample_rate: f64) -> Vec<f64> {
        let n = (duration_secs * sample_rate) as usize;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f64 / sample_rate;
            let val = (2.0 * std::f64::consts::PI * freq * t).sin() * 0.5;
            // Stereo: left slightly louder for direction test
            samples.push(val * 0.8);
            samples.push(val * 0.4);
        }
        samples
    }

    fn make_noise(duration_secs: f64, sample_rate: f64) -> Vec<f64> {
        let n = (duration_secs * sample_rate) as usize;
        let mut samples = Vec::with_capacity(n * 2);
        // Pseudo-random using simple LCG
        let mut state: u64 = 42;
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f64 / 2147483648.0 - 1.0) * 0.5;
            samples.push(val);
            samples.push(val * 0.9);
        }
        samples
    }

    fn make_silence(duration_secs: f64, sample_rate: f64) -> Vec<f64> {
        let n = (duration_secs * sample_rate) as usize;
        vec![0.0; n * 2]
    }

    #[test]
    fn test_process_frame_produces_text_summary() {
        let mut sonar = SonarText::new();
        let samples = make_sine(440.0, 0.1, 44100.0);
        let scene = sonar.process_frame(&samples);
        assert!(!scene.summary.is_empty());
        assert!(scene.summary.len() > 5);
    }

    #[test]
    fn test_detect_events_labels_sine() {
        let sonar = SonarText::new();
        let samples = make_sine(440.0, 0.1, 44100.0);
        let events = sonar.detect_events(&samples);
        assert!(!events.is_empty());
        assert_eq!(events[0].label, "sine");
    }

    #[test]
    fn test_detect_events_labels_noise() {
        let sonar = SonarText::new();
        let samples = make_noise(0.1, 44100.0);
        let events = sonar.detect_events(&samples);
        assert!(!events.is_empty());
        assert_eq!(events[0].label, "noise");
    }

    #[test]
    fn test_detect_events_labels_silence() {
        let sonar = SonarText::new();
        let samples = make_silence(0.1, 44100.0);
        let events = sonar.detect_events(&samples);
        assert!(!events.is_empty());
        assert_eq!(events[0].label, "silence");
    }

    #[test]
    fn test_detect_voice_identifies_voice_like() {
        let sonar = SonarText::new();
        // Speech-like: 150 Hz sine with moderate energy
        let samples = make_sine(150.0, 0.1, 44100.0);
        let voice = sonar.detect_voice(&samples);
        assert!(voice.is_speech);
    }

    #[test]
    fn test_detect_voice_rejects_silence() {
        let sonar = SonarText::new();
        let samples = make_silence(0.1, 44100.0);
        let voice = sonar.detect_voice(&samples);
        assert!(!voice.is_speech);
    }

    #[test]
    fn test_update_baseline_accumulates() {
        let mut sonar = SonarText::new();
        let s1 = make_noise(0.05, 44100.0);
        let p1 = sonar.update_baseline(&s1);
        assert!(p1.avg_db > -100.0);

        let s2 = make_silence(0.05, 44100.0);
        let p2 = sonar.update_baseline(&s2);
        // Second frame includes silence, avg should drop
        assert!(p2.avg_db <= p1.avg_db);
    }

    #[test]
    fn test_check_alerts_flags_loud() {
        let sonar = SonarText::new();
        let events = vec![SoundEvent {
            label: "noise".into(),
            confidence: 0.9,
            start_time: 0.0,
            duration: 0.1,
            direction: 0.0,
            frequency_range: (0.0, 20000.0),
        }];
        let alerts = sonar.check_alerts(&events);
        assert!(!alerts.is_empty());
        assert!(alerts.iter().any(|a| a.severity >= 1));
    }

    #[test]
    fn test_check_alerts_quiet_event() {
        let sonar = SonarText::new();
        let events = vec![SoundEvent {
            label: "silence".into(),
            confidence: 0.95,
            start_time: 0.0,
            duration: 0.1,
            direction: 0.0,
            frequency_range: (0.0, 100.0),
        }];
        let alerts = sonar.check_alerts(&events);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_estimate_direction_nonuniform_stereo() {
        let sonar = SonarText::new();
        // Left louder than right → negative angle
        let left = vec![0.8; 100];
        let right = vec![0.2; 100];
        let angle = DirectionEstimator::estimate(&left, &right);
        assert!(angle < 0.0, "Expected negative angle (left-dominant), got {}", angle);

        // Right louder than left → positive angle
        let left = vec![0.2; 100];
        let right = vec![0.8; 100];
        let angle = DirectionEstimator::estimate(&left, &right);
        assert!(angle > 0.0, "Expected positive angle (right-dominant), got {}", angle);
    }

    #[test]
    fn test_ambient_profile_persists_across_frames() {
        let mut sonar = SonarText::new();
        let s1 = make_sine(440.0, 0.05, 44100.0);
        sonar.update_baseline(&s1);

        // Process a different frame
        let s2 = make_noise(0.05, 44100.0);
        sonar.process_frame(&s2);

        // Ambient profile should still be set
        assert!(sonar.ambient.is_some());
        let ambient = sonar.ambient.as_ref().unwrap();
        assert!(ambient.avg_db > -100.0);
        assert!(ambient.baseline_updated > 0);
    }

    #[test]
    fn test_silence_produces_quiet_description() {
        let mut sonar = SonarText::new();
        let samples = make_silence(0.1, 44100.0);
        let scene = sonar.process_frame(&samples);
        assert!(
            scene.summary.to_lowercase().contains("quiet"),
            "Expected 'quiet' in summary, got: {}",
            scene.summary
        );
    }
}
