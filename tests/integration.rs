//! Integration tests for plato-sonar-text

use plato_sonar_text::*;

fn make_sine(freq: f64, duration_secs: f64, sample_rate: f64) -> Vec<f64> {
    let n = (duration_secs * sample_rate) as usize;
    let mut samples = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / sample_rate;
        let val = (2.0 * std::f64::consts::PI * freq * t).sin() * 0.5;
        samples.push(val * 0.8);
        samples.push(val * 0.4);
    }
    samples
}

fn make_noise(duration_secs: f64, sample_rate: f64) -> Vec<f64> {
    let n = (duration_secs * sample_rate) as usize;
    let mut samples = Vec::with_capacity(n * 2);
    let mut state: u64 = 42;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as f64 / 2147483648.0 - 1.0) * 0.5;
        samples.push(val);
        samples.push(val * 0.9);
    }
    samples
}

#[test]
fn test_sonar_text_creation() {
    let sonar = SonarText::new();
    let scene = sonar.detect_events(&make_silence(0.01, 44100.0));
    assert!(scene.is_empty() || scene[0].label == "silence");
}

#[test]
fn test_process_frame_returns_acoustic_scene() {
    let mut sonar = SonarText::new();
    let samples = make_sine(440.0, 0.05, 44100.0);
    let scene = sonar.process_frame(&samples);
    assert!(!scene.summary.is_empty());
    assert!(scene.sound_level_db > -100.0);
}

#[test]
fn test_direction_estimator_symmetry() {
    let left = vec![0.8f64; 100];
    let right = vec![0.2f64; 100];
    let angle = DirectionEstimator::estimate(&left, &right);
    assert!(angle < 0.0, "Left-dominant should be negative, got {}", angle);

    let angle = DirectionEstimator::estimate(&right, &left);
    assert!(angle > 0.0, "Right-dominant should be positive, got {}", angle);
}

#[test]
fn test_baseline_profile_tracks_ambient() {
    let mut sonar = SonarText::new();
    let noise = make_noise(0.05, 44100.0);
    let profile = sonar.update_baseline(&noise);
    assert!(profile.avg_db > -100.0);
    assert!(profile.baseline_updated > 0);
    assert!(!profile.frequency_spectrum_summary.is_empty());
}

#[test]
fn test_alert_generation_for_loud_noise() {
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
    assert!(alerts[0].severity >= 1);
}

#[test]
fn test_voice_activity_detection() {
    let sonar = SonarText::new();
    // 150 Hz sine should be detected as voice-like
    let samples = make_sine(150.0, 0.05, 44100.0);
    let voice = sonar.detect_voice(&samples);
    assert!(voice.is_speech);

    // Silence should not be detected as voice
    let silence = vec![0.0f64; 1000];
    let voice = sonar.detect_voice(&silence);
    assert!(!voice.is_speech);
}

fn make_silence(duration_secs: f64, sample_rate: f64) -> Vec<f64> {
    let n = (duration_secs * sample_rate) as usize;
    vec![0.0; n * 2]
}
