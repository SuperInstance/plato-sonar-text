# plato-sonar-text — Acoustic Scene Analysis → Text

Translate acoustic environments into text descriptions for agents. Sound events, voice activity, ambient profiles, and alerts — all as human-readable strings.

**Part of the [Plato](https://github.com/SuperInstance/plato-shell) ecosystem.**

## What This Gives You

- **AcousticScene** — summary, sound level (dB), dominant frequency, detected events
- **SoundEvent** — label, confidence, timing, direction, frequency range
- **Voice activity** — speech detection with energy level and direction
- **Ambient profiles** — baseline noise characteristics for anomaly detection
- **AcousticAlert** — severity-ranked alerts for significant sounds

## Quick Start

```rust
use plato_sonar_text::*;

// Analyze an audio buffer
let analyzer = SonarText::new();
let scene = analyzer.analyze(&audio_samples);

println!("Scene: {}", scene.summary);
println!("Level: {:.1} dB", scene.sound_level_db);
println!("Voice: {}", scene.voice_active);

for event in &scene.events {
    println!("  {} ({:.0}% from {:.0}°)", event.label, event.confidence * 100.0, event.direction);
}
```

## How It Fits

The text interface layer on top of signal processing. [plato-correlator](https://github.com/SuperInstance/plato-correlator) fuses sonar shadows with vision shadows. [plato-shell](https://github.com/SuperInstance/plato-shell) presents acoustic scenes as room descriptions. [openconstruct-jetson](https://github.com/SuperInstance/openconstruct-jetson) provides the GPU-accelerated FFT that feeds this module.

## Installation

```toml
[dependencies]
plato-sonar-text = "0.1"
```

## License

MIT
