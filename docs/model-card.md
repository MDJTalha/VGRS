# Model card: VGRS acoustic baseline and optional classifier

## Status

The default version is a signal-measurement pipeline, not a demographic classifier. Application version: 0.2.0. The optional public audEERING adapter is non-commercial research only; it remains experimental and is blocked in production mode. Production classifier output is accepted only through a custom adapter with a matching validation report and explicit license approval.

## Intended use

Development, quality inspection and research into speech-signal characteristics. It is not an identity-verification system, medical device, age-verification system or decision-making classifier. Do not use voice estimates for hiring, access control, eligibility or another high-impact decision.

## Inputs and preprocessing

Audio inputs are limited by default to 25 MB per file, 2–120 seconds and sample rates from 8 to 96 kHz. The application caps total batch body size at 100 MB and batch size at 20. SoundFile metadata is checked before reading supported formats; FFmpeg is the fallback for other supported codecs. Audio is converted to mono and resampled to 16 kHz. The conversion averages channels; it is not a speech-enhancement process.

## Acoustic outputs

Duration, source/output sample rate, channel count, RMS energy, peak amplitude, active-audio fraction, clipping fraction, spectral centroid, 85% spectral rolloff, zero-crossing rate and a basic autocorrelation F0 summary (median, 10th percentile, 90th percentile, standard deviation and voiced-frame count). SNR, jitter, shimmer and speaking rate are explicitly unavailable in this baseline. Fundamental-frequency tracking can fail on creaky, breathy, unvoiced, noisy or polyphonic audio.

## Optional research classifier

The opt-in `audeering-research` adapter expects locally installed `audonnx` and an extracted public six-layer ONNX model directory. Its public weights are licensed CC BY-NC-SA 4.0 for non-commercial use; the publisher describes the open model as English-focused. It outputs an approximate age scalar and three logits (female, male, child). VGRS maps the scalar to project-defined perceived age bands, maps only the female/male classes to feminine-coded/masculine-coded vocal-presentation labels, and abstains when the child class wins. It emits no confidence scores because confidence calibration for VGRS has not been established. The research adapter is prohibited in production mode and does not claim language/subgroup accuracy.

## Production classifier contract

A separately licensed model may be connected via the custom classifier plug-in contract. Production startup requires a matching validation report, at least 500 held-out distinct speakers, per-class and subgroup metrics, calibration/abstention measures, dataset-rights review, explicit label definitions and reviewer approval. These are process gates rather than a substitute for qualified independent review.

## Limitations and risks

Acoustic properties vary with speaker, language, accent, emotion, vocal effort, health, voice training, microphone, room and noise. Perceived vocal presentation is not gender identity. A voice-based estimate does not verify chronological age. Model results can have subgroup error disparities and may be out of domain. The model must abstain when quality or domain checks fail.

## Evaluation state

Automated tests cover API behavior, generated tone features, duration bounds, FFmpeg fallback, abstention contract and production configuration gates. No independently labeled, representative VGRS speech dataset has been evaluated, and no VGRS production classification macro-F1, balanced accuracy, per-class precision/recall, calibration or fairness claim is available. A synthetic tone is a code-path fixture, not evidence of natural-speech demographic accuracy.
