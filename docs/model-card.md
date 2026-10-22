# Model card: VGRS acoustic feature baseline

## Status

This is a signal-measurement pipeline, not a demographic classifier. Version: 0.1.0. Implementation: `backend/app/audio_pipeline.py`. It runs locally on Python with NumPy, SciPy and SoundFile; FFmpeg is an optional local decoder fallback.

## Intended use

Development, quality inspection and research into speech-signal characteristics. It is not an identity-verification system, medical device, age-verification system or decision-making classifier. Do not use for hiring, access control, eligibility or another high-impact decision.

## Inputs and preprocessing

Audio supplied as WAV/FLAC/OGG/AIFF or selected common compressed formats, subject to installed decoder support. Current defaults allow 2–120 seconds, sample rates from 8 to 96 kHz and 25 MB per file. Inputs are converted to mono and resampled to 16 kHz. The conversion averages channels; it is not a speech-enhancement process.

## Outputs

Duration, original/output sample rate, channel count, RMS energy, peak amplitude, active-audio fraction, clipping fraction, spectral centroid, 85% spectral rolloff, zero-crossing rate and a basic autocorrelation F0 summary (median, 10th percentile, 90th percentile, standard deviation and voiced frame count). SNR, jitter, shimmer and speaking rate are currently explicitly unavailable. Fundamental-frequency tracking is simple and can fail on creaky, breathy, unvoiced, noisy or polyphonic audio.

## Classification status

Age-group and perceived vocal-presentation classification are not configured. The API returns `model_unavailable`, with null labels and scores. There is no calibrated probability, model accuracy, confusion matrix or generalization claim for these attributes.

## Limitations and risks

Acoustic properties vary with speaker, language, accent, emotion, vocal effort, illness, voice training, microphone, room and noise. These features do not identify gender identity or establish chronological age. A synthetic tone is a pipeline fixture, not evidence of performance on human voice recordings. Fairness and generalization to specific populations have not been evaluated.

## Evaluation state

Automated baseline test suite exercises generated tones, audio validation and API contracts. There is no independently labeled real-speech dataset and no classifier evaluation report with accuracy values. See `evaluation-report.md` for what is and is not verified.
