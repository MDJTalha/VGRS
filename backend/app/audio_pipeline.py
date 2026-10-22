from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import subprocess
from math import gcd
from typing import Any
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16000
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a", ".webm"}

class AudioValidationError(ValueError):
    """Uploaded recording cannot safely be analyzed."""

@dataclass
class AudioAnalysis:
    duration_seconds: float
    source_sample_rate: int
    sample_rate: int
    channels: int
    features: dict[str, Any]
    quality: dict[str, Any]
    waveform: np.ndarray

def _ffmpeg_executable() -> str:
    """Resolve the bundled FFmpeg binary, with a system PATH fallback."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        # Keeps the error path useful when dependencies were not installed fully.
        return "ffmpeg"

def _decode(payload: bytes, filename: str, *, max_duration: float = 120.0) -> tuple[np.ndarray, int, int]:
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise AudioValidationError("Unsupported audio extension.")
    try:
        source = BytesIO(payload)
        info = sf.info(source)
        if info.samplerate < 8000 or info.samplerate > 96000:
            raise AudioValidationError("Sample rate must be between 8 kHz and 96 kHz.")
        if info.channels < 1 or info.channels > 8:
            raise AudioValidationError("Audio must contain between 1 and 8 channels.")
        if info.duration > max_duration:
            raise AudioValidationError(f"Recording exceeds the {max_duration:.0f}-second duration limit.")
        source.seek(0)
        samples, sr = sf.read(source, dtype="float32", always_2d=True)
        return samples, int(sr), int(samples.shape[1])
    except AudioValidationError:
        raise
    except Exception as decode_error:
        # FFmpeg fallback streams bytes through stdin/stdout; original audio is not saved.
        try:
            proc = subprocess.run(
                [_ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                 "-t", str(max_duration + 0.25), "-threads", "1", "-f", "wav", "-acodec", "pcm_f32le", "pipe:1"],
                input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25, check=True,
            )
            samples, sr = sf.read(BytesIO(proc.stdout), dtype="float32", always_2d=True)
            return samples, int(sr), int(samples.shape[1])
        except FileNotFoundError as exc:
            raise AudioValidationError(
                "FFmpeg is unavailable. Reinstall backend requirements (which bundle FFmpeg on supported platforms), "
                "or install FFmpeg and add it to PATH for MP3/M4A/WebM support."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioValidationError("Audio decoding timed out; try a shorter recording.") from exc
        except Exception as exc:
            raise AudioValidationError("The uploaded file is not valid or decodable audio.") from exc

def _pitch_estimates(mono: np.ndarray, sr: int) -> list[float]:
    window, hop = int(0.040 * sr), int(0.020 * sr)
    if len(mono) < window:
        return []
    pitches: list[float] = []
    for start in range(0, len(mono) - window + 1, hop):
        frame = mono[start:start + window].astype(np.float64, copy=True)
        frame -= frame.mean()
        frame *= np.hanning(len(frame))
        if np.sqrt(np.mean(frame * frame) + 1e-12) < 0.008:
            continue
        fft = np.fft.rfft(frame, n=2048)
        corr = np.fft.irfft(fft * np.conjugate(fft), n=2048)[:window]
        lo, hi = max(1, int(sr / 400)), min(len(corr) - 1, int(sr / 65))
        if hi <= lo:
            continue
        lag = lo + int(np.argmax(corr[lo:hi + 1]))
        if corr[lag] / (corr[0] + 1e-12) >= 0.30:
            value = sr / lag
            if 65 <= value <= 400:
                pitches.append(float(value))
    return pitches

def analyze_audio(payload: bytes, filename: str, *, max_duration: float = 120.0,
                  min_duration: float = 2.0) -> AudioAnalysis:
    if not payload:
        raise AudioValidationError("The uploaded file is empty.")
    samples, source_sr, channels = _decode(payload, filename, max_duration=max_duration)
    if source_sr < 8000 or source_sr > 96000:
        raise AudioValidationError("Sample rate must be between 8 kHz and 96 kHz.")
    if channels < 1 or channels > 8:
        raise AudioValidationError("Audio must contain between 1 and 8 channels.")
    if samples.size == 0 or not np.isfinite(samples).all():
        raise AudioValidationError("Audio contains no usable samples or invalid values.")
    duration = len(samples) / source_sr
    if duration < min_duration:
        raise AudioValidationError(f"Recording is too short ({duration:.2f}s); minimum is {min_duration:.1f}s.")
    if duration > max_duration:
        raise AudioValidationError(f"Recording exceeds the {max_duration:.0f}-second duration limit.")

    peak = float(np.max(np.abs(samples)))
    mono = samples.mean(axis=1, dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
    if rms < 0.0015:
        raise AudioValidationError("Recording is silent or too quiet. Record speech closer to the microphone.")
    original_rms = rms
    if source_sr != TARGET_SAMPLE_RATE:
        common = gcd(source_sr, TARGET_SAMPLE_RATE)
        mono = resample_poly(mono, TARGET_SAMPLE_RATE // common, source_sr // common).astype(np.float32)
    sr = TARGET_SAMPLE_RATE
    duration = len(mono) / sr

    frame_len, hop = int(.025 * sr), int(.010 * sr)
    frame_rms = []
    for start in range(0, max(1, len(mono) - frame_len + 1), hop):
        frame = mono[start:start + frame_len]
        if len(frame) == frame_len:
            frame_rms.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12)))
    frame_values = np.asarray(frame_rms or [0.0])
    threshold = max(.003, float(np.percentile(frame_values, 90)) * .10)
    active_fraction = float(np.mean(frame_values > threshold))
    clipping_fraction = float(np.mean(np.abs(samples) >= .999))
    centered = mono.astype(np.float64) - float(np.mean(mono))
    chunk = centered[:min(len(centered), sr * 10)]
    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), d=1 / sr)
    total = float(spectrum.sum())
    centroid = float(np.sum(freqs * spectrum) / total) if total > 1e-12 else None
    if total > 1e-12:
        cumulative = np.cumsum(spectrum)
        rolloff = float(freqs[min(int(np.searchsorted(cumulative, .85 * cumulative[-1])), len(freqs) - 1)])
    else:
        rolloff = None
    zcr = float(np.mean(np.abs(np.diff(np.signbit(mono.astype(np.float64)))))) if len(mono) > 1 else None
    f0 = _pitch_estimates(mono, sr)

    warnings: list[str] = []
    if clipping_fraction > .005:
        warnings.append("Clipping detected; reduce input gain if possible.")
    if active_fraction < .25:
        warnings.append("Low active-audio fraction; clip may contain long silences.")
    if active_fraction > 0 and active_fraction < .45:
        warnings.append("Only a limited proportion of the recording contains active audio.")
    if peak < .03:
        warnings.append("Low recording level; measurements may be affected by background noise.")
    if not f0:
        warnings.append("Reliable pitch tracking was unavailable for this recording.")

    features: dict[str, Any] = {
        "duration_seconds": round(duration, 3), "sample_rate_hz": sr,
        "source_sample_rate_hz": source_sr, "channel_count": channels,
        "rms_energy": round(original_rms, 6), "peak_amplitude": round(peak, 6),
        "active_audio_fraction": round(active_fraction, 4),
        "clipping_fraction": round(clipping_fraction, 6),
        "spectral_centroid_hz": round(centroid, 2) if centroid is not None else None,
        "spectral_rolloff_85_hz": round(rolloff, 2) if rolloff is not None else None,
        "zero_crossing_rate": round(zcr, 5) if zcr is not None else None,
        "f0_median_hz": round(float(np.median(f0)), 2) if f0 else None,
        "f0_p10_hz": round(float(np.percentile(f0, 10)), 2) if f0 else None,
        "f0_p90_hz": round(float(np.percentile(f0, 90)), 2) if f0 else None,
        "f0_std_hz": round(float(np.std(f0)), 2) if f0 else None,
        "f0_voiced_frames": len(f0), "snr_db": None, "jitter": None, "shimmer": None,
        "speaking_rate_words_per_minute": None,
    }
    return AudioAnalysis(duration, source_sr, sr, channels, features,
                         {"status": "acceptable" if not warnings else "check_recording", "warnings": warnings},
                         mono.astype(np.float32, copy=False))
