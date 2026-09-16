"""Synthetic 12-lead-style ECG waveform generator.

Each beat is built as a sum of Gaussian components approximating the P, Q, R, S
and T deflections -- the same construction used by the ECGSYN model of
McSharry et al. (2003). Rhythm-level behaviour (rate, RR variability, presence
of atrial activity, ectopy) is then layered on top per arrhythmia class.

Why this exists
---------------
1. It generates the sample gallery shipped with the demo, so the app has
   something to analyse out of the box with no dataset download.
2. It provides ground-truth signals for the digitizer round-trip test: render a
   known waveform to an ECG-paper image, read it back, and check the recovered
   heart rate matches.

What it is NOT
--------------
This is not a substitute for real data. Synthetic beats are drawn from the same
assumptions the rule engine tests for, so scoring the rule engine against this
generator is circular and proves nothing about clinical accuracy. Real accuracy
numbers come only from PhysioNet recordings -- see ml/prepare_data.py.
"""

from __future__ import annotations

import numpy as np

FS_DEFAULT = 360.0  # Hz, matches the MIT-BIH Arrhythmia Database


# --------------------------------------------------------------------------
# Beat morphology
# --------------------------------------------------------------------------

# (amplitude mV, centre ms relative to R peak, width ms)
NORMAL_BEAT = {
    "P": (0.15, -160.0, 26.0),
    "Q": (-0.10, -26.0, 9.0),
    "R": (1.00, 0.0, 11.0),
    "S": (-0.26, 26.0, 12.0),
    "T": (0.32, 195.0, 58.0),
}


def _gaussian(t_ms: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-((t_ms - mu) ** 2) / (2.0 * sigma**2))


def _beat_waveform(
    t_ms: np.ndarray,
    *,
    p_scale: float = 1.0,
    qrs_width: float = 1.0,
    qrs_scale: float = 1.0,
    t_scale: float = 1.0,
    t_invert: bool = False,
    st_shift: float = 0.0,
) -> np.ndarray:
    """Render one beat, sampled at offsets ``t_ms`` from the R peak."""
    out = np.zeros_like(t_ms, dtype=float)

    amp, mu, sig = NORMAL_BEAT["P"]
    if p_scale > 0:
        out += _gaussian(t_ms, amp * p_scale, mu, sig)

    for key in ("Q", "R", "S"):
        amp, mu, sig = NORMAL_BEAT[key]
        out += _gaussian(t_ms, amp * qrs_scale, mu * qrs_width, sig * qrs_width)

    amp, mu, sig = NORMAL_BEAT["T"]
    t_amp = amp * t_scale * (-1.0 if t_invert else 1.0)
    out += _gaussian(t_ms, t_amp, mu * max(qrs_width, 1.0), sig)

    if st_shift:
        # Broad plateau between S and T, i.e. the ST segment.
        out += _gaussian(t_ms, st_shift, 90.0, 45.0)

    return out


# --------------------------------------------------------------------------
# Rhythm construction
# --------------------------------------------------------------------------


def _rr_series(rng, duration, mean_rr, sd_rr, *, irregular=False, floor=0.22):
    """Beat-to-beat RR intervals in seconds, summing to ~duration."""
    rr, total = [], 0.0
    while total < duration + mean_rr:
        if irregular:
            # Irregularly irregular: heavy-tailed, no underlying periodicity.
            interval = mean_rr * rng.lognormal(mean=0.0, sigma=sd_rr)
        else:
            interval = rng.normal(mean_rr, sd_rr)
        interval = float(np.clip(interval, floor, 3.0))
        rr.append(interval)
        total += interval
    return np.array(rr)


def _baseline_artifacts(rng, n, fs, *, wander=0.06, mains=0.004, emg=0.008, mains_hz=60.0):
    t = np.arange(n) / fs
    out = np.zeros(n)
    # Respiratory baseline wander, ~0.2-0.4 Hz.
    for f, a in ((rng.uniform(0.15, 0.35), wander), (rng.uniform(0.05, 0.12), wander * 0.6)):
        out += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    # Mains interference.
    out += mains * np.sin(2 * np.pi * mains_hz * t + rng.uniform(0, 2 * np.pi))
    # Broadband muscle noise.
    out += rng.normal(0.0, emg, n)
    return out


def _place_beats(signal, fs, beat_times, morph_fn, window_ms=520.0):
    """Additively stamp beats into ``signal`` at the given times."""
    half = int(window_ms / 1000.0 * fs)
    offsets = (np.arange(-half, half + 1) / fs) * 1000.0
    n = len(signal)
    for i, bt in enumerate(beat_times):
        centre = int(round(bt * fs))
        lo, hi = centre - half, centre + half + 1
        wave = morph_fn(offsets, i)
        s_lo, s_hi = max(lo, 0), min(hi, n)
        if s_lo >= s_hi:
            continue
        signal[s_lo:s_hi] += wave[s_lo - lo : s_hi - lo]
    return signal


# --------------------------------------------------------------------------
# Per-class generators
# --------------------------------------------------------------------------


def _sinus(rng, duration, fs, bpm_lo, bpm_hi, sd, *, p_scale=1.0):
    bpm = rng.uniform(bpm_lo, bpm_hi)
    mean_rr = 60.0 / bpm
    rr = _rr_series(rng, duration, mean_rr, mean_rr * sd)
    beats = np.cumsum(rr) - rr[0] + rng.uniform(0.10, 0.45)
    n = int(duration * fs)
    sig = np.zeros(n)

    def morph(off, _i):
        return _beat_waveform(off, p_scale=p_scale, qrs_scale=rng.uniform(0.92, 1.08))

    _place_beats(sig, fs, beats[beats < duration], morph)
    return sig, {"target_bpm": bpm}


def _gen_normal(rng, duration, fs):
    return _sinus(rng, duration, fs, 62, 96, 0.035)


def _gen_bradycardia(rng, duration, fs):
    return _sinus(rng, duration, fs, 38, 56, 0.035)


def _gen_tachycardia(rng, duration, fs):
    return _sinus(rng, duration, fs, 108, 155, 0.028)


def _gen_svt(rng, duration, fs):
    """Fast, regular, narrow-complex. P waves buried; T wave distorted."""
    bpm = rng.uniform(165, 235)
    mean_rr = 60.0 / bpm
    rr = _rr_series(rng, duration, mean_rr, mean_rr * 0.015, floor=0.20)
    beats = np.cumsum(rr) - rr[0] + rng.uniform(0.05, 0.25)
    n = int(duration * fs)
    sig = np.zeros(n)

    def morph(off, _i):
        return _beat_waveform(
            off,
            p_scale=0.0,          # no discrete P
            qrs_width=0.82,       # short/narrow QRS
            qrs_scale=rng.uniform(0.88, 1.02),
            t_scale=0.55,         # distorted, flattened T
            st_shift=-0.05,
        )

    _place_beats(sig, fs, beats[beats < duration], morph, window_ms=300.0)
    return sig, {"target_bpm": bpm}


def _gen_afib(rng, duration, fs):
    """No P waves, fibrillatory baseline, irregularly irregular ventricular response."""
    bpm = rng.uniform(95, 155)
    mean_rr = 60.0 / bpm
    rr = _rr_series(rng, duration, mean_rr, 0.22, irregular=True)
    beats = np.cumsum(rr) - rr[0] + rng.uniform(0.05, 0.35)
    n = int(duration * fs)
    sig = np.zeros(n)

    def morph(off, _i):
        return _beat_waveform(off, p_scale=0.0, qrs_scale=rng.uniform(0.85, 1.12))

    _place_beats(sig, fs, beats[beats < duration], morph)

    # Fibrillatory waves: disorganised atrial activity, 350-600/min.
    t = np.arange(n) / fs
    fwave = np.zeros(n)
    for _ in range(6):
        f = rng.uniform(5.8, 9.5)
        fwave += rng.uniform(0.012, 0.030) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    return sig + fwave, {"target_bpm": bpm}


def _gen_atrial_flutter(rng, duration, fs):
    """Sawtooth flutter waves at ~250-350/min with fixed-ratio conduction."""
    flutter_bpm = rng.uniform(250, 340)
    ratio = int(rng.choice([2, 3, 4]))
    vent_bpm = flutter_bpm / ratio
    mean_rr = 60.0 / vent_bpm
    rr = _rr_series(rng, duration, mean_rr, mean_rr * 0.02)
    beats = np.cumsum(rr) - rr[0] + rng.uniform(0.05, 0.30)
    n = int(duration * fs)
    sig = np.zeros(n)

    def morph(off, _i):
        return _beat_waveform(off, p_scale=0.0, qrs_scale=rng.uniform(0.90, 1.08), t_scale=0.5)

    _place_beats(sig, fs, beats[beats < duration], morph)

    # Sawtooth atrial activity across the whole strip.
    t = np.arange(n) / fs
    f = flutter_bpm / 60.0
    phase = (t * f) % 1.0
    sawtooth = (2.0 * phase - 1.0) * 0.16
    return sig + sawtooth, {"target_bpm": vent_bpm, "flutter_bpm": flutter_bpm, "ratio": ratio}


def _gen_vpb(rng, duration, fs):
    """Sinus rhythm interrupted by wide premature ectopics with compensatory pauses."""
    bpm = rng.uniform(66, 92)
    base_rr = 60.0 / bpm
    times, ectopic_idx, t_now, i = [], set(), rng.uniform(0.15, 0.45), 0
    while t_now < duration:
        times.append(t_now)
        # Roughly every 3rd-5th beat is ectopic.
        if i > 0 and rng.random() < 0.26:
            ectopic_idx.add(len(times) - 1)
            t_now += base_rr * rng.uniform(0.52, 0.68)   # premature
            times.append(t_now)
            ectopic_idx.discard(len(times) - 1)
            ectopic_idx.add(len(times) - 1)
            t_now += base_rr * rng.uniform(1.32, 1.52)   # compensatory pause
        else:
            t_now += base_rr * rng.normal(1.0, 0.03)
        i += 1

    beats = np.array([t for t in times if t < duration])
    n = int(duration * fs)
    sig = np.zeros(n)

    def morph(off, idx):
        if idx in ectopic_idx:
            return _beat_waveform(
                off,
                p_scale=0.0,        # no preceding P
                qrs_width=2.6,      # wide, bizarre
                qrs_scale=rng.uniform(1.25, 1.70),
                t_scale=1.5,
                t_invert=True,      # discordant T
            )
        return _beat_waveform(off, qrs_scale=rng.uniform(0.93, 1.07))

    _place_beats(sig, fs, beats, morph, window_ms=620.0)
    return sig, {"target_bpm": bpm, "ectopic_count": len(ectopic_idx)}


def _gen_vfib(rng, duration, fs):
    """Chaotic ventricular activity: no organised QRS, no measurable rate."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    sig = np.zeros(n)
    # A handful of drifting oscillators with wandering frequency and amplitude.
    for _ in range(5):
        f0 = rng.uniform(3.5, 7.5)
        drift = np.cumsum(rng.normal(0, 0.9, n)) / fs
        amp = rng.uniform(0.25, 0.65) * (
            1.0 + 0.45 * np.sin(2 * np.pi * rng.uniform(0.2, 0.7) * t + rng.uniform(0, 6.28))
        )
        sig += amp * np.sin(2 * np.pi * f0 * t + drift + rng.uniform(0, 6.28))
    sig *= 0.55
    sig += rng.normal(0, 0.05, n)
    return sig, {"target_bpm": None}


_GENERATORS = {
    "normal": _gen_normal,
    "atrial_flutter": _gen_atrial_flutter,
    "tachycardia": _gen_tachycardia,
    "bradycardia": _gen_bradycardia,
    "svt": _gen_svt,
    "afib": _gen_afib,
    "vfib": _gen_vfib,
    "vpb": _gen_vpb,
}


def generate(
    label: str,
    *,
    duration: float = 10.0,
    fs: float = FS_DEFAULT,
    seed: int | None = None,
    clean: bool = False,
) -> tuple[np.ndarray, dict]:
    """Generate one synthetic ECG strip.

    Returns ``(signal_mV, metadata)``. ``clean=True`` omits noise and baseline
    wander, which is what the digitizer round-trip test uses.
    """
    if label not in _GENERATORS:
        raise KeyError(f"unknown class {label!r}; expected one of {sorted(_GENERATORS)}")
    rng = np.random.default_rng(seed)
    sig, meta = _GENERATORS[label](rng, duration, fs)

    if not clean:
        sig = sig + _baseline_artifacts(rng, len(sig), fs)

    meta.update({"label": label, "fs": fs, "duration": duration, "seed": seed})
    return sig.astype(np.float32), meta


def generate_batch(label: str, count: int, *, seed: int = 0, **kw):
    """Yield ``count`` independent strips of one class with reproducible seeds."""
    for i in range(count):
        yield generate(label, seed=seed * 100_003 + i, **kw)
