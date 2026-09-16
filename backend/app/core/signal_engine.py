"""Signal-domain ECG analysis.

Everything in this module is deterministic DSP -- no learned parameters, no
training required. It runs the Pan-Tompkins QRS detector (Pan & Tompkins, IEEE
TBME 1985) and derives the measurements a clinician actually reads off a strip:
rate, RR statistics, QRS duration, atrial activity and rhythm organisation.

These measurements serve two purposes:

1. They are reported to the user alongside the model prediction, so the output
   is interpretable rather than an opaque label.
2. They feed the rule engine (``rules.py``), which acts as an independent
   second opinion that can corroborate or contradict the neural network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import signal as sps


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


def bandpass(x: np.ndarray, fs: float, lo: float = 5.0, hi: float = 15.0) -> np.ndarray:
    """Pan-Tompkins passband: isolates QRS energy, rejects P/T and drift."""
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    if lo >= hi:
        return x.astype(float)
    b, a = sps.butter(2, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x)


def remove_baseline(x: np.ndarray, fs: float, cutoff: float = 0.5) -> np.ndarray:
    """High-pass to remove respiratory baseline wander."""
    nyq = fs / 2.0
    b, a = sps.butter(2, cutoff / nyq, btype="high")
    return sps.filtfilt(b, a, x)


def notch(x: np.ndarray, fs: float, freq: float = 60.0, q: float = 30.0) -> np.ndarray:
    """Remove mains interference. Harmless if the frequency is not present."""
    if freq >= fs / 2.0:
        return x
    b, a = sps.iirnotch(freq / (fs / 2.0), q)
    return sps.filtfilt(b, a, x)


def preprocess(x: np.ndarray, fs: float, mains_hz: float = 60.0) -> np.ndarray:
    """Standard cleanup applied before any measurement is taken."""
    x = np.asarray(x, dtype=float)
    x = x - np.median(x)
    x = remove_baseline(x, fs)
    x = notch(x, fs, mains_hz)
    return x


# --------------------------------------------------------------------------
# QRS detection
# --------------------------------------------------------------------------


def detect_qrs(x: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins QRS detection. Returns R-peak sample indices.

    Pipeline: bandpass -> differentiate -> square -> moving-window integrate ->
    adaptive dual-threshold peak selection with RR-based search-back.
    """
    if len(x) < int(fs):
        return np.array([], dtype=int)

    filtered = bandpass(x, fs)
    diff = np.ediff1d(filtered, to_begin=0.0)
    squared = diff**2

    win = max(1, int(round(0.150 * fs)))  # 150 ms integration window
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    refractory = int(round(0.200 * fs))  # 200 ms physiological refractory period
    if integrated.max() <= 0:
        return np.array([], dtype=int)

    peaks, _ = sps.find_peaks(integrated, distance=refractory)
    if peaks.size == 0:
        return np.array([], dtype=int)

    # Adaptive dual threshold, as in the original paper.
    spki = float(np.percentile(integrated[peaks], 75))
    npki = float(np.percentile(integrated, 40))
    threshold = npki + 0.25 * (spki - npki)

    accepted: list[int] = []
    for p in peaks:
        val = integrated[p]
        if val >= threshold:
            accepted.append(int(p))
            spki = 0.125 * val + 0.875 * spki
        else:
            npki = 0.125 * val + 0.875 * npki
        threshold = npki + 0.25 * (spki - npki)

    if len(accepted) < 2:
        return np.array([], dtype=int)

    # Search-back: an RR gap longer than 1.66x the running median usually means
    # a beat was missed below threshold.
    accepted = _search_back(np.array(accepted), integrated, threshold, fs)

    # Snap each integrator peak onto the true R deflection in the raw signal.
    return _localise_r_peaks(x, accepted, fs)


def _search_back(peaks: np.ndarray, integrated: np.ndarray, threshold: float, fs: float):
    if peaks.size < 3:
        return peaks
    out = [int(peaks[0])]
    rr = np.diff(peaks)
    med = float(np.median(rr))
    for i in range(1, len(peaks)):
        gap = peaks[i] - out[-1]
        if gap > 1.66 * med and gap > int(0.5 * fs):
            lo, hi = out[-1] + int(0.20 * fs), peaks[i] - int(0.20 * fs)
            if hi > lo:
                seg = integrated[lo:hi]
                cand = int(np.argmax(seg)) + lo
                if integrated[cand] >= 0.35 * threshold:
                    out.append(cand)
        out.append(int(peaks[i]))
    return np.array(sorted(set(out)))


def _localise_r_peaks(x: np.ndarray, peaks, fs: float) -> np.ndarray:
    """Refine detection indices to the local extremum of the raw signal."""
    half = int(round(0.060 * fs))
    n = len(x)
    centred = x - np.median(x)
    out = []
    for p in peaks:
        lo, hi = max(0, int(p) - half), min(n, int(p) + half + 1)
        if hi <= lo:
            continue
        seg = centred[lo:hi]
        # Take whichever deflection is larger; some leads are R-negative.
        idx = int(np.argmax(np.abs(seg))) + lo
        out.append(idx)
    if not out:
        return np.array([], dtype=int)
    out = np.array(sorted(set(out)))
    # Drop anything closer than the refractory period after refinement.
    keep = [out[0]]
    for p in out[1:]:
        if p - keep[-1] >= int(0.20 * fs):
            keep.append(p)
    return np.array(keep, dtype=int)


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------


@dataclass
class Measurements:
    """Clinical measurements extracted from one strip."""

    duration_s: float = 0.0
    beat_count: int = 0

    heart_rate_bpm: float | None = None
    rr_mean_ms: float | None = None
    rr_sd_ms: float | None = None
    rr_min_ms: float | None = None
    rr_max_ms: float | None = None
    rmssd_ms: float | None = None
    pnn50_pct: float | None = None
    # Coefficient of variation of RR. The single best AFib discriminator.
    rr_cv: float | None = None
    # Fraction of consecutive RR pairs differing by more than 12.5%.
    irregularity_index: float | None = None

    qrs_duration_ms: float | None = None
    qrs_p90_ms: float | None = None
    qrs_wide_fraction: float | None = None

    p_wave_present: bool = False
    p_wave_ratio: float | None = None      # P amplitude / R amplitude
    p_wave_consistency: float | None = None  # 0-1, beat-to-beat correlation

    atrial_rate_bpm: float | None = None   # dominant atrial frequency, if organised
    atrial_organisation: float | None = None  # autocorrelation strength at that rate
    # True only when the atrial activity is organised enough for the rate above
    # to mean anything. Without this gate the detector always returns *some*
    # dominant frequency, and a meaningless number reported as an "atrial rate"
    # is worse than reporting nothing.
    atrial_activity_organised: bool = False
    flutter_ratio: float | None = None     # atrial rate / ventricular rate

    # Fraction of the strip where |signal| exceeds 20% of peak. Low in organised
    # rhythms (mostly isoelectric baseline), high in fibrillation.
    baseline_occupancy: float | None = None
    # Mean correlation of each beat against the median beat template.
    qrs_template_match: float | None = None

    r_peaks: list = field(default_factory=list)
    signal_quality: float = 1.0
    quality_notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _rr_stats(m: Measurements, rr_ms: np.ndarray) -> None:
    m.rr_mean_ms = float(np.mean(rr_ms))
    m.rr_sd_ms = float(np.std(rr_ms))
    m.rr_min_ms = float(np.min(rr_ms))
    m.rr_max_ms = float(np.max(rr_ms))
    m.heart_rate_bpm = float(60_000.0 / np.median(rr_ms))
    m.rr_cv = float(m.rr_sd_ms / m.rr_mean_ms) if m.rr_mean_ms else None

    if len(rr_ms) > 1:
        d = np.diff(rr_ms)
        m.rmssd_ms = float(np.sqrt(np.mean(d**2)))
        m.pnn50_pct = float(np.mean(np.abs(d) > 50.0) * 100.0)
        rel = np.abs(d) / rr_ms[:-1]
        m.irregularity_index = float(np.mean(rel > 0.125))
    else:
        m.rmssd_ms = 0.0
        m.pnn50_pct = 0.0
        m.irregularity_index = 0.0


def _beat_matrix(x: np.ndarray, peaks: np.ndarray, fs: float, pre_ms=250.0, post_ms=400.0):
    """Stack beats aligned on their R peaks. Returns (matrix, time_axis_ms)."""
    pre, post = int(pre_ms / 1000 * fs), int(post_ms / 1000 * fs)
    rows = []
    for p in peaks:
        lo, hi = p - pre, p + post
        if lo < 0 or hi > len(x):
            continue
        rows.append(x[lo:hi])
    if not rows:
        return np.empty((0, pre + post)), np.linspace(-pre_ms, post_ms, pre + post)
    return np.vstack(rows), np.linspace(-pre_ms, post_ms, pre + post)


def _qrs_duration(x: np.ndarray, peaks: np.ndarray, fs: float):
    """QRS width per beat, and the fraction of beats that are abnormally wide.

    Measured on the 8-25 Hz band, which carries depolarisation energy but not
    the slower P and T repolarisation waves. The boundary is found by growing a
    *contiguous* region outward from the R peak until the envelope falls below
    25% of its peak -- taking every sample above threshold within a window
    would also capture neighbouring P/T energy and overestimate the width.
    """
    # The analytic (Hilbert) envelope has no zero crossings, so the region does
    # not terminate at the R-to-S transition the way |raw signal| would.
    env = np.abs(sps.hilbert(bandpass(x, fs, 8.0, min(25.0, fs / 2 * 0.9))))
    k = max(1, int(0.008 * fs))
    env = np.convolve(env, np.ones(k) / k, mode="same")

    span = int(0.110 * fs)  # QRS cannot plausibly exceed ~220 ms
    n = len(env)
    widths = []
    for p in peaks:
        lo, hi = max(0, int(p) - span), min(n, int(p) + span + 1)
        seg = env[lo:hi]
        if seg.size < 5:
            continue
        centre = int(np.clip(int(p) - lo, 0, seg.size - 1))
        # 0.25 of the envelope peak. Calibrated so a normal complex measures
        # ~100 ms, matching the accepted physiological range of 80-100 ms.
        thr = 0.25 * (seg[centre] if seg[centre] > 0 else seg.max())
        if thr <= 0:
            continue
        left = centre
        while left > 0 and seg[left - 1] >= thr:
            left -= 1
        right = centre
        while right < seg.size - 1 and seg[right + 1] >= thr:
            right += 1
        widths.append((right - left) / fs * 1000.0)

    if not widths:
        return None, None, None
    widths = np.array(widths)
    # 120 ms is the conventional threshold for a "wide" complex.
    return (
        float(np.median(widths)),
        float(np.percentile(widths, 90)),
        float(np.mean(widths > 120.0)),
    )


def _p_wave(beats: np.ndarray, taxis: np.ndarray, rr_median_ms: float | None = None):
    """Look for consistent atrial activity in the PR window before each QRS.

    The search window is scaled to the RR interval. At very fast rates the PR
    window closes entirely -- the preceding T wave runs into the next QRS -- and
    atrial activity genuinely cannot be assessed from a surface trace. In that
    case we report P waves as not present rather than mistaking a T wave for a
    P wave, which is exactly how a clinician reads a narrow-complex tachycardia.
    """
    if beats.shape[0] < 3:
        return False, None, None

    start = -240.0
    if rr_median_ms:
        start = -min(240.0, 0.42 * rr_median_ms)
    end = -70.0
    # A P wave plus a PR segment needs roughly 90 ms of room. Below that the
    # preceding T wave abuts the QRS and anything found there is a T wave.
    if end - start < 90.0:
        # PR window too short to hold a resolvable P wave.
        return False, None, None

    window = (taxis >= start) & (taxis <= end)
    if window.sum() < 5:
        return False, None, None

    # Baseline-correct each beat on the TP segment just before the P window.
    base_win = (taxis >= start - 30.0) & (taxis <= start)
    if base_win.sum() > 0:
        beats = beats - beats[:, base_win].mean(axis=1, keepdims=True)

    seg = beats[:, window]
    template = seg.mean(axis=0)
    p_amp = float(np.max(np.abs(template)))

    r_amp = float(np.median(np.max(np.abs(beats), axis=1))) or 1.0
    ratio = p_amp / r_amp

    # Consistency: how well individual beats match the averaged P template.
    # Genuine P waves are phase-locked to the QRS; noise and f-waves are not.
    corrs = []
    tz = template - template.mean()
    denom_t = np.linalg.norm(tz)
    if denom_t > 1e-9:
        for row in seg:
            rz = row - row.mean()
            d = np.linalg.norm(rz) * denom_t
            if d > 1e-9:
                corrs.append(float(np.dot(rz, tz) / d))
    consistency = float(np.mean(corrs)) if corrs else 0.0

    # Consistency matters more than amplitude: fibrillatory waves can be as
    # large as a P wave but are not phase-locked to the QRS, so they average
    # out and correlate poorly with the template.
    present = bool(ratio > 0.045 and consistency > 0.60)
    return present, float(ratio), consistency


def _atrial_activity(x: np.ndarray, peaks: np.ndarray, fs: float, rr_median_ms=None):
    """Detect organised atrial activity independent of the ventricular rhythm.

    The QRS complexes are blanked and interpolated over, then the residual is
    autocorrelated. Atrial flutter produces a sawtooth at 250-350/min that
    survives QRS removal and shows a strong autocorrelation peak at the
    corresponding lag.

    The lag search is capped at ``RR / 1.7``. Without that cap, a fast
    ventricular rhythm places its own RR periodicity inside the flutter lag
    range and the residual T-wave train is misread as flutter -- which is
    exactly what happens with SVT at 200 bpm.
    """
    resid = np.asarray(x, dtype=float).copy()
    if peaks.size:
        half = int(0.075 * fs)
        mask = np.ones(len(resid), dtype=bool)
        for p in peaks:
            mask[max(0, int(p) - half) : min(len(resid), int(p) + half)] = False
        if mask.sum() < fs:
            return None, None
        idx = np.arange(len(resid))
        resid = np.interp(idx, idx[mask], resid[mask])

    resid = remove_baseline(resid, fs, cutoff=1.0)
    resid = resid - resid.mean()
    if len(resid) < int(2 * fs):
        return None, None

    ac = np.correlate(resid, resid, mode="full")[len(resid) - 1 :]
    if ac[0] <= 0:
        return None, None
    ac = ac / ac[0]

    lag_lo = int(60.0 / 380.0 * fs)  # 380/min
    lag_hi = int(60.0 / 200.0 * fs)  # 200/min
    if rr_median_ms:
        lag_hi = min(lag_hi, int(rr_median_ms / 1000.0 * fs / 1.7))
    if lag_hi - lag_lo < 3 or lag_hi >= len(ac):
        return None, 0.0

    seg = ac[lag_lo:lag_hi]
    best = int(np.argmax(seg)) + lag_lo
    return float(60.0 * fs / best), float(ac[best])


def _baseline_occupancy(x: np.ndarray) -> float:
    """Fraction of samples whose magnitude exceeds 20% of the peak deflection.

    An organised rhythm spends most of its time on an isoelectric baseline, so
    this is low (typically 0.10-0.30). Ventricular fibrillation has no baseline
    at all, so it is high (typically > 0.45)."""
    z = np.asarray(x, dtype=float)
    z = z - np.median(z)
    peak = float(np.percentile(np.abs(z), 99.0))
    if peak <= 1e-9:
        return 0.0
    return float(np.mean(np.abs(z) > 0.20 * peak))


def _template_match(beats: np.ndarray) -> float | None:
    """Mean correlation of each beat against the median beat."""
    if beats.shape[0] < 3:
        return None
    template = np.median(beats, axis=0)
    tz = template - template.mean()
    dt = np.linalg.norm(tz)
    if dt < 1e-9:
        return None
    corrs = []
    for row in beats:
        rz = row - row.mean()
        d = np.linalg.norm(rz) * dt
        if d > 1e-9:
            corrs.append(float(np.dot(rz, tz) / d))
    return float(np.mean(corrs)) if corrs else None


def _assess_quality(x: np.ndarray, fs: float, m: Measurements) -> None:
    notes, score = [], 1.0

    hf = bandpass(x, fs, 40.0, min(90.0, fs / 2 * 0.9))
    hf_ratio = float(np.std(hf) / (np.std(x) + 1e-9))
    if hf_ratio > 0.55:
        notes.append("High-frequency noise dominates the trace.")
        score -= 0.35

    if m.duration_s < 5.0:
        notes.append(f"Strip is only {m.duration_s:.1f}s; 10s recommended.")
        score -= 0.20

    if m.beat_count and m.beat_count < 4 and m.baseline_occupancy is not None:
        if m.baseline_occupancy < 0.4:
            notes.append("Fewer than 4 beats detected; rate estimate is imprecise.")
            score -= 0.25

    if float(np.std(x)) < 1e-4:
        notes.append("Signal is nearly flat; check lead placement or input scaling.")
        score = 0.05

    m.signal_quality = float(np.clip(score, 0.0, 1.0))
    m.quality_notes = notes


def analyse(x: np.ndarray, fs: float, *, mains_hz: float = 60.0) -> Measurements:
    """Run the full measurement suite on one ECG strip (in mV)."""
    x = np.asarray(x, dtype=float)
    clean = preprocess(x, fs, mains_hz)

    m = Measurements()
    m.duration_s = float(len(x) / fs)
    m.baseline_occupancy = _baseline_occupancy(clean)

    peaks = detect_qrs(clean, fs)
    m.r_peaks = [int(p) for p in peaks]
    m.beat_count = int(peaks.size)

    if peaks.size >= 2:
        rr_ms = np.diff(peaks) / fs * 1000.0
        _rr_stats(m, rr_ms)
        m.qrs_duration_ms, m.qrs_p90_ms, m.qrs_wide_fraction = _qrs_duration(clean, peaks, fs)

        beats, taxis = _beat_matrix(clean, peaks, fs)
        if beats.shape[0]:
            m.p_wave_present, m.p_wave_ratio, m.p_wave_consistency = _p_wave(
                beats, taxis, rr_median_ms=float(np.median(rr_ms))
            )
            m.qrs_template_match = _template_match(beats)

    rr_med = float(np.median(np.diff(peaks) / fs * 1000.0)) if peaks.size >= 2 else None
    atrial_bpm, organisation = _atrial_activity(clean, peaks, fs, rr_median_ms=rr_med)
    m.atrial_rate_bpm = atrial_bpm
    m.atrial_organisation = organisation
    m.atrial_activity_organised = bool(organisation is not None and organisation > 0.10)
    if atrial_bpm and m.heart_rate_bpm:
        m.flutter_ratio = float(atrial_bpm / m.heart_rate_bpm)

    _assess_quality(clean, fs, m)
    return m
