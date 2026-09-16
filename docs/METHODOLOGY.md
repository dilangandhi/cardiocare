# Methodology

This document states precisely what has been validated, what has not, and where
the numbers come from. Read it before quoting any figure from this repository.

---

## 1. What the system does

An ECG reaches the system either as an image (a photo or scan of a printout) or
as a raw signal file. Images are digitised back into a signal first. The signal
is then measured, and those measurements drive a rule engine. In parallel, when
trained weights are present, a convolutional network reads the image directly.
The two results are compared and reported together.

```
image ──> digitiser ──┬──> CNN (reads the image)      ──┐
                      │                                 ├──> fused finding
signal ───────────────┴──> measurements ──> rule engine ┘
```

The two paths are deliberately independent. The CNN sees pixels; the rule engine
sees heart rate, RR statistics, QRS duration and atrial activity. They can only
agree by both being right about the underlying rhythm, so agreement carries
information and disagreement is reported rather than averaged away.

---

## 2. Class definitions

Eight classes, defined in `backend/app/core/taxonomy.py`, which is the single
source of truth for the training pipeline, the service and the frontend.

| Class | Defining signature |
|---|---|
| Normal sinus rhythm | 60–100 bpm, regular RR, P before every QRS |
| Sinus bradycardia | < 60 bpm, otherwise normal conduction |
| Sinus tachycardia | > 100 bpm with preserved P waves |
| Supraventricular tachycardia | Fast, regular, narrow QRS, P waves unresolvable |
| Atrial flutter | Sawtooth atrial activity at 250–350/min |
| Atrial fibrillation | No discrete P waves, irregularly irregular RR |
| Ventricular fibrillation | No identifiable QRS, no isoelectric baseline |
| Ventricular premature beat | Wide premature QRS, compensatory pause |

### A note on the source labels

Sinus tachycardia and bradycardia have no dedicated rhythm annotation in
MIT-BIH. `ml/prepare_data.py` derives them from windows annotated `(N` by
measuring the rate and relabelling above 100 bpm or below 60 bpm. This is a
defensible reading of the annotation but it is a derivation, not a label a
cardiologist wrote down, and it should be described that way.

---

## 3. Measurements

All in `backend/app/core/signal_engine.py`. No learned parameters.

**QRS detection** — Pan-Tompkins (Pan & Tompkins, *IEEE Trans. Biomed. Eng.*
1985): bandpass 5–15 Hz, differentiate, square, 150 ms moving-window integration,
adaptive dual threshold, RR-based search-back for missed beats, then refinement
onto the true R deflection in the raw signal.

**QRS duration** — measured on the analytic (Hilbert) envelope of the 8–25 Hz
band, as a contiguous region grown outward from the R peak until the envelope
falls below 25% of its peak.

Three approaches were tried before this one. Taking every sample above threshold
within a fixed window measured 164 ms for a normal complex, because it captured
neighbouring P and T energy. A derivative threshold measured 22–50 ms, because
the derivative peak at R is far sharper than the complex itself. Thresholding
the raw signal terminated the region at the R-to-S zero crossing. The envelope
has no zero crossings and the contiguity constraint excludes P and T; the
calibrated result is **103 ms for a normal complex**, against an accepted
physiological range of 80–100 ms.

**Atrial activity** — the QRS complexes are blanked and interpolated over, and
the residual is autocorrelated. The lag search is capped at `RR / 1.7`.

That cap is load-bearing. Without it, a supraventricular tachycardia at 200 bpm
places its own RR periodicity inside the flutter lag range, and the residual
T-wave train is read as flutter: measured organisation was **0.75 for SVT against
0.22 for genuine flutter** — the detector fired more strongly on the wrong class.
With the cap, flutter scores 0.21 and every other class scores at or below 0.05.

**P waves** — the search window scales with the RR interval and requires at
least 90 ms of room. Below that the preceding T wave abuts the QRS and anything
found in the window is a T wave. In that case the system reports P waves as not
present, which is exactly how a clinician reads a narrow-complex tachycardia.
Detection is scored on beat-to-beat template correlation rather than amplitude,
because fibrillatory waves can be as large as a P wave but are not phase-locked
to the QRS.

**Baseline occupancy** — the fraction of the strip where the signal magnitude
exceeds 20% of its peak. An organised rhythm spends most of its time on an
isoelectric baseline (0.14–0.30 measured); ventricular fibrillation has no
baseline at all (0.61 measured). The two ranges do not overlap.

---

## 4. What has actually been validated

### 4.1 The digitiser — genuine validation

Render a signal whose properties are known, read it back from the image, compare.
Nothing in the digitiser knows what rate it is supposed to recover, so this is a
real end-to-end test rather than a restatement of assumptions.

| Condition | Median heart-rate error |
|---|---|
| Clean render, 254 DPI | **0.89 bpm** |
| Simulated phone photo (rotation, uneven lighting, blur, sensor noise) | **0.41 bpm** |

Beat counts and RR variability are recovered exactly on all eight reference
strips: `rr_cv` differs by less than 0.002 between source and round trip.

Grid period is recovered to within 0.6 px/mm across 8, 10 and 12 px/mm renders.
Trace coverage exceeds 97% on simulated photographs.

Adding parabolic sub-pixel interpolation to the autocorrelation peak halved the
error. An image scaled to fit a maximum width almost never lands on an integer
grid period, and rounding to the nearest pixel puts a systematic 1.7% error
straight into the time axis and therefore into every rate measurement.

### 4.2 Ink separation must be orientation-agnostic

The first implementation isolated ink by subtracting a Gaussian-blurred
background from the red channel. That works for deflections but silently erases
structures extended in one direction, because an isotropic blur of a long thin
line reproduces the line itself.

The measurable consequence: the flat plateau of the calibration pulse vanished
from the ink mask entirely, so the pulse was never detected and never trimmed.
Its trailing edge then survived into the signal, where Pan-Tompkins read it as a
QRS complex. A single phantom beat at the start of a strip produces one short RR
and one long RR, and on a 10-second bradycardia strip that inflated measured
`rr_cv` **from 0.055 to 0.177** — across the 0.13 irregularity threshold, turning
a regular sinus rhythm into an atrial fibrillation call.

Replacing it with a **black top-hat** (morphological closing minus the image)
fixed it. Closing removes dark structures thinner than the structuring element
regardless of their orientation, so a horizontal plateau and a vertical riser are
treated alike. After the change:

| | before | after |
|---|---|---|
| Calibration pulse detected | no | yes |
| Recovered duration (10 s strip) | 10.55 s | 10.08 s |
| Trace coverage | 0.94 | 0.99 |
| Bradycardia `rr_cv` (source 0.055) | 0.177 | 0.055 |
| Median photo round-trip error | 0.57 bpm | 0.41 bpm |

The general lesson: background-subtraction methods for document images should be
morphological rather than Gaussian whenever the content contains long straight
runs, which ECG traces do by definition — the isoelectric baseline is one.

### 4.3 Input resolution — a real finding

Digitisation systematically widens the measured QRS, and the size of the effect
depends on input resolution.

| Render resolution | Measured normal QRS | Wide-complex false-positive rate |
|---|---|---|
| 152 DPI (6 px/mm) | 120 ms | 0.44 |
| 203 DPI (8 px/mm) | 114 ms | 0.00 |
| 254 DPI (10 px/mm) | 112 ms | 0.06 |
| 305 DPI (12 px/mm) | 112 ms | 0.00 |

At 152 DPI the inflation pushes normal complexes past the 120 ms threshold and
normal beats are classified as ventricular ectopy. Below 180 effective samples
per second the service now emits an explicit warning rather than reporting a
wrong number confidently. Reference strips are rendered at 254 DPI.

This is a general result about digitising ECG images, not an artefact of this
implementation, and it is worth stating in any write-up: **an ECG image below
roughly 200 DPI cannot support reliable QRS duration measurement.**

### 4.4 The rule engine — circular, and not evidence of accuracy

The rule engine scores **99% (198/200)** against `ml/synth.py`.

**Do not quote that number as an accuracy figure.** The synthetic generator and
the rule engine were written against the same physiological assumptions by the
same author. The generator produces atrial fibrillation with irregular RR and no
P waves because that is what atrial fibrillation is; the rule engine detects it
by looking for irregular RR and absent P waves. The test confirms that the
plumbing works and guards against regressions. It says nothing about performance
on real recordings, where morphology varies between patients, leads are placed
imperfectly, and rhythms are not textbook.

Real accuracy requires PhysioNet data. `ml/evaluate.py` scores the rule engine on
the same held-out test split as the CNN, and *that* number is quotable.

### 4.5 Class ordering must be pinned, not assumed

torchvision's `ImageFolder` assigns class indices by sorting folder names
alphabetically. For this taxonomy the two orders disagree on five of eight
classes:

| index | alphabetical | taxonomy |
|---|---|---|
| 0 | afib | normal |
| 2 | bradycardia | tachycardia |
| 3 | **normal** | **bradycardia** |
| 5 | tachycardia | afib |

A model trained without remapping reaches full accuracy and reports nothing
wrong, because the labels are self-consistent throughout training. The failure
appears only at serving time, where the service maps output index 3 to
bradycardia while the model means normal. Observed symptom: a normal sinus strip
at 84 bpm reported as "Sinus bradycardia", with the dual-path check firing
`model and rules disagree` on every single strip.

`ml/train.py` now remaps ImageFolder's indices to taxonomy order and asserts the
result. Checkpoints trained before the fix can be repaired at export time by
permuting the rows of the final linear layer, which reorders the outputs without
disturbing the learned features.

Worth noting what caught this: nothing in training did. Validation accuracy was
99.4%. It was the **independent rule engine disagreeing on every strip** that
exposed it. A single-path classifier would have shipped this bug invisibly.

### 4.6 Digitisation smooths chaotic rhythms

Following an ink trace column by column interpolates a continuous path through
overlapping ink. On disorganised waveforms this raises apparent morphological
organisation: beat-template correlation for ventricular fibrillation measured
**0.03 on the source signal but 0.52 after a round trip through an image**.

The `template_chaotic` threshold was therefore raised from 0.45 to 0.60, which is
still far below the 0.80–1.00 range every organised rhythm occupies. Before the
change, digitised ventricular fibrillation was classified as atrial fibrillation
— an error between two conditions with completely different management.

Baseline occupancy is unaffected by the smoothing (0.62 source, 0.60 digitised)
and is the more reliable of the two ventricular-fibrillation criteria.

### 4.7 Known failure mode: 2:1 atrial flutter

Atrial flutter conducted 2:1 is misclassified as SVT or sinus tachycardia. When
every second flutter wave falls inside a QRS complex, blanking the QRS also
removes half the sawtooth, and the surviving waves occur at the ventricular rate
— which is precisely what makes the two indistinguishable.

This is not a shortcoming peculiar to this implementation. 2:1 flutter
masquerading as SVT is a well-documented trap in clinical electrocardiography,
which is why the standard bedside manoeuvre is to slow AV conduction and unmask
the flutter waves. A surface trace analysed in isolation cannot always resolve it.

---

## 5. Training methodology

### 5.1 The split is by record, not by window

`ml/prepare_data.py` assigns whole recordings to train, validation or test. No
recording contributes windows to more than one split.

This matters more than any other choice in the pipeline. Windows cut from one
recording share a patient, a lead placement and a beat morphology. Shuffling
windows before splitting places near-duplicates in both train and test, and the
resulting accuracy measures memorisation rather than generalisation. Published
ECG classification results are frequently inflated this way, sometimes by ten
percentage points or more.

If a previously reported figure came from a window-wise split, the record-wise
number will be lower. That is the honest number, and being able to explain the
difference is a strength.

### 5.2 Other choices

- **Transfer learning from ImageNet.** Ten thousand windows is far too few to
  train a ResNet-50 from scratch. Early layers transfer as edge and texture
  detectors regardless of domain.
- **No horizontal flips.** An ECG read backwards is not an ECG — P before QRS
  before T is the entire diagnosis. Augmentation is limited to mild rotation,
  translation, scale, brightness, contrast and erasing, matching how a real
  photograph varies.
- **Photo-realistic augmentation.** Training images include rotated,
  unevenly-lit, blurred and noisy copies, so the model sees what users upload
  rather than only clean renders.
- **Class-weighted loss and balanced sampling.** Ventricular fibrillation
  windows are roughly two orders of magnitude rarer than sinus rhythm.
  Unweighted training simply predicts the majority class.
- **Balanced accuracy is the headline metric.** Plain accuracy on an imbalanced
  eight-class problem is close to meaningless; a model that never predicts VFib
  can still score well on it.
- **The test split is touched once.** Checkpoints are selected on validation
  balanced accuracy.

---

## 6. Reporting behaviour

- Findings below the confidence floor (default 0.45) are reported as
  **inconclusive**, not as a diagnosis.
- Disagreement between the CNN and the rule engine forces an inconclusive
  result and is stated explicitly.
- Poor signal quality proportionally reduces reported confidence.
- Every finding carries its supporting criteria, the criteria it failed, the
  measurements behind them, the model name and weight hash, and the
  digitisation parameters.
- With no model loaded, the service reports rules-only and says so. It never
  fabricates a model prediction.

---

## 7. Limitations

1. **Single lead.** Real diagnosis uses 12 leads. Axis deviation, bundle branch
   block localisation and infarct territory are invisible here.
2. **Rhythm only.** No ischaemia, no infarction, no conduction blocks, no
   electrolyte disturbance, no pacemaker detection.
3. **Eight classes.** A rhythm outside this set will be forced into the nearest
   class. There is no "other" category.
4. **Synthetic reference strips.** The shipped gallery is generated, not
   recorded. It demonstrates the pipeline; it does not evidence accuracy.
5. **2:1 atrial flutter** is misclassified, as described above.
6. **Not validated on real recordings** at the time of writing, because no
   trained model or prepared dataset accompanies this repository.

---

## 8. Regulatory status

None. This is a research prototype. It has not been reviewed or cleared by any
regulatory authority, has not been clinically validated, and must not be used
for diagnosis or to guide treatment.

---

## 9. References

- Pan J, Tompkins WJ. A real-time QRS detection algorithm. *IEEE Trans Biomed Eng.* 1985.
- McSharry PE, Clifford GD, Tarassenko L, Smith LA. A dynamical model for generating synthetic electrocardiogram signals. *IEEE Trans Biomed Eng.* 2003.
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng Med Biol Mag.* 2001.
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation.* 2000.
- He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. *CVPR.* 2016.
