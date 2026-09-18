# CardioCare

**Automated ECG rhythm analysis across eight arrhythmia classes.**

Upload a photo of an ECG printout or a raw signal file and get a classification
with the measurements and reasoning that support it.
**[Live demo →](https://cardiocare-y5nk.onrender.com)** (free tier — first load takes ~50s to wake)

> **Research use only — not a medical device.** This software has not been
> clinically validated or reviewed by any regulatory authority. It must not be
> used for diagnosis or to guide treatment.

<!-- Add a screenshot once deployed: ![workstation](docs/figures/workstation.png) -->

---

## What makes this different from a bare classifier

Most ECG projects are a CNN that emits a label. This one runs **two independent
paths and makes them check each other**:

| Path | Reads | Needs training? |
|---|---|---|
| Convolutional network | the image | yes |
| Rule engine | measurements from the digitised signal | **no** |

The CNN sees pixels. The rule engine sees heart rate, RR variability, QRS
duration, P-wave consistency and atrial organisation. They can only agree by
both being right, so the interface reports:

```
Sinus bradycardia          ✓ corroborated by measurements
47 bpm · QRS 103 ms · regular RR · P wave before every QRS
```

or, when they conflict:

```
Inconclusive               ⚠ model and rules disagree
Manual review required
```

Because the rule engine needs no training, **the demo works the moment you clone
it.** The CNN slots in afterwards.

---

## The eight classes

| Class | Signature |
|---|---|
| Normal sinus rhythm | 60–100 bpm, regular, P before every QRS |
| Sinus bradycardia | < 60 bpm |
| Sinus tachycardia | > 100 bpm with P waves preserved |
| Supraventricular tachycardia | fast, regular, narrow QRS, P unresolvable |
| Atrial flutter | sawtooth atrial waves at 250–350/min |
| Atrial fibrillation | no P waves, irregularly irregular RR |
| Ventricular fibrillation | no identifiable QRS, no isoelectric baseline |
| Ventricular premature beat | wide premature QRS, compensatory pause |

---

## Quickstart

### Docker (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/cardiocare.git
cd cardiocare
docker compose up --build
```

Open <http://localhost:7860>.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ml/make_samples.py            # builds the reference gallery
uvicorn app.main:app --app-dir backend --reload --port 7860
```

Click any reference strip in the sidebar, or drop in an ECG image of your own.

---

## How the image pipeline works

```
photo ──> deskew ──> separate trace from grid ──> calibrate ──> follow trace ──> signal
```

**Deskew.** Rotation is estimated from the near-horizontal grid lines via a
probabilistic Hough transform.

**Separate.** ECG paper is printed with a red grid. Red ink reflects red light,
so *in the red channel the grid nearly disappears while the black stylus trace
stays dark*. That single channel choice removes most of the grid before any
thresholding, which is far more robust than detecting and subtracting a grid
pattern afterwards.

**Calibrate.** The grid period is recovered by autocorrelating its projection,
refined to sub-pixel precision with a parabolic fit. One small square is 1 mm,
which at 25 mm/s and 10 mm/mV gives both axes their real units.

**Follow.** The trace is walked column by column. Where a column contains
several ink runs, the run closest to the previous column is chosen, so the
follower stays on one continuous waveform instead of jumping between segments.

### Measured accuracy of the round trip

Render a signal, read it back from the image, compare:

| Condition | Median heart-rate error |
|---|---|
| Clean render at 254 DPI | **0.89 bpm** |
| Simulated phone photo | **0.41 bpm** |

Beat counts and RR variability are recovered exactly on all eight reference
strips, and all eight are correctly classified from their own rendered images.

**Resolution matters more than you would expect.** Below ~200 DPI, digitisation
inflates the measured QRS enough to turn normal complexes into false ventricular
ectopy (false-positive rate 0.44 at 152 DPI, 0.00 at 203 DPI). The service warns
when input resolution is too low rather than reporting a wrong number
confidently. Full numbers in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Training a model

The repository ships **without weights**. Everything runs on the rule engine
until you train one. There is no pretrained model to download — the eight
classes and the ECG-paper input format are specific to this project.

### Fast path: prove the pipeline on synthetic data (~30 min, laptop)

```bash
pip install torch torchvision
python ml/make_synthetic_dataset.py --out data_synth --per-class 400
python ml/train.py --data data_synth --arch mobilenet_v2 --epochs 12 --size 160
python ml/export_onnx.py --checkpoint models/mobilenet_v2_best.pt
cp models/mobilenet_v2.onnx models/mobilenet_v2.json models/
```

This produces a genuinely trained model and lights up the dual-path interface.
It will score near 100% on its own test split and poorly on real ECGs, because
it has learned the generator's assumptions rather than cardiac
electrophysiology. The export is tagged `training_data: synthetic` and the
service displays that warning on every finding, so it cannot be mistaken for a
clinically trained model. **Do not quote its accuracy.**

### Real path: PhysioNet

```bash
pip install -r requirements-ml.txt

# 1. Build the dataset from PhysioNet (downloads automatically)
#    Pulls from mitdb, afdb, vfdb and cudb -- see below for why four.
python ml/prepare_data.py --out data --window 10 --stride 5

# 2. Train
python ml/train.py --data data --arch resnet50 --epochs 30

# 3. Evaluate both paths on the same held-out test set
python ml/evaluate.py --data data --checkpoint models/resnet50_best.pt

# 4. Export for serving
python ml/export_onnx.py --checkpoint models/resnet50_best.pt
```

Drop the resulting `.onnx` into `models/` and restart. The header lights up with
the model name and weight hash, and findings become dual-path.

No GPU? Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) and run
it on a free Colab T4.

Architectures: `resnet50`, `efficientnet_b0`, `mobilenet_v2`, `densenet121`.

### Four databases, not one

`(VFL` and `(SVTA` each appear in exactly **one** MIT-BIH recording. Since
records are split whole, that recording lands entirely in train or entirely in
test, leaving the other split with no examples of the class at all. Ventricular
fibrillation is therefore drawn from `vfdb` and `cudb`, and atrial fibrillation
and flutter from `afdb`, giving 128 recordings across four databases. Each is
split separately so all four reach train, validation and test.

Ventricular tachycardia windows are dropped rather than folded into the
ventricular fibrillation class — VT has organised wide complexes and VF has
none, and merging them would teach the model that an organised rhythm is a
chaotic one.

### The split is by record, not by window

Windows cut from one recording share a patient, a lead placement and a beat
morphology. Shuffling before splitting puts near-duplicates in both train and
test, and the resulting accuracy measures memorisation. Published ECG results
are frequently inflated this way. `prepare_data.py` assigns whole recordings to
one split, so test performance reflects genuinely unseen patients — and reports
a lower, honest number.

---

## Project layout

```
backend/app/core/
  taxonomy.py       the 8 classes; single source of truth
  signal_engine.py  Pan-Tompkins detection and clinical measurements
  rules.py          transparent rule engine, every threshold documented
  digitize.py       image -> calibrated signal
  render.py         signal -> calibrated ECG paper
  fusion.py         combines CNN and rules, flags disagreement
  report.py         PDF report generation
backend/app/main.py FastAPI service
frontend/           the workstation UI (no build step)
ml/
  synth.py          physiology-based ECG generator
  prepare_data.py   PhysioNet -> record-wise split -> rendered images
  train.py          four architectures, balanced accuracy
  evaluate.py       scores CNN and rule engine on the same test set
  export_onnx.py    checkpoint -> ONNX
docs/METHODOLOGY.md what is validated and what is not
```

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | service and model status |
| `GET /api/classes` | the taxonomy |
| `GET /api/samples` | reference strips |
| `POST /api/analyze/image` | upload a photo or scan |
| `POST /api/analyze/signal` | upload CSV / JSON millivolt samples |
| `POST /api/analyze/sample` | analyse a named reference strip |
| `POST /api/report` | render the finding as a PDF |

```bash
curl -F "file=@ecg.png" http://localhost:7860/api/analyze/image | jq .finding
```

Configuration: `CARDIOCARE_MODEL_DIR`, `CARDIOCARE_CONFIDENCE_FLOOR` (default
0.45), `CARDIOCARE_CNN_WEIGHT` (default 0.60).

---

## Deploying

The container honours `$PORT` and defaults to 7860, so the same image runs
anywhere. The runtime uses `onnxruntime` (~50 MB) rather than `torch` (~900 MB),
which keeps the image small enough for free tiers.

| Host | Free tier | Notes |
|---|---|---|
| **Render** | yes | `render.yaml` included. Sleeps after 15 min idle, cold start 30-60 s. |
| **Google Cloud Run** | yes | 2M requests/month. Scales to zero, faster cold starts. Needs a billing account. |
| **Fly.io** | limited | Good latency, small always-on allowance. |
| **Hugging Face Spaces** | **no** | Docker Spaces require PRO ($9/mo); only Static Spaces are free. |

### Render (simplest free option)

1. Push this repository to GitHub.
2. At [render.com](https://render.com): New → Web Service → connect the repo.
3. Runtime **Docker**, instance type **Free**. Render reads `render.yaml`.
4. Deploy. First build takes about 5 minutes.

### Hugging Face Spaces (requires PRO)

Create a Space with SDK **Docker**, copy `deploy/huggingface/README.md` to the
Space root as `README.md` for its YAML frontmatter, then push. The Dockerfile
builds the sample gallery at image-build time, so the demo needs no external
calls at runtime.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest backend/tests -v
```

The suite covers the taxonomy, the measurement engine against known
physiological ranges, the rule engine, the digitiser round trip (including
simulated photographs), fusion behaviour when no model is loaded, and the HTTP
contract.

---

## Honest status

**Validated:**
- Digitiser round trip: 0.41–0.89 bpm median rate error
- All 8 reference strips correctly classified from their rendered images
- Grid recovery within 0.6 px/mm
- QRS duration calibrated to 103 ms for a normal complex (physiological range 80–100 ms)
- Resolution threshold for reliable width measurement, quantified

**Not validated:**
- Clinical accuracy. The rule engine scores 99% against the synthetic generator,
  but the same author wrote both, so that number is circular and must not be
  quoted as accuracy. Real figures require PhysioNet data and `ml/evaluate.py`.
- No trained model ships with this repository.

**Known failure:** atrial flutter with 2:1 conduction is misclassified as SVT —
a documented trap in clinical electrocardiography, not a quirk of this code.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full account.

---

## Licence

MIT. See [LICENSE](LICENSE).

Not a medical device. Not intended for the diagnosis, cure, mitigation,
treatment, or prevention of disease.
