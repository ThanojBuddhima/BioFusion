# BioFusion — Pneumonia Screening

BioFusion is a clinical decision-support prototype from Team GMora for the
BioFusion Hackathon 2026. It uses a ResNet-50 image classifier to screen chest
X-rays for **NORMAL** or **PNEUMONIA**, with Grad-CAM visualizations intended
to help a clinician inspect regions that influenced a prediction.

The repository includes two interfaces:

- A responsive, installable **Streamlit web app** for uploading an X-ray or
  capturing a phone photo.
- An optional **Flask kiosk** that connects a camera and an ESP32-controlled
  light strip, and can generate a PDF report with a QR code.

> **Medical-use notice:** BioFusion is a research and hackathon prototype, not a
> medical device. It is not a diagnosis and must not be used to make treatment
> decisions or replace a qualified healthcare professional. Model results can
> be wrong, and the experiments documented here do not establish clinical
> validity or suitability for any patient population.

## Contents

- [Features](#features)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Run the Streamlit app](#run-the-streamlit-app)
- [Train a model](#train-a-model)
- [Run the kiosk](#run-the-kiosk)
- [Model, data, and evaluation](#model-data-and-evaluation)
- [Deployment](#deployment)
- [Research and project documents](#research-and-project-documents)
- [Limitations and safety](#limitations-and-safety)

## Features

### Streamlit app

The app is located in [`streamlit_app/`](streamlit_app/) and provides:

- **Home:** overview of the screening flow and headline metrics.
- **Screening:** upload a digital X-ray or take a phone photo; select a
  patient/parent or clinician view; review a triage band, probability, and
  next-step guidance.
- **Image checks:** image resolution, focus, contrast, and a colourfulness
  warning are checked before inference. Phone photos use more permissive
  quality thresholds. When a film boundary is found, the photo can be
  perspective-corrected.
- **Explainability:** Grad-CAM heatmaps are available in the clinician view
  and on mobile.
- **Model performance:** shows metrics from `training_metrics.json` when it is
  present, or a documented fallback when it is absent.
- **Dataset analytics:** shows a summary of the pediatric chest X-ray dataset.
- **Mobile/PWA support:** responsive mobile navigation, install prompts,
  manifest, service worker, and an offline page. Offline support does not make
  model inference available without the running server and model.

Triage is based on the model's pneumonia probability and an operating threshold
loaded from the metrics JSON when available. The app includes an uncertainty
band and language that directs users to clinical review rather than presenting
the result as a diagnosis.

### Kiosk

The optional kiosk is in [`kiosk/`](kiosk/). It provides a Flask web interface
for scanning with a USB camera, optional ESP32-controlled illumination/status
LEDs, image inference and Grad-CAM, and downloadable PDF reports linked by QR
code. The kiosk can run without an ESP32 in software-only mode.

## Repository layout

```text
.
├── streamlit_app/
│   ├── app.py                         # Streamlit home page
│   ├── pages/
│   │   ├── 1_Live_Prediction.py       # Screening workflow
│   │   ├── 2_Model_Insights.py        # Metrics and evaluation charts
│   │   └── 3_Dataset_Explorer.py      # Dataset summary
│   ├── utils/
│   │   ├── model.py                   # ResNet-50 loading and inference
│   │   ├── preprocessing.py           # Image preprocessing
│   │   ├── gradcam.py                 # Grad-CAM
│   │   ├── clinical.py                # Triage bands and guidance
│   │   ├── quality.py                 # Image quality checks
│   │   ├── mobile_scan.py             # Phone-photo perspective correction
│   │   ├── device.py                  # Mobile detection and navigation
│   │   └── ui.py                      # Shared visual styles and components
│   ├── static/                        # PWA manifest, service worker, icons
│   └── requirements.txt               # Web app dependencies
├── kiosk/
│   ├── server/                        # Flask app, camera, inference, reports
│   ├── esp32/                         # ESP32 firmware and wiring guide
│   ├── captures/                      # Example captures
│   └── requirements.txt               # Kiosk dependencies
├── train_model.py                     # Training and evaluation entry point
├── phone_photo_aug.py                 # Phone-photo augmentation utilities
├── requirements-train.txt             # Training dependencies
├── training_metrics_*.json            # Saved experiment metrics
├── Pneumonia_Detection_Report.md      # Project report
├── BioFusion_Pneumonia_Detection.ipynb
├── GMora_Notebook.ipynb
├── deploy.sh                          # VPS/systemd deployment helper
└── nginx.conf                         # Nginx reverse-proxy configuration
```

Additional competition, presentation, paper, and proceedings-template
materials are listed under [Research and project documents](#research-and-project-documents).

## Requirements

- Python 3 with `pip` and `venv`.
- For training: network access to download datasets through KaggleHub. Kaggle
  account/API access may be required by Kaggle for dataset downloads.
- A CUDA-capable PyTorch installation is optional; training and inference can
  use CPU, though training will generally be slower.
- For kiosk camera use: a compatible USB camera and OS camera permissions.
- For kiosk hardware controls: an ESP32 and the components described in
  [`kiosk/esp32/README.md`](kiosk/esp32/README.md).

The dependency files specify minimum package versions rather than a fully
locked environment. Use the same PyTorch and torchvision release family.

## Run the Streamlit app

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Or start it from its app directory:

```bash
cd streamlit_app
streamlit run app.py
```

Streamlit prints the local URL when the server starts.

### Checkpoint and metrics files

The screening page looks for these files at the **repository root**:

- `pneumonia_resnet50_best.pth` — trained model state dictionary.
- `training_metrics.json` — metrics and the tuned operating threshold.

When the checkpoint is missing, the app displays a demo-mode warning and uses
ImageNet backbone weights with a newly initialized binary classifier head.
**Those demo predictions are not meaningful for pneumonia screening.** The
committed `training_metrics_*.json` files are experiment records; they are not
automatically treated as the app's active `training_metrics.json`.

## Train a model

Install the training dependencies:

```bash
python -m pip install -r requirements-train.txt
```

Train the default pediatric model:

```bash
python train_model.py
```

By default, the script downloads the Kermany chest X-ray pneumonia dataset
through KaggleHub, trains a ResNet-50 binary classifier, and writes:

- `pneumonia_resnet50_best.pth`
- `training_metrics.json`

The default run uses a seeded, stratified development split carved from the
training data and the dataset's test split for evaluation. It uses ImageNet
normalization at 224 × 224 pixels, weighted cross-entropy, augmentation,
partial fine-tuning of `layer4` and the classifier head, cosine learning-rate
scheduling, early stopping, and a development-set threshold selected to target
a minimum pneumonia recall. CUDA training uses automatic mixed precision when
available.

Useful options:

```bash
python train_model.py --help
python train_model.py --epochs 15 --batch-size 24
python train_model.py --full-finetune
python train_model.py --kfolds 5
python train_model.py --combined-adult
```

- `--full-finetune` unfreezes the full backbone.
- `--kfolds N` runs stratified N-fold cross-validation and records per-fold
  and aggregate metrics. The held-out test set is evaluated for each fold by
  the current script; interpret those results accordingly.
- `--combined-adult` downloads the additional COVID-19 Radiography Database,
  combines eligible adult images with the pediatric images, excludes COVID
  images from the positive class, and enables phone-photo augmentation.
- `--output` and `--metrics-out` change the checkpoint and metrics output
  paths. Other available options include epochs, batch size, learning rates,
  early-stopping patience, and the minimum target recall.

For example, to train a combined-data model directly to the paths expected by
the Streamlit app:

```bash
python train_model.py --combined-adult \
  --output pneumonia_resnet50_best.pth \
  --metrics-out training_metrics.json
```

Dataset terms and access requirements apply to the downloaded data. Do not
redistribute the datasets or trained weights unless their respective terms
permit it.

## Run the kiosk

Install kiosk dependencies and start the Flask server:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r kiosk/requirements.txt
cd kiosk/server
python app.py
```

Open `http://localhost:5050` by default. Configure a trained model by placing
the state dictionary at:

```text
kiosk/server/model/pneumonia_resnet50_best.pth
```

Without it, the kiosk also runs in demo mode and its predictions are not
clinically meaningful. To use a checkpoint produced by the training script,
for example:

```bash
mkdir -p kiosk/server/model
cp pneumonia_resnet50_best.pth kiosk/server/model/pneumonia_resnet50_best.pth
```

### Kiosk environment variables

| Variable | Default | Description |
|---|---:|---|
| `KIOSK_SERIAL_PORT` | `/dev/tty.usbserial-0001` | Preferred ESP32 serial port; the bridge also attempts auto-detection. |
| `KIOSK_BAUD_RATE` | `115200` | ESP32 serial baud rate. |
| `KIOSK_CAMERA_INDEX` | `0` | OpenCV camera index. |
| `KIOSK_HOST` | `0.0.0.0` | Flask bind address. |
| `KIOSK_PORT` | `5050` | Flask port. |
| `KIOSK_DEBUG` | `false` | Flask debug mode; keep disabled outside local development. |

The kiosk creates `kiosk/captures/` and `kiosk/reports/` as needed. Its session
state and report registry are in memory and designed for a single kiosk session,
not multi-user or persistent production operation.

See [`kiosk/README.md`](kiosk/README.md) for the kiosk workflow and
[`kiosk/esp32/README.md`](kiosk/esp32/README.md) for components, wiring, firmware,
and serial commands.

## Model, data, and evaluation

### Model and preprocessing

- Architecture: ResNet-50 pretrained on ImageNet, with a two-class output head.
- Labels: `NORMAL` (index 0) and `PNEUMONIA` (index 1).
- Input: RGB, 224 × 224, ImageNet mean and standard-deviation normalization.
- Grad-CAM: hooks the final ResNet layer (`layer4[-1]`) to create a heatmap.
- Phone photos: the web flow can detect and perspective-correct a film boundary;
  otherwise it uses the photo as captured. The optional combined-data training
  enables phone-photo simulation augmentation.
- Classification threshold: may be tuned on a development split to emphasize
  pneumonia sensitivity. A lower threshold can increase false positives.

### Data

The primary dataset is the public **Chest X-Ray Images (Pneumonia)** dataset
from Kermany et al., distributed via Kaggle at
[paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
The app's dataset page describes its 5,863 images as pediatric and labels them
NORMAL/PNEUMONIA. The optional combined-data experiment also uses the
[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
and maps eligible non-COVID pneumonia images to the positive class.

### Saved experiment records

The repository includes metric JSON files for multiple configurations. Example
headline values below are read from each file's reported `test_report` and
`auc_roc`; experiments use different data, thresholds, or evaluation setups, so
these rows are **not a like-for-like benchmark**.

| Metrics file | Experiment indicated by file/config | Accuracy | Pneumonia recall | AUC-ROC |
|---|---|---:|---:|---:|
| `training_metrics_baseline.json` | Pediatric baseline, default 0.5 threshold | 90.71% | 98.21% | 0.9793 |
| `training_metrics_pediatric_only.json` | Pediatric-only, sensitivity-tuned threshold | 87.02% | 99.74% | 0.9516 |
| `training_metrics_cv.json` | Pediatric cross-validation run; best-fold test report | 92.15% | 98.21% | 0.9516 |
| `training_metrics_combined.json` | Pediatric + adult combined data | 94.47% | 94.27% | 0.9878 |
| `training_metrics_phone.json` | Combined-data run with phone-photo augmentation | 94.25% | 94.32% | 0.9877 |

For the cross-validation record, `training_metrics_cv.json` also contains
five-fold aggregate statistics; its headline row above is the saved best-fold
report, not the five-fold mean. Refer to the JSON files for per-class metrics,
confusion matrices, thresholds, training history, configuration, and available
cross-validation summaries. Metrics are experimental results on the dataset
splits used by the training code—not external, prospective, or clinical
validation.

## Deployment

[`deploy.sh`](deploy.sh) is a VPS helper for the **Streamlit app**. It creates a
Python virtual environment, installs CPU PyTorch and app dependencies, creates
a systemd service running as `www-data` on `127.0.0.1:8502`, then restarts it.
[`nginx.conf`](nginx.conf) contains the reverse-proxy configuration.

Before using the script, review its repository URL, server paths, permissions,
service settings, and proxy configuration for your environment. In particular,
the current script clones `KusalPabasara/BioFusion`; update that URL if this is
not the intended deployment source. The helper does not provision a server,
domain, TLS certificate, or fully configure nginx by itself.

## Research and project documents

- [`Pneumonia_Detection_Report.md`](Pneumonia_Detection_Report.md) — project
  report covering motivation, methodology, results, application, and future
  work.
- [`BioFusion_Pneumonia_Detection.ipynb`](BioFusion_Pneumonia_Detection.ipynb)
  and [`GMora_Notebook.ipynb`](GMora_Notebook.ipynb) — notebooks and exploratory
  work.
- [`PneumoScan_AI_Team_GMora_Report.pptx`](PneumoScan_AI_Team_GMora_Report.pptx),
  [`PneumoScan_AI_Team_GMora_Semifinal_Report.docx`](PneumoScan_AI_Team_GMora_Semifinal_Report.docx),
  and [`GMora_Report.pdf`](GMora_Report.pdf) — presentation and report
  deliverables.
- [`ADScAI_2026_Proceedings_Template/`](ADScAI_2026_Proceedings_Template/) —
  conference paper template, LaTeX source, figures, and references.
- `DELEGATE BOOKLET _ Guidelines and competition framework.pdf` — competition
  materials.
- `PIIS0092867418301545.pdf` — source paper included with the project materials.
- [`report.html`](report.html), [`report_bw.md`](report_bw.md), and
  [`report_content_by_page.md`](report_content_by_page.md) — additional report
  renderings and content.

## Limitations and safety

- This repository is a prototype, not a regulated or clinically validated
  medical device. It must not be used for autonomous diagnosis, treatment, or
  reassurance.
- Demo mode uses ImageNet feature weights with a randomly initialized
  classifier head. Its predictions must not be interpreted as pneumonia
  detection.
- Dataset composition, labels, acquisition devices, and demographics constrain
  generalization. The pediatric dataset and optional combined-data experiments
  do not prove performance in other hospitals, age groups, devices, or
  populations.
- Threshold tuning emphasizes a chosen sensitivity target and can increase
  false positives. Probabilities are model outputs, not calibrated clinical
  risk estimates.
- Grad-CAM is a visualization of model activation/gradient behavior, not proof
  that the model reasoned correctly or attended to clinically valid findings.
- The kiosk is a single-session prototype with local file output and no
  authentication, access control, or production patient-data governance.
- Do not upload identifiable patient information to an unapproved deployment.
  Secure, privacy-preserving infrastructure and clinical, regulatory, and
  security reviews are required before any real-world use.
