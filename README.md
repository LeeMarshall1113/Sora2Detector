# 🧪 Sora2 Batch Tester

**Batch testing harness for Sora2Detector** — runs all detection backends (Metadata, Watermark, Audio, Video) in parallel across a dataset and generates a unified Markdown report with accuracy metrics.

This tool reproduces **frontend decision logic** exactly:

> If **Metadata** or **Watermark** “light up” → **AI**
> Else → weighted Audio/Video (0.6/0.4) average threshold → **AI / Not AI**

Then automatically maps those binary results into the three evaluation classes:

* **real**
* **sora-watermark**
* **sora-no-watermark**

---

## 🔧 Features

* ✅ **Exact parity with frontend detection logic**
* 🧵 Concurrent backend execution with configurable worker count
* ⏱️ Per-backend **timeout handling and timing metrics**
* 🧩 Automatic **ground-truth inference** from filenames
* 📈 Generates **one consolidated Markdown report**

  * Per-file predictions and reasons
  * Backend coverage table
  * Confusion matrix & per-class metrics
  * Highlighted errors and empty outputs
* 📂 Accepts both **folders and single files**
* 🪶 Portable — single script, no dependencies beyond Python stdlib

---

## 🧰 Requirements

* **Python 3.9+** (tested on 3.12)
* `backend.py`, `backend-2.py`, `backend-3.py`, and `backend-4.py` located in:

  * The same directory as this script *(default)*, or
  * A custom path passed via `--backend-dir`.

Each backend should print short summary output (e.g., JSON or a confidence line) that includes probability or detection terms like “detected watermark”, “AI-generated”, or a confidence percentage.

---

## 🚀 Usage

### Basic example

```bash
python batch-tester.py --folder "C:\Videos\batch_set" --out "C:\Videos\results" --verbose
```

### Specify backends

```bash
python batch-tester.py \
  --folder "C:\TestClips\SoraSet1" \
  --backend-dir "C:\Repos\Sora2Detector" \
  --out "C:\Results\batch_report.md" \
  --timeout 150 --workers 6 --verbose
```

### Run on one file

```bash
python batch-tester.py --root "C:\Test\real-1.mp4" --out report.md
```

### Use patterns or custom extensions

```bash
python batch-tester.py --folder "D:\Data" --pattern "**/*.mp4" --out results
```

---

## 🧩 Filename Convention (Ground Truth)

The tester automatically infers **ground truth (GT)** labels from filenames:

| Example filename          | Inferred GT label   |
| ------------------------- | ------------------- |
| `real-1.mp4`              | `real`              |
| `sora-watermark-1.mp4`    | `sora-watermark`    |
| `sora-no-watermark-1.mp4` | `sora-no-watermark` |

> Use `--strict-labels` to skip any files that don’t match these patterns.

---

## 🧠 Decision Logic

The same as your **web frontend / controller logic**:

```python
# Stage 1 (short-circuit)
if Metadata or Watermark "light up":
    ai_binary = "AI"
    ai_source = "Metadata" or "Watermark"

# Stage 2 (fallback)
weighted = (0.6 * Audio + 0.4 * Video)
if weighted > 0.5:
    ai_binary = "AI"
    ai_source = "Weighted"
else:
    ai_binary = "Not AI"
```

Then:

| ai_binary | ai_source            | Final 3-Class Label |
| --------- | -------------------- | ------------------- |
| Not AI    | any                  | real                |
| AI        | Metadata / Watermark | sora-watermark      |
| AI        | Weighted             | sora-no-watermark   |

---

## 📊 Output Report

A single Markdown report with all results is generated at `--out`.

### Example:

```
# Sora2 Batch Report

- Generated: 2025-11-03 17:00:08
- Source: C:\Users\leema\Videos\batch_set
- Threshold: 0.50, Weights: Audio 0.60, Video 0.40

## Backend Coverage

| Backend | OK | TIMEOUT | ERROR | EMPTY_OUTPUT |
|----------|---:|---:|---:|---:|
| Metadata | 3 | 0 | 0 | 0 |
| Watermark | 2 | 0 | 0 | 1 |
| Audio | 3 | 0 | 0 | 0 |
| Video | 3 | 0 | 0 | 0 |

## Per-file Results

| File | GT | Pred | AI? | Source | Audio% | Video% | Reason | Runtime (ms) | Correct |
|------|----|------|-----|---------|---------|---------|---------|---------------|----------|
| real-1.mp4 | real | real | Not AI | Weighted | 43 | 25 | weighted_avg <= 0.5 | 125460 | ✅ |
| sora-no-watermark-1.mp4 | sora-no-watermark | sora-no-watermark | AI | Weighted | 95 | 100 | weighted_avg > 0.5 | 120096 | ✅ |
| sora-watermark-1.mp4 | sora-watermark | sora-no-watermark | AI | Weighted | 97 | 100 | weighted_avg > 0.5 | 108214 | ❌ |
```

---

## 🧪 Backend Integration

Each backend is an independent Python executable. Example minimal `backend.py`:

```python
import sys, random, json

if __name__ == "__main__":
    path = sys.argv[1]
    conf = random.random()
    result = {
        "summary": f"confidence: {conf:.2f}",
        "short": "AI-generated" if conf > 0.5 else "authentic",
        "confidence": conf,
    }
    print(json.dumps(result))
```

Each backend’s output is parsed using:

* JSON (keys: `summary`, `short`, `confidence`, `result`)
* or plain text heuristics (`confidence: 0.93`, “AI-generated”, etc.)

---

## ⚙️ Command Options

| Flag              | Description                        |
| ----------------- | ---------------------------------- |
| `--folder`        | Directory containing video samples |
| `--root`          | Single file or directory           |
| `--backend-dir`   | Location of backend scripts        |
| `--workers`       | Number of concurrent file workers  |
| `--timeout`       | Max seconds per backend process    |
| `--pattern`       | Glob (e.g., `**/*.mp4`)            |
| `--extensions`    | Comma-separated list (`.mp4,.mov`) |
| `--out`           | Output file or directory           |
| `--strict-labels` | Skip files without GT              |
| `--verbose`       | Show per-backend timing logs       |

---

## 📈 Accuracy Metrics

At the bottom of the Markdown report:

* **Accuracy:** global (correct / total)
* **Confusion matrix:** `real`, `sora-watermark`, `sora-no-watermark`
* **Precision / Recall / F1 / Support** per class

These metrics allow quick regression checks between model updates or backend improvements.

---

## 🧱 Project Layout

```
Sora2Detector/
│
├── backend.py
├── backend-2.py
├── backend-3.py
├── backend-4.py
├── batch-tester.py
└── reports/
    └── batch_report.md
```

---

## 🧾 Example Workflow

1. **Prepare dataset**

   ```
   batch_set/
   ├── real-1.mp4
   ├── sora-watermark-1.mp4
   └── sora-no-watermark-1.mp4
   ```

2. **Run tester**

   ```bash
   python batch-tester.py --folder batch_set --out results --workers 4 --timeout 120 --verbose
   ```

3. **Inspect output**

   * `results/batch_report.md`
   * Confusion matrix for quick accuracy comparison
   * Empty-output warnings to debug backends

4. **Refine thresholds / backends**

   * Adjust confidence cutoffs inside backends
   * Rerun test suite to measure consistency

---

## 🧩 Future Enhancements

* [ ] CSV/JSON report export (`--out-format csv`)
* [ ] Aggregated ROC & precision-recall plots
* [ ] Auto-threshold sweep testing
* [ ] Web dashboard integration with the frontend
* [ ] “binary-only” mode (AI/Not AI only)

---

## 🧑‍💻 Contributing

1. Fork the repo and create a branch:

   ```bash
   git checkout -b feature/new-backend
   ```
2. Follow naming convention: `backend-<n>.py`
3. Implement stdout output (JSON or plain text with a confidence)
4. Test locally with:

   ```bash
   python batch-tester.py --folder testset --verbose
   ```
5. Submit a PR with backend docstring describing I/O behavior.

---

## ⚖️ License

Apache 2.0 — freely usable for research, evaluation, and integration into AI detection systems.
Attribution recommended for derivative detection pipelines.

---

### 💡 Maintainer

**Lee Marshall**
University of Central Florida
[GitHub](https://github.com/LeeMarshall1113) • [UCF CS '26](https://www.ucf.edu)
If you want to help maintain this, let me know and we can talk
