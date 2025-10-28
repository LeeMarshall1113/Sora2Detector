# Sora2Detector
---

```markdown
# 🧠 Sora2Detector — Modular Multi-Modal Video Analysis Framework

**Sora2Detector** is a first-generation framework for *multi-modal, multi-stage video analysis*.  
It coordinates a sequence of independent detection modules — including metadata extraction, watermark localization, audio feature analysis, and motion-based video evaluation — through a unified orchestration layer and reproducible API.

The system is designed to support **research-grade reproducibility** and **transparent cross-modality evaluation**.  
Each backend operates as an isolated Python process, producing interpretable outputs that the controller aggregates into a structured summary.  
This structure enables direct comparison between heterogeneous detection pipelines (e.g., classical OCR vs. CNN-based watermark detectors) without manual integration work.

Sora2Detector combines these design goals:
- **Modularity:** each detector is an independent Python file with a standard interface.  
- **Reproducibility:** identical results from command-line or API contexts.  
- **Transparency:** all stdout/stderr are preserved and viewable through the web interface.  
- **Scalability:** easily extended to additional modalities (e.g., OCR, NLP, model fingerprinting).  

---

## 📂 Project Structure

```

Sora2Detector/

├─ api.py                # FastAPI web server (upload + analysis API + static UI)

├─ controller.py         # Orchestrator that runs all backend detector

├─ backend.py            # Metadata analyzer

├─ backend-2.py          # Watermark detector

├─ backend-3.py          # Audio classifier

├─ backend-4.py          # Video/motion detector

├─ static/

│  └─ index.html         # Simple upload UI for the browser

├─ requirements.txt      # Python dependencies

└─ README.md             # You are here

````

---

## ⚙️ Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI server

```bash
uvicorn api:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🌐 Web Interface

Once running, the system serves a clean web UI where you can:

1. Upload an `.mp4` file
2. Wait for the backend detectors to execute
3. View concise summaries for each detector
4. Expand sections for full stdout/stderr logs

If a detector doesn’t produce any output, it automatically returns:

```
AI
```

to indicate that no explicit detection was found (useful for distinguishing “silent” models from “errors”).

---

## 🧠 How It Works

### 1. Upload endpoint (`/analyze`)

`api.py` handles incoming uploads via FastAPI:

* Streams `.mp4` files to a temp folder (no large in-memory buffers)
* Passes the file path to `controller.py`
* Waits for all detectors to finish
* Returns structured JSON of all results

### 2. Controller orchestrator

`controller.py` runs each backend sequentially:

```python
backends = [
    ("Metadata", "backend.py"),
    ("Watermark", "backend-2.py"),
    ("Audio", "backend-3.py"),
    ("Video", "backend-4.py"),
]
```

For each backend:

* Executes `[python backend-x.py <video>]`
* Captures stdout and stderr
* Extracts a **short summary** (based on JSON, keywords, or first lines)
* Returns full output for display

Outputs answer based on whether or not it is AI
If a timeout or error occurs → a descriptive message is returned.

### 3. Web frontend (`static/index.html`)

* Uploads videos via `fetch('/analyze', { method: 'POST', body: FormData })`
* Renders each detector’s result in a neat card UI with expandable logs
* Uses no frameworks — pure HTML, CSS, and JavaScript for simplicity

---

## 🧩 Example JSON Response

Example returned by `/analyze`:

```json
{
  "file": "C:\\videos\\demo.mp4",
  "results": {
    "Metadata": {
      "short": "duration=3.2s, codec=h264",
      "full": "... full ffprobe output ..."
    },
    "Watermark": {
      "short": "AI",
      "full": "No visible watermark detected"
    },
    "Audio": {
      "short": "speech: 97% confidence",
      "full": "... full classifier output ..."
    },
    "Video": {
      "short": "motion score 0.81",
      "full": "... full detector output ..."
    }
  }
}
```

---

## 🧰 Customization

### Change timeout per backend

Edit in `controller.py` inside `run_backend()`:

```python
timeout=120  # seconds
```

### Add new detectors

Add a new entry in `BACKENDS`:

```python
("MyNewDetector", "backend-5.py"),
```

Ensure your script accepts a single video path as its first argument and **prints** or **returns JSON**.

### Backend return guideline

Each backend should **always print one summary line** or JSON object at the end, e.g.:

```python
print(json.dumps({"summary": "Watermark found (confidence 0.87)"}))
```

If nothing is printed, the controller will automatically substitute `"AI"`.
This is due to the fact that the output will print 'not Ai" when it isn't AI
But when it does detect AI it returns nothing
This will be fixed in the future but I was under a time constraint at my hackathon
For all intents and purposes it functions as it ought to, it just has a bandaid

---

## ⚡ Command Line Mode

You can run the controller manually (without the web server):

```bash
python controller.py path/to/video.mp4
```

You’ll see formatted output like:

```
=== Sora2Detector Combined Analysis ===
File: path/to/video.mp4

Metadata: duration=3.2s, codec=h264
Watermark: AI
Audio: speech detected
Video: motion score 0.81

=== End of Analysis ===
```

---

## 🛡️ Notes

* Default timeout per backend: **120 s**
* Temporary uploads are deleted automatically after processing
* Designed for Windows / Linux / macOS (Tested only on windows with
  AMD hardware, will test other options in future)
* Works with Python ≥ 3.10 (tested on 3.12 and 3.13)

---

Attributtions
Data collected from 
https://live.ece.utexas.edu/research/chug/index.html#download 
Sora2
https://dagshub.com/datasets/audio/
## 🧩 Tech Stack

| Component            | Description                              |
| -------------------- | ---------------------------------------- |
| **Python 3.12+**     | Core runtime                             |
| **FastAPI**          | Web server for uploads and API responses |
| **Uvicorn**          | ASGI server (hot reload)                 |
| **HTML**             | Lightweight frontend                     |
| **subprocess.run()** | Isolated backend execution               |
| **JSON**             | Common exchange format                   |

---

## 🚀 Future Enhancements

* More accurate AI models trained with more data
* Fix the bug with the output of some being blank when it should say ai (high priority bugfix)
* fix the output of the precentages to be what the ai says (high priority bugfix)
* Have a lightweight option availible
* Make the website function properly for hosting
* Option to select between different video AI detection (IE sora2 vs Veo 3)
* Create an Omni detector for a large variety of ai detectors
* Ensure other hardware is supported (ie Intel and Nvidia)
* Have it work on social media platforms to help seemlessly check for AI
* Have the way I trained the AI be made open to allow other people do the same
* Deal with potential poison pill of Sora2
* 

---

## 🧾 License

This project is provided as-is under the **Mad License 1.0**.
Use, modify, and distribute freely with attribution for small scale projects.
Large scale projects ie corporate ventures contact the email listed in the license

---

### 💡 Maintainer

**Lee Marshall**
University of Central Florida
[GitHub](https://github.com/LeeMarshall1113) • [UCF CS '26](https://www.ucf.edu)
If you want to help maintain this, let me know and we can talk
