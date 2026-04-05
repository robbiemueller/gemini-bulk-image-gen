# Gemini Bulk Image Generator

A simple desktop GUI for batch-processing images through the [Google Gemini](https://ai.google.dev/) image-generation API. Point it at a folder of images, write a prompt, and it generates a new AI image for each one — saving the results to an output folder of your choice.

Originally built to create framed wall art product mockups from print designs (walnut, oak, and white wood frames in a mid-century modern interior), but works for any image transformation prompt.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## Features

- Browse to select input and output folders
- Write any prompt — or use the built-in default
- **Prompt sets** — group multiple prompts together and run every image through all of them in one batch
- **Two run modes**:
  - **Real-time** — processes images one by one with live progress and ETA
  - **Batch (50% off)** — submits all work as a single async job via the [Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch-api), with higher rate limits and half the cost
- Choose model, aspect ratio, and output resolution
- Progress bar, real-time log, and estimated time remaining
- Remembers your folders and settings between sessions
- Optional secure API key saving via the OS keychain (Windows Credential Manager / macOS Keychain / Linux Secret Service) — never stored in plaintext
- **Cost estimate** — shows estimated cost (with batch discount if applicable) and asks for confirmation before starting
- Skips already-processed images (with optional overwrite)
- Handles rate limits (429) and server overload (503) automatically with exponential backoff, jitter, and a circuit breaker

## Requirements

- Python 3.10 or later
- A Google AI Studio account with **billing enabled**
  - Image generation has no free tier — [enable billing here](https://aistudio.google.com/plan_information)
- A `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey)

## Installation

```bash
git clone https://github.com/your-username/gemini-bulk-image-gen.git
cd gemini-bulk-image-gen
pip install -r requirements.txt
```

> `tkinter` is part of the Python standard library. On some Linux distros you may need to install it separately: `sudo apt install python3-tk`

## Setup

Set your API key as an environment variable before running:

```bash
# macOS / Linux
export GOOGLE_API_KEY="your_key_here"

# Windows (PowerShell)
$env:GOOGLE_API_KEY = "your_key_here"
```

You can also paste the key directly into the **API Key** field in the GUI. Check **Save API key** to store it securely in your OS keychain so you don't need to re-enter it each time — it is never written to disk in plaintext.

## Usage

```bash
python generate_mockups.py
```

1. Enter your API key (pre-filled if the environment variable is set)
2. Select a **model**, **aspect ratio**, and **resolution**
3. Browse to your **input folder** (containing the images to process)
4. Browse to your **output folder** (where results will be saved)
5. Choose a **prompt set** or write a single prompt (see below)
6. Click **Run**

### Single prompt mode

Leave the prompt set dropdown on **"-- Single prompt --"** and write your prompt in the text area. Output files are saved as `<original_name>_mockup.png`.

### Prompt sets (multi-prompt batch)

Prompt sets let you run every image through multiple prompts in one go — useful for generating several different mockup styles per design.

1. Click **New Set…** to open the prompt set editor
2. Give the set a name (e.g. "Etsy Mockups")
3. Click **+ Add** to add prompts — give each a short name and write the prompt text
4. Use **↑ Up** / **↓ Down** to reorder prompts (the order determines output numbering)
5. Click **Save Set**
6. Select your set from the dropdown and click **Run**

Output files are numbered by prompt order: `<original_name>_1.png`, `<original_name>_2.png`, etc.

You can create as many sets as you need and switch between them from the dropdown. The selected set is remembered between sessions.

### Run modes

The **Run mode** dropdown lets you choose how images are processed:

#### Real-time (default)

Images are processed one at a time. You see live progress, per-image status, and an estimated time remaining. Best for smaller batches or when you want to monitor results as they come in.

#### Batch (50% off)

All work is submitted as a single asynchronous job via the [Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch-api). The app uploads your images, submits the job, then polls until it completes and downloads the results.

**Advantages:**
- **50% cheaper** — batch pricing is half the real-time rate
- **Higher rate limits** — avoids the 503/429 errors common in large real-time runs
- **Fire and forget** — submit a large job and let it run

**Trade-offs:**
- No per-image progress during processing — you see the job status (pending → running → succeeded) but not individual images
- Turnaround is variable — usually minutes, but Google targets up to 24 hours
- Results arrive all at once when the job completes

**Resuming a batch job:**
Once a batch job is submitted, the job state is saved locally. You can safely close the app, restart your computer, or come back later — click **Resume Batch** to reconnect to the job, poll for completion, and download results. The Resume button appears automatically on startup if a pending job is detected.

The batch discount is reflected in the cost estimate confirmation dialog.

## Models & pricing

| Model | Speed | Resolutions | Cost per image (approx.) |
|---|---|---|---|
| `gemini-3-pro-image-preview` | Slower | 1K, 2K, 4K | $0.13 (1K/2K) · $0.24 (4K) |
| `gemini-3.1-flash-image-preview` | Fast | 512, 1K, 2K, 4K | $0.045–$0.151 |
| `gemini-2.5-flash-image` | Fastest | 1K, 2K | $0.039 |

All models support 10 standard aspect ratios (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `4:5`, `5:4`, `2:3`, `3:2`, `21:9`). The `gemini-3.1-flash-image-preview` model also supports ultra-wide/tall ratios: `1:4`, `4:1`, `1:8`, `8:1`.

Before each run, the tool shows a **cost estimate** based on the number of images, prompts, model, and resolution — and asks you to confirm before proceeding. Batch mode applies a 50% discount to the estimate.

See [Google AI pricing](https://ai.google.dev/gemini-api/docs/pricing) for current rates.

## Tips

- **Aspect ratio** — set this to match what you want in the output, not the input. The model will crop/compose accordingly.
- **Resolution** — `1K` is fine for web use. Use `2K` or `4K` for print or large displays. Higher resolutions cost more and take longer. `512` (0.5K) is only available on `gemini-3.1-flash-image-preview`.
- **Prompt** — be specific about style, framing, lighting, and context. The model uses both your image and the prompt together.
- **Rate limits / 503 errors** — the tool retries automatically with exponential backoff on both 429 (rate limit) and 503 (server overload) errors. If 5 consecutive items fail with 503 after all retries, the run stops automatically (circuit breaker). For large batches, consider using **Batch mode** which has higher rate limits.
- **Batch mode for large runs** — if you're processing hundreds of images, Batch mode avoids rate-limit issues entirely and costs 50% less.

## Supported input formats

`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.gif`

## License

MIT
