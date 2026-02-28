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
- Choose model, aspect ratio, and output resolution
- Progress bar and real-time log
- Remembers your folders and settings between sessions
- Optional secure API key saving via the OS keychain (Windows Credential Manager / macOS Keychain / Linux Secret Service) — never stored in plaintext
- Skips already-processed images (with optional overwrite)
- Handles rate limits automatically with retries

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

## Models

| Model | Speed | Max Resolution | Notes |
|---|---|---|---|
| `gemini-3-pro-image-preview` | Slower | 4K | Highest quality |
| `gemini-2.5-flash-image` | Faster | 2K | Cheaper per image |

## Pricing

Each generated image costs approximately **$0.04** (1,290 output tokens at $30/1M tokens for the flash model). Check [Google AI pricing](https://ai.google.dev/pricing) for current rates.

## Tips

- **Aspect ratio** — set this to match what you want in the output, not the input. The model will crop/compose accordingly.
- **Resolution** — `1K` is fine for web use. Use `2K` or `4K` for print or large displays. Higher resolutions cost more and take longer.
- **Prompt** — be specific about style, framing, lighting, and context. The model uses both your image and the prompt together.
- **Rate limits** — the tool retries automatically on 429 errors. For large batches, the 5-second delay between requests helps stay within limits.

## Supported input formats

`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.gif`

## License

MIT
