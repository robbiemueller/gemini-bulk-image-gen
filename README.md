# Gemini Bulk Image Generator

A simple desktop GUI for batch-processing images through the [Google Gemini](https://ai.google.dev/) image-generation API. Point it at a folder of images, write a prompt, and it generates a new AI image for each one — saving the results to an output folder of your choice.

Originally built to create product mockups from print designs, but works for any image transformation prompt.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## Features

- Browse to select input and output folders
- Write any prompt — or use the built-in default
- Choose model, aspect ratio, and output resolution
- Progress bar and real-time log
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

You can also paste the key directly into the API Key field in the GUI.

## Usage

```bash
python generate_mockups.py
```

1. Enter your API key (pre-filled if the environment variable is set)
2. Select a **model**, **aspect ratio**, and **resolution**
3. Browse to your **input folder** (containing the images to process)
4. Browse to your **output folder** (where results will be saved)
5. Edit the **prompt** to describe the image transformation you want
6. Click **Run**

Output files are saved as `<original_name>_mockup.png` in the output folder.

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
