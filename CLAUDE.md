# CLAUDE.md

## Project overview

Single-file Python GUI tool (`generate_mockups.py`) that batch-processes a folder of images through the Google Gemini image-generation API and saves the results. Built with `tkinter` (stdlib) + `google-genai` + `Pillow`.

## Architecture

Everything lives in one file — no package structure, no modules. Keep it that way unless the scope grows significantly.

- **`collect_images()`** — scans input folder for supported image types
- **`process_image()`** — sends one image + prompt to the Gemini API, saves the result
- **`_RateLimitRetry`** — internal exception used to signal 429 retries from `process_image` back up to the worker loop
- **`App(tk.Tk)`** — the entire GUI; the worker runs in a background `threading.Thread`; all UI updates from the worker go through `self.after(0, ...)` to stay on the main thread
- **Prompt sets** — named, ordered groups of prompts stored in `prompt_sets.json`; each image is processed through every prompt in the selected set
- **Output naming** — single-prompt mode: `{stem}_mockup.png`; prompt-set mode: `{stem}_1.png`, `{stem}_2.png`, etc. (numbered by prompt order)

## Key constraints

- **`threading.Event` for stop signal** — use `self._stop_event` (not a plain bool) to signal the worker thread to stop
- **`self.after(0, fn, arg)`** for all UI updates from the worker thread — never touch tkinter widgets directly from the background thread
- **`ttk` widgets for buttons** — `tk.Button` ignores `bg`/`fg` on macOS; use `ttk.Button` instead
- **`TkFixedFont`** for the log widget — cross-platform monospace; don't hardcode `Consolas` or `Menlo`
- **API key** — read from `GOOGLE_API_KEY` env var or the GUI field; never log or print it
- **Keychain** — when "Save API key" is checked, the key is stored via `keyring` (Windows Credential Manager / macOS Keychain / Linux Secret Service); it is **never** written to `config.json` or any file on disk

## Dependencies

```
google-genai>=1.0.0
Pillow>=10.0.0
keyring>=24.0.0
tkinter  # stdlib, no install needed
```

## Running locally

```bash
# install deps
pip install -r requirements.txt

# set API key
export GOOGLE_API_KEY="your_key_here"   # macOS/Linux
$env:GOOGLE_API_KEY = "your_key_here"   # Windows PowerShell

# run
python generate_mockups.py
```

## Model notes

- `gemini-3-pro-image-preview` — highest quality, supports 1K/2K/4K output
- `gemini-2.5-flash-image` — faster and cheaper, max 2K
- Both use `client.models.generate_content()` with `types.ImageConfig` for aspect ratio and resolution
- Imagen models (`imagen-3`, `imagen-4`) use a different API (`generate_images`) and don't accept a reference image — not supported here
