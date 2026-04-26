# CLAUDE.md

## Project overview

Single-file Python GUI tool (`generate_mockups.py`) that batch-processes a folder of images through Google Gemini *or* OpenAI image-generation models and saves the results. Built with `tkinter` (stdlib) + `google-genai` + `openai` + `Pillow`.

## Architecture

Everything lives in one file — no package structure, no modules. Keep it that way unless the scope grows significantly.

- **Provider registry** — module-level constants `PROVIDERS`, `MODELS_BY_PROVIDER`, `PROVIDER_FOR_MODEL`, `COST_PER_IMAGE`; the GUI Provider dropdown swaps the model list and the size/quality controls.
- **`collect_images()`** — scans input folder for supported image types
- **`process_image()`** — Gemini: sends one image + prompt to `client.models.generate_content()` and saves the result
- **`process_image_openai()`** — OpenAI: sends one image + prompt to `client.images.edit()`, decodes the base64 PNG, saves the result. Raises the same `_RateLimitRetry` / `_ServiceUnavailable` exceptions so the existing retry loop reuses the Gemini path's logic.
- **`_RateLimitRetry`** / **`_ServiceUnavailable`** — internal exceptions for 429/503 retries with exponential backoff, jitter, and a circuit breaker (5 consecutive 503-failed items stops the run); provider-agnostic
- **`App(tk.Tk)`** — the entire GUI; the worker runs in a background `threading.Thread`; all UI updates from the worker go through `self.after(0, ...)` to stay on the main thread. `_on_provider_change()` reconfigures model list, size/quality dropdowns, run-mode lock, and API-key field.
- **Run modes** — *Real-time* (one API call at a time with progress) or *Batch* (submit all work as a single async Gemini Batch API job at 50% cost). **Batch is Gemini-only** — the run-mode dropdown is disabled and locked to Real-time when OpenAI is selected (OpenAI does not publish a Batch API for image generation).
- **Batch worker** — `_run_batch_worker` uploads images via the File API, builds a JSONL manifest, submits a batch job, polls for completion, then downloads and decodes results. Gemini-only.
- **Batch resilience** — after job creation, state is saved to `batch_state.json` (job name, key→output mapping, output dir). The app can be closed and restarted; a "Resume Batch" button reconnects to the job via `_resume_batch()` / `_resume_batch_worker()`
- **Prompt sets** — named, ordered groups of prompts stored in `prompt_sets.json`; each image is processed through every prompt in the selected set
- **Output naming** — single-prompt mode: `{stem}_mockup.png`; prompt-set mode: `{stem}_1.png`, `{stem}_2.png`, etc. (numbered by prompt order)

## Key constraints

- **`threading.Event` for stop signal** — use `self._stop_event` (not a plain bool) to signal the worker thread to stop
- **`self.after(0, fn, arg)`** for all UI updates from the worker thread — never touch tkinter widgets directly from the background thread
- **`ttk` widgets for buttons** — `tk.Button` ignores `bg`/`fg` on macOS; use `ttk.Button` instead
- **`TkFixedFont`** for the log widget — cross-platform monospace; don't hardcode `Consolas` or `Menlo`
- **API key** — read from the active provider's env var (`GOOGLE_API_KEY` for Gemini, `OPENAI_API_KEY` for OpenAI) or the GUI field; never log or print it
- **Keychain** — when "Save API key" is checked, the key is stored via `keyring` under a per-provider service name (`gemini-bulk-image-gen` / `openai-bulk-image-gen`) so the two keys don't clobber each other. Never written to `config.json` or any file on disk.

## Dependencies

```
google-genai>=1.0.0
openai>=1.0.0
Pillow>=10.0.0
keyring>=24.0.0
tkinter  # stdlib, no install needed
```

## Running locally

```bash
# install deps
pip install -r requirements.txt

# set API key (whichever provider you'll use)
export GOOGLE_API_KEY="your_gemini_key_here"   # macOS/Linux
export OPENAI_API_KEY="your_openai_key_here"
$env:GOOGLE_API_KEY = "your_gemini_key_here"   # Windows PowerShell
$env:OPENAI_API_KEY = "your_openai_key_here"

# run
python generate_mockups.py
```

## Model notes

### Gemini
- `gemini-3-pro-image-preview` — highest quality, supports 1K/2K/4K output
- `gemini-3.1-flash-image-preview` — fast and cheap, supports 512/1K/2K/4K output; also supports extra aspect ratios (1:4, 4:1, 1:8, 8:1)
- `gemini-2.5-flash-image` — oldest, cheapest, up to 2K
- Image size `"512"` has no `K` suffix (API requirement); all others use `"1K"`, `"2K"`, `"4K"`
- All use `client.models.generate_content()` with `types.ImageConfig` for aspect ratio and resolution
- Imagen models (`imagen-3`, `imagen-4`) use a different API (`generate_images`) and don't accept a reference image — not supported here

### OpenAI
- `gpt-image-2` — latest, highest quality; pricing varies by size + quality
- `gpt-image-1.5`, `gpt-image-1`, `gpt-image-1-mini` — older / cheaper variants (no on-file pricing — cost dialog shows "unknown")
- Uses `client.images.edit(image=..., model=..., prompt=..., size=..., quality=...)`; the source image is the reference for the edit
- Sizes: `auto`, `1024x1024`, `1024x1536`, `1536x1024`, `2048x2048`, `3840x2160` (model picks when `auto`)
- Quality: `auto`, `low`, `medium`, `high`
- Response is base64-encoded PNG (`result.data[0].b64_json`); decoded and saved via PIL
- No Batch API for image generation — run-mode is locked to Real-time

### Cost estimate
`COST_PER_IMAGE` dict — Gemini entries keyed by image size (`"1K"`); OpenAI entries keyed by `"{size}/{quality}"` (`"1024x1024/medium"`). Shown to user in a confirmation dialog before each run; falls back to "cost unknown" for combos that aren't on file.
