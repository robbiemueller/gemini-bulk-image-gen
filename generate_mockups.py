"""
generate_mockups.py

Opens a GUI to select an input folder, output folder, and prompt, then sends
each image to the selected provider's image-generation API (Google Gemini or
OpenAI) and saves the resulting PNGs.

Requirements:
    - Gemini: billing enabled on Google AI Studio (no free tier for image gen)
    - OpenAI: an account with image-generation access

Setup:
    1. Enable Gemini billing (if using Gemini): https://aistudio.google.com/plan_information
    2. Set the API key for whichever provider(s) you'll use:
           Windows (PowerShell):
               $env:GOOGLE_API_KEY = "your_gemini_key_here"
               $env:OPENAI_API_KEY = "your_openai_key_here"
           macOS/Linux:
               export GOOGLE_API_KEY="your_gemini_key_here"
               export OPENAI_API_KEY="your_openai_key_here"
    3. Run:
           python generate_mockups.py
"""

import base64
import io
import json
import os
import random
import re
import sys
import tempfile
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
import platform

try:
    import keyring
    import keyring.errors
    _KEYRING_AVAILABLE = True
except ImportError:
    _KEYRING_AVAILABLE = False

from google import genai
from google.genai import types

try:
    from openai import OpenAI
    import openai as _openai_module
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}

# --- Providers ---
PROVIDERS = ["Gemini", "OpenAI"]
DEFAULT_PROVIDER = "Gemini"

GEMINI_MODELS = [
    "gemini-3-pro-image-preview",      # highest quality, 1K–4K
    "gemini-3.1-flash-image-preview",  # fast, cheap, 512–4K
    "gemini-2.5-flash-image",          # oldest, cheapest, up to 2K
]
OPENAI_MODELS = [
    "gpt-image-2",        # latest, highest quality
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",   # cheapest
]
MODELS_BY_PROVIDER = {
    "Gemini": GEMINI_MODELS,
    "OpenAI": OPENAI_MODELS,
}
PROVIDER_FOR_MODEL = {m: "Gemini" for m in GEMINI_MODELS}
PROVIDER_FOR_MODEL.update({m: "OpenAI" for m in OPENAI_MODELS})

DEFAULT_MODEL = GEMINI_MODELS[0]

# --- Gemini sizing controls ---
# Standard ratios supported by all models.
# 3.1-flash also supports 1:4, 4:1, 1:8, 8:1 (appended at the end).
ASPECT_RATIOS = [
    "1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "2:3", "3:2", "21:9",
    "1:4", "4:1", "1:8", "8:1",
]
DEFAULT_ASPECT_RATIO = "1:1"

# "512" is the 0.5K tier — note: no "K" suffix (API requirement).
IMAGE_SIZES = ["512", "1K", "2K", "4K"]
DEFAULT_IMAGE_SIZE = "1K"

# --- OpenAI sizing controls ---
# Source: https://developers.openai.com/api/docs/guides/image-generation
OPENAI_SIZES = ["auto", "1024x1024", "1024x1536", "1536x1024", "2048x2048", "3840x2160"]
DEFAULT_OPENAI_SIZE = "1024x1024"

OPENAI_QUALITIES = ["auto", "low", "medium", "high"]
DEFAULT_OPENAI_QUALITY = "auto"

# --- Per-image cost lookup (USD) ---
# Gemini: keyed by image_size  ("512" / "1K" / "2K" / "4K")
#   Source: https://ai.google.dev/gemini-api/docs/pricing (March 2026)
# OpenAI: keyed by "{size}/{quality}" e.g. "1024x1024/medium"
#   Source: https://developers.openai.com/api/docs/guides/image-generation
#   (gpt-image-2 only — other variants left blank so the cost dialog falls through
#    to the "unknown — proceed?" path)
COST_PER_IMAGE = {
    "gemini-3-pro-image-preview": {
        "1K": 0.134, "2K": 0.134, "4K": 0.24,
    },
    "gemini-3.1-flash-image-preview": {
        "512": 0.045, "1K": 0.067, "2K": 0.101, "4K": 0.151,
    },
    "gemini-2.5-flash-image": {
        "1K": 0.039, "2K": 0.039,
    },
    "gpt-image-2": {
        "1024x1024/low":    0.006, "1024x1024/medium": 0.053, "1024x1024/high": 0.211,
        "1024x1536/low":    0.005, "1024x1536/medium": 0.041, "1024x1536/high": 0.165,
        "1536x1024/low":    0.005, "1536x1024/medium": 0.041, "1536x1024/high": 0.165,
    },
}

DEFAULT_PROMPT = (
    "Create a high-resolution, photorealistic interior product mockup featuring the provided artwork. "
    "The artwork is displayed identically inside three landscape-oriented frames resting on a rustic, "
    "wide-plank wood floor and leaning against a minimalistic, warm off-white wall with a subtle "
    "baseboard. Arrange the frames in a staggered, overlapping line from left to right: the leftmost "
    "frame is dark walnut wood (positioned furthest back), the center frame is natural light oak wood "
    "(slightly overlapping the left frame), and the rightmost frame is bright white wood (positioned "
    "at the front, slightly overlapping the center frame). Use soft, natural window lighting from the "
    "side to cast a realistic glass reflection and glare over the rightmost white frame. Warm, "
    "inviting, mid-century modern interior photography style with a shallow depth of field."
)

MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
    ".tiff": "image/tiff",
    ".gif":  "image/gif",
}

# Seconds to wait between successful requests
REQUEST_DELAY = 5

# Retry settings for rate-limit (429) and service-unavailable (503) errors
MAX_RETRIES = 5
RETRY_BASE_DELAY = 60

# Circuit breaker: stop the run after this many consecutive items fail with 503
CIRCUIT_BREAKER_THRESHOLD = 5

# Batch API settings
BATCH_POLL_INTERVAL = 30   # seconds between status polls
BATCH_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}

# Run mode options
RUN_MODES = ["Real-time", "Batch (50% off)"]

# Sentinel value for the prompt-set dropdown
_SINGLE_PROMPT = "-- Single prompt --"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(folder: Path) -> list[Path]:
    """Return all supported image files in folder (non-recursive)."""
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def sanitize_error(msg: str) -> str:
    """Strip API keys and other secrets from error messages before logging."""
    return re.sub(r'key=[A-Za-z0-9_-]+', 'key=REDACTED', msg)


def extract_retry_delay(error_message: str) -> int:
    match = re.search(r'retry[^\d]*(\d+)', str(error_message), re.IGNORECASE)
    return int(match.group(1)) + 2 if match else RETRY_BASE_DELAY


def process_image(client: genai.Client, model: str, prompt: str,
                  image_path: Path, output_path: Path,
                  aspect_ratio: str = DEFAULT_ASPECT_RATIO,
                  image_size: str = DEFAULT_IMAGE_SIZE) -> bool:
    mime_type = MIME_MAP.get(image_path.suffix.lower(), "image/jpeg")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                    ),
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    img = Image.open(io.BytesIO(part.inline_data.data))
                    img.save(output_path)
                    return True
            return False

        except Exception as exc:
            msg = str(exc)
            if "503" in msg or "UNAVAILABLE" in msg:
                raise _ServiceUnavailable(attempt, sanitize_error(msg))
            if "429" in msg and attempt < MAX_RETRIES:
                wait = extract_retry_delay(msg)
                raise _RateLimitRetry(wait, attempt)
            else:
                raise


def process_image_openai(client, model: str, prompt: str,
                         image_path: Path, output_path: Path,
                         size: str = DEFAULT_OPENAI_SIZE,
                         quality: str = DEFAULT_OPENAI_QUALITY) -> bool:
    """Send one image + prompt to OpenAI's images.edit endpoint, save the result."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {"model": model, "prompt": prompt}
            if size and size != "auto":
                kwargs["size"] = size
            if quality and quality != "auto":
                kwargs["quality"] = quality
            with open(image_path, "rb") as f:
                result = client.images.edit(image=f, **kwargs)

            b64 = result.data[0].b64_json
            if not b64:
                return False
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            img.save(output_path)
            return True

        except Exception as exc:
            msg = str(exc)
            # Detect rate limit / service unavailable across SDK versions.
            rate_limited = (
                _OPENAI_AVAILABLE
                and isinstance(exc, getattr(_openai_module, "RateLimitError", ()))
            ) or "429" in msg
            unavailable = (
                "503" in msg
                or "UNAVAILABLE" in msg.upper()
                or "overloaded" in msg.lower()
            )

            if unavailable:
                raise _ServiceUnavailable(attempt, sanitize_error(msg))
            if rate_limited and attempt < MAX_RETRIES:
                raise _RateLimitRetry(extract_retry_delay(msg), attempt)
            raise


class _RateLimitRetry(Exception):
    def __init__(self, wait: int, attempt: int):
        self.wait = wait
        self.attempt = attempt


class _ServiceUnavailable(Exception):
    """The model is temporarily overloaded (503)."""
    def __init__(self, attempt: int, detail: str = ""):
        self.attempt = attempt
        self.detail = detail


# ---------------------------------------------------------------------------
# Persistent config  (config.json — gitignored, never committed)
# API key is stored in the OS keychain via `keyring`, never in config.json
# ---------------------------------------------------------------------------

CONFIG_FILE = SCRIPT_DIR / "config.json"

_KEYCHAIN_SERVICES = {
    "Gemini": "gemini-bulk-image-gen",
    "OpenAI": "openai-bulk-image-gen",
}
_KEYCHAIN_USERS = {
    "Gemini": "google_api_key",
    "OpenAI": "openai_api_key",
}
_API_KEY_ENV_VARS = {
    "Gemini": "GOOGLE_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
}
_SAVE_API_KEY     = "save_api_key"   # boolean flag stored in config.json


def load_config() -> dict:
    """Load config.json, return empty dict on any error."""
    try:
        if CONFIG_FILE.exists():
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_config(data: dict) -> None:
    """Write config.json (no secrets), silently ignore write errors."""
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def keychain_load_api_key(provider: str = "Gemini") -> str:
    """Return the saved API key for *provider* from the OS keychain, or '' on any error."""
    if not _KEYRING_AVAILABLE:
        return ""
    try:
        return keyring.get_password(
            _KEYCHAIN_SERVICES[provider], _KEYCHAIN_USERS[provider]
        ) or ""
    except Exception:
        return ""


def keychain_save_api_key(api_key: str, provider: str = "Gemini") -> None:
    """Store the API key for *provider* in the OS keychain, silently ignore errors."""
    if not _KEYRING_AVAILABLE:
        return
    try:
        keyring.set_password(
            _KEYCHAIN_SERVICES[provider], _KEYCHAIN_USERS[provider], api_key
        )
    except Exception:
        pass


def keychain_delete_api_key(provider: str = "Gemini") -> None:
    """Remove the API key for *provider* from the OS keychain, silently ignore errors."""
    if not _KEYRING_AVAILABLE:
        return
    try:
        keyring.delete_password(
            _KEYCHAIN_SERVICES[provider], _KEYCHAIN_USERS[provider]
        )
    except keyring.errors.PasswordDeleteError:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Saved prompts  (prompts.json — gitignored, never committed)
# Stored as an ordered list of {"name": str, "text": str} objects so the
# display order is preserved across sessions.
# ---------------------------------------------------------------------------

PROMPTS_FILE = SCRIPT_DIR / "prompts.json"


def load_prompts() -> list[dict]:
    """Return saved prompts as [{"name": ..., "text": ...}, ...], or []."""
    try:
        if PROMPTS_FILE.exists():
            with PROMPTS_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def save_prompts(prompts: list[dict]) -> None:
    """Write prompts list to prompts.json, silently ignore errors."""
    try:
        with PROMPTS_FILE.open("w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Prompt sets  (prompt_sets.json — gitignored, never committed)
# Each set is {"name": str, "prompts": [{"name": str, "text": str}, ...]}
# ---------------------------------------------------------------------------

PROMPT_SETS_FILE = SCRIPT_DIR / "prompt_sets.json"


def load_prompt_sets() -> list[dict]:
    """Return saved prompt sets, or []."""
    try:
        if PROMPT_SETS_FILE.exists():
            with PROMPT_SETS_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def save_prompt_sets(sets: list[dict]) -> None:
    """Write prompt sets to prompt_sets.json, silently ignore errors."""
    try:
        with PROMPT_SETS_FILE.open("w", encoding="utf-8") as f:
            json.dump(sets, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Batch job state  (batch_state.json — persists active batch job across restarts)
# ---------------------------------------------------------------------------

BATCH_STATE_FILE = SCRIPT_DIR / "batch_state.json"


def load_batch_state() -> dict | None:
    """Return the saved batch job state, or None."""
    try:
        if BATCH_STATE_FILE.exists():
            with BATCH_STATE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def save_batch_state(state: dict) -> None:
    """Persist the active batch job state to disk."""
    try:
        with BATCH_STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def clear_batch_state() -> None:
    """Remove the batch state file."""
    try:
        if BATCH_STATE_FILE.exists():
            BATCH_STATE_FILE.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bulk Image Generator (Gemini + OpenAI)")
        self.resizable(True, True)
        self.geometry("880x800")
        self.minsize(700, 580)
        self._stop_event = threading.Event()
        self._batch_job_name: str | None = None  # active batch job for cancellation
        self._config = load_config()
        self._prompts: list[dict] = load_prompts()       # individual saved prompts
        self._prompt_sets: list[dict] = load_prompt_sets()  # prompt sets
        # Suppresses the auto-clear of the API key field when the provider
        # change is being driven by _load_state (so the env var / saved value
        # we just populated isn't wiped).
        self._suppress_api_key_swap = False
        self._build_ui()
        self._load_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # --- API key ---
        self._api_key_label = tk.Label(self, text="API Key:", anchor="w")
        self._api_key_label.grid(row=0, column=0, sticky="w", **pad)
        self.api_key_var = tk.StringVar(
            value=os.environ.get(_API_KEY_ENV_VARS[DEFAULT_PROVIDER], "")
        )
        api_row = tk.Frame(self)
        api_row.grid(row=0, column=1, columnspan=2, sticky="ew", **pad)
        api_entry = tk.Entry(api_row, textvariable=self.api_key_var, show="*", width=44)
        api_entry.pack(side="left")
        self.save_api_key_var = tk.BooleanVar(value=False)
        self._save_key_cb = tk.Checkbutton(api_row, text="Save API key",
                                           variable=self.save_api_key_var)
        self._save_key_cb.pack(side="left", padx=(10, 0))

        # --- Provider + Model ---
        tk.Label(self, text="Provider:", anchor="w").grid(row=1, column=0, sticky="w", **pad)
        self.provider_var = tk.StringVar(value=DEFAULT_PROVIDER)
        prov_frame = tk.Frame(self)
        prov_frame.grid(row=1, column=1, columnspan=2, sticky="ew", **pad)
        self._provider_combo = ttk.Combobox(prov_frame, textvariable=self.provider_var,
                                            values=PROVIDERS, width=10, state="readonly")
        self._provider_combo.pack(side="left")
        self._provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        tk.Label(prov_frame, text="  Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self._model_combo = ttk.Combobox(prov_frame, textvariable=self.model_var,
                                         values=MODELS_BY_PROVIDER[DEFAULT_PROVIDER],
                                         width=34)
        self._model_combo.pack(side="left", padx=(4, 0))
        tk.Label(prov_frame, text="  (or type a custom model ID)", fg="gray",
                 font=("TkDefaultFont", 8)).pack(side="left")

        # --- Input folder ---
        tk.Label(self, text="Input folder:", anchor="w").grid(row=2, column=0, sticky="w", **pad)
        self.input_var = tk.StringVar()
        tk.Entry(self, textvariable=self.input_var, width=45).grid(
            row=2, column=1, sticky="ew", **pad)
        tk.Button(self, text="Browse…", command=self._pick_input).grid(
            row=2, column=2, sticky="w", padx=(0, 10))

        # --- Output folder ---
        tk.Label(self, text="Output folder:", anchor="w").grid(row=3, column=0, sticky="w", **pad)
        self.output_var = tk.StringVar()
        tk.Entry(self, textvariable=self.output_var, width=45).grid(
            row=3, column=1, sticky="ew", **pad)
        tk.Button(self, text="Browse…", command=self._pick_output).grid(
            row=3, column=2, sticky="w", padx=(0, 10))

        # --- Options ---
        options_frame = tk.Frame(self)
        options_frame.grid(row=4, column=1, columnspan=2, sticky="w", **pad)

        self.overwrite_var = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Overwrite existing outputs",
                       variable=self.overwrite_var).pack(side="left", padx=(0, 20))

        # Two dropdowns whose label/values flip based on the active provider.
        # Gemini: Aspect ratio + Resolution (1K/2K/4K/512)
        # OpenAI: Size (1024x1024 etc.) + Quality (auto/low/medium/high)
        self.aspect_ratio_var  = tk.StringVar(value=DEFAULT_ASPECT_RATIO)
        self.image_size_var    = tk.StringVar(value=DEFAULT_IMAGE_SIZE)
        self.openai_size_var    = tk.StringVar(value=DEFAULT_OPENAI_SIZE)
        self.openai_quality_var = tk.StringVar(value=DEFAULT_OPENAI_QUALITY)

        self._opt1_label = tk.Label(options_frame, text="Aspect ratio:")
        self._opt1_label.pack(side="left")
        self._opt1_combo = ttk.Combobox(options_frame, textvariable=self.aspect_ratio_var,
                                        values=ASPECT_RATIOS, width=10, state="readonly")
        self._opt1_combo.pack(side="left", padx=(4, 16))

        self._opt2_label = tk.Label(options_frame, text="Resolution:")
        self._opt2_label.pack(side="left")
        self._opt2_combo = ttk.Combobox(options_frame, textvariable=self.image_size_var,
                                        values=IMAGE_SIZES, width=10, state="readonly")
        self._opt2_combo.pack(side="left", padx=(4, 0))
        self._opt_hint = tk.Label(options_frame,
                                  text="(512: 3.1 flash only · 4K: pro & 3.1 flash)",
                                  fg="gray", font=("TkDefaultFont", 8))
        self._opt_hint.pack(side="left", padx=(6, 0))

        # Second options row for run mode
        options_frame2 = tk.Frame(self)
        options_frame2.grid(row=5, column=1, columnspan=2, sticky="w", **pad)

        tk.Label(options_frame2, text="Run mode:").pack(side="left")
        self.run_mode_var = tk.StringVar(value=RUN_MODES[0])
        self._run_mode_combo = ttk.Combobox(options_frame2, textvariable=self.run_mode_var,
                                            values=RUN_MODES, width=14, state="readonly")
        self._run_mode_combo.pack(side="left", padx=(4, 12))
        self._run_mode_hint = tk.Label(
            options_frame2,
            text="Batch sends all work as one async job · cheaper · higher rate limits",
            fg="gray", font=("TkDefaultFont", 8))
        self._run_mode_hint.pack(side="left")

        # --- Prompt Set selector (NEW) ---
        tk.Label(self, text="Prompt set:", anchor="w").grid(row=6, column=0, sticky="w", **pad)

        set_col = tk.Frame(self)
        set_col.grid(row=6, column=1, columnspan=2, sticky="ew", **pad)

        self._prompt_set_var = tk.StringVar(value=_SINGLE_PROMPT)
        self._prompt_set_combo = ttk.Combobox(set_col, textvariable=self._prompt_set_var,
                                              state="readonly", width=34)
        self._prompt_set_combo.pack(side="left")
        self._prompt_set_combo.bind("<<ComboboxSelected>>", self._on_prompt_set_selected)

        tk.Button(set_col, text="New Set…", command=self._new_prompt_set).pack(
            side="left", padx=(8, 0))
        tk.Button(set_col, text="Edit Set…", command=self._edit_prompt_set).pack(
            side="left", padx=(4, 0))
        tk.Button(set_col, text="Delete Set", command=self._delete_prompt_set).pack(
            side="left", padx=(4, 0))

        # Info label showing set contents
        self._set_info_var = tk.StringVar(value="")
        self._set_info_label = tk.Label(self, textvariable=self._set_info_var,
                                        anchor="w", fg="gray",
                                        font=("TkDefaultFont", 8), wraplength=600,
                                        justify="left")
        self._set_info_label.grid(row=7, column=1, columnspan=2, sticky="w",
                                  padx=10, pady=(0, 2))

        # --- Prompt (single-prompt editor) ---
        self._prompt_label = tk.Label(self, text="Prompt:", anchor="nw")
        self._prompt_label.grid(row=8, column=0, sticky="nw", **pad)

        prompt_col = tk.Frame(self)
        prompt_col.grid(row=8, column=1, columnspan=2, sticky="nsew", **pad)
        prompt_col.columnconfigure(0, weight=1)

        # Toolbar: saved-prompt picker + Save + Delete
        prompt_toolbar = tk.Frame(prompt_col)
        prompt_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        prompt_toolbar.columnconfigure(0, weight=1)

        self._saved_prompt_var = tk.StringVar()
        self._prompt_combo = ttk.Combobox(prompt_toolbar, textvariable=self._saved_prompt_var,
                                          state="readonly", width=38)
        self._prompt_combo.grid(row=0, column=0, sticky="ew")
        self._prompt_combo.bind("<<ComboboxSelected>>", self._on_prompt_selected)

        self._save_prompt_btn = tk.Button(prompt_toolbar, text="💾 Save", width=8,
                                          command=self._save_prompt)
        self._save_prompt_btn.grid(row=0, column=1, padx=(6, 0))
        self._delete_prompt_btn = tk.Button(prompt_toolbar, text="🗑 Delete", width=8,
                                            command=self._delete_prompt)
        self._delete_prompt_btn.grid(row=0, column=2, padx=(4, 0))

        # Text area
        self.prompt_text = scrolledtext.ScrolledText(prompt_col, width=60, height=4, wrap=tk.WORD)
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)
        self.prompt_text.grid(row=1, column=0, sticky="nsew")
        prompt_col.rowconfigure(1, weight=1)

        # --- Progress ---
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.grid(row=9, column=0, columnspan=3, sticky="ew", **pad)

        self.status_label = tk.Label(self, text="", anchor="w", fg="gray")
        self.status_label.grid(row=10, column=0, columnspan=3, sticky="w", **pad)

        # --- Log ---
        self.log = scrolledtext.ScrolledText(self, width=70, height=8, state="disabled",
                                             wrap=tk.WORD, bg="#1e1e1e", fg="#d4d4d4",
                                             font=("TkFixedFont", 9))
        self.log.grid(row=11, column=0, columnspan=3, sticky="nsew", **pad)

        # --- Buttons ---
        # Native ttk themes (vista, aqua) ignore fg/bg on buttons.
        # "clam" is a built-in cross-platform theme that honours colours;
        # we scope it to a named style so all other widgets keep their OS look.
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=12, column=0, columnspan=3, pady=10)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Run.TButton",
                        foreground="white", background="#0078d7",
                        font=("TkDefaultFont", 10, "bold"),
                        padding=6)
        style.map("Run.TButton",
                  background=[("active", "#005fa3"), ("disabled", "#a0a0a0")],
                  foreground=[("disabled", "#d0d0d0")])
        style.configure("Stop.TButton", padding=6)
        style.configure("Resume.TButton",
                        foreground="white", background="#2d8a4e",
                        font=("TkDefaultFont", 10, "bold"),
                        padding=6)
        style.map("Resume.TButton",
                  background=[("active", "#1e6b3a"), ("disabled", "#a0a0a0")],
                  foreground=[("disabled", "#d0d0d0")])
        self.run_btn = ttk.Button(btn_frame, text="Run", width=14,
                                  style="Run.TButton", command=self._start)
        self.run_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(btn_frame, text="Stop", width=14,
                                   style="Stop.TButton",
                                   state="disabled", command=self._request_stop)
        self.stop_btn.pack(side="left", padx=6)
        self.resume_btn = ttk.Button(btn_frame, text="Resume Batch", width=14,
                                     style="Resume.TButton",
                                     state="disabled",
                                     command=self._resume_batch)
        self.resume_btn.pack(side="left", padx=6)

        # Make columns/rows resize gracefully.
        self.columnconfigure(1, weight=1)
        self.rowconfigure(8, weight=1)    # prompt editor
        self.rowconfigure(11, weight=2)   # log

        # Disable "Save API key" if keyring is not installed
        if not _KEYRING_AVAILABLE:
            self._save_key_cb.configure(state="disabled")

    # ------------------------------------------------------------------
    # Folder pickers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Provider switch
    # ------------------------------------------------------------------

    def _on_provider_change(self, _event=None):
        """Reconfigure model list, size/quality controls, run mode, and API key
        for the currently selected provider."""
        provider = self.provider_var.get()

        # Model dropdown
        models = MODELS_BY_PROVIDER.get(provider, [])
        self._model_combo.configure(values=models)
        if self.model_var.get() not in models:
            self.model_var.set(models[0] if models else "")

        # Size / quality controls
        if provider == "OpenAI":
            self._opt1_label.configure(text="Size:")
            self._opt1_combo.configure(textvariable=self.openai_size_var,
                                       values=OPENAI_SIZES)
            self._opt2_label.configure(text="Quality:")
            self._opt2_combo.configure(textvariable=self.openai_quality_var,
                                       values=OPENAI_QUALITIES)
            self._opt_hint.configure(
                text="(size 'auto' lets the model pick · gpt-image-2 priced per size+quality)")
        else:
            self._opt1_label.configure(text="Aspect ratio:")
            self._opt1_combo.configure(textvariable=self.aspect_ratio_var,
                                       values=ASPECT_RATIOS)
            self._opt2_label.configure(text="Resolution:")
            self._opt2_combo.configure(textvariable=self.image_size_var,
                                       values=IMAGE_SIZES)
            self._opt_hint.configure(
                text="(512: 3.1 flash only · 4K: pro & 3.1 flash)")

        # Run mode — Batch is Gemini-only
        if provider == "OpenAI":
            self.run_mode_var.set(RUN_MODES[0])  # Real-time
            self._run_mode_combo.configure(state="disabled")
            self._run_mode_hint.configure(
                text="Batch mode is Gemini-only — OpenAI runs in real-time.")
        else:
            self._run_mode_combo.configure(state="readonly")
            self._run_mode_hint.configure(
                text="Batch sends all work as one async job · cheaper · higher rate limits")

        # API key field — swap to whatever was saved for this provider, falling
        # back to the provider's env var. Skipped during _load_state to avoid
        # clobbering a value the user (or _load_state) just set.
        if not self._suppress_api_key_swap:
            saved = keychain_load_api_key(provider) if self.save_api_key_var.get() else ""
            self.api_key_var.set(
                saved or os.environ.get(_API_KEY_ENV_VARS[provider], "")
            )

    def _pick_input(self):
        folder = filedialog.askdirectory(title="Select input folder")
        if folder:
            self.input_var.set(folder)
            if not self.output_var.get():
                subfolder = "openai" if self.provider_var.get() == "OpenAI" else "gemini"
                self.output_var.set(str(Path(folder) / subfolder))

    def _pick_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_var.set(folder)

    # ------------------------------------------------------------------
    # Prompt set management
    # ------------------------------------------------------------------

    def _refresh_prompt_set_combo(self):
        """Sync the prompt-set combobox with self._prompt_sets."""
        names = [_SINGLE_PROMPT] + [s["name"] for s in self._prompt_sets]
        self._prompt_set_combo["values"] = names
        if self._prompt_set_var.get() not in names:
            self._prompt_set_var.set(_SINGLE_PROMPT)
        self._update_set_info()

    def _update_set_info(self):
        """Update the info label below the prompt-set selector."""
        name = self._prompt_set_var.get()
        if name == _SINGLE_PROMPT:
            self._set_info_var.set("Using the single prompt editor below.")
            return
        for s in self._prompt_sets:
            if s["name"] == name:
                prompts = s.get("prompts", [])
                n = len(prompts)
                if n == 0:
                    self._set_info_var.set("Set is empty — add prompts via Edit Set.")
                else:
                    preview = ", ".join(p["name"] for p in prompts[:8])
                    if n > 8:
                        preview += f", … ({n} total)"
                    self._set_info_var.set(f"{n} prompt(s): {preview}")
                return
        self._set_info_var.set("")

    def _on_prompt_set_selected(self, _event=None):
        self._update_set_info()
        self._set_prompt_editor_enabled(self._prompt_set_var.get() == _SINGLE_PROMPT)

    def _set_prompt_editor_enabled(self, enabled: bool):
        """Enable or disable the single-prompt editor area."""
        state = "normal" if enabled else "disabled"
        fg = "black" if enabled else "gray"
        self._prompt_label.configure(fg=fg)
        self._prompt_combo.configure(state="readonly" if enabled else "disabled")
        self._save_prompt_btn.configure(state=state)
        self._delete_prompt_btn.configure(state=state)
        self.prompt_text.configure(state=state,
                                   bg="#ffffff" if enabled else "#f0f0f0")

    def _new_prompt_set(self):
        """Open the set editor dialog to create a new prompt set."""
        self._open_set_editor(None)

    def _edit_prompt_set(self):
        """Open the set editor dialog for the currently selected set."""
        name = self._prompt_set_var.get()
        if name == _SINGLE_PROMPT:
            self._log("Select a prompt set first, or click 'New Set…' to create one.")
            return
        for s in self._prompt_sets:
            if s["name"] == name:
                self._open_set_editor(s)
                return

    def _delete_prompt_set(self):
        """Delete the currently selected prompt set after confirmation."""
        name = self._prompt_set_var.get()
        if name == _SINGLE_PROMPT:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Delete Prompt Set")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self)

        tk.Label(dlg, text=f'Delete set "{name}"?').pack(padx=20, pady=(16, 4))
        tk.Label(dlg, text="This cannot be undone.", fg="gray",
                 font=("TkDefaultFont", 8)).pack(padx=20, pady=(0, 12))

        def _confirm():
            self._prompt_sets = [s for s in self._prompt_sets if s["name"] != name]
            save_prompt_sets(self._prompt_sets)
            self._prompt_set_var.set(_SINGLE_PROMPT)
            self._refresh_prompt_set_combo()
            dlg.destroy()

        btn_row = tk.Frame(dlg)
        btn_row.pack(pady=(0, 12))
        tk.Button(btn_row, text="Delete", width=10, command=_confirm).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", width=10, command=dlg.destroy).pack(side="left", padx=6)

        dlg.bind("<Return>", lambda _e: _confirm())
        dlg.bind("<Escape>", lambda _e: dlg.destroy())

        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_reqwidth()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_reqheight()) // 2
        dlg.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # Prompt set editor dialog
    # ------------------------------------------------------------------

    def _open_set_editor(self, existing_set: dict | None):
        """Open a dialog to create or edit a prompt set.

        If *existing_set* is None a new empty set is created; otherwise the
        existing set is edited in-place.
        """
        dlg = tk.Toplevel(self)
        dlg.title("Edit Prompt Set" if existing_set else "New Prompt Set")
        dlg.resizable(True, True)
        dlg.geometry("720x480")
        dlg.minsize(520, 350)
        dlg.grab_set()
        dlg.transient(self)

        # Working copy of prompts (list of dicts)
        if existing_set:
            work_prompts: list[dict] = [dict(p) for p in existing_set.get("prompts", [])]
        else:
            work_prompts = []

        # --- Set name ---
        name_frame = tk.Frame(dlg)
        name_frame.pack(fill="x", padx=12, pady=(12, 6))
        tk.Label(name_frame, text="Set name:").pack(side="left")
        set_name_var = tk.StringVar(value=existing_set["name"] if existing_set else "")
        tk.Entry(name_frame, textvariable=set_name_var, width=40).pack(side="left", padx=(6, 0))

        # --- Main pane: list + editor ---
        pane = tk.Frame(dlg)
        pane.pack(fill="both", expand=True, padx=12, pady=4)
        pane.columnconfigure(1, weight=1)
        pane.rowconfigure(0, weight=1)

        # Left: prompt listbox
        list_frame = tk.LabelFrame(pane, text="Prompts")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        list_frame.rowconfigure(0, weight=1)

        listbox = tk.Listbox(list_frame, width=24, exportselection=False)
        listbox.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        list_sb = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        list_sb.grid(row=0, column=1, sticky="ns", pady=4)
        listbox.configure(yscrollcommand=list_sb.set)

        # Right: prompt text editor
        edit_frame = tk.LabelFrame(pane, text="Prompt text")
        edit_frame.grid(row=0, column=1, sticky="nsew")
        edit_frame.rowconfigure(1, weight=1)
        edit_frame.columnconfigure(0, weight=1)

        # Prompt name field
        pname_row = tk.Frame(edit_frame)
        pname_row.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
        tk.Label(pname_row, text="Name:").pack(side="left")
        prompt_name_var = tk.StringVar()
        prompt_name_entry = tk.Entry(pname_row, textvariable=prompt_name_var, width=30)
        prompt_name_entry.pack(side="left", padx=(4, 0))

        # Prompt text area
        prompt_edit = scrolledtext.ScrolledText(edit_frame, width=40, height=8, wrap=tk.WORD)
        prompt_edit.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        # ---- Helper functions ----

        def _refresh_listbox():
            listbox.delete(0, "end")
            for i, p in enumerate(work_prompts, 1):
                listbox.insert("end", f"{i}. {p['name']}")

        def _save_current_to_work():
            """Save whatever is in the editor back to the work_prompts list."""
            sel = listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            work_prompts[idx]["name"] = prompt_name_var.get().strip() or f"Prompt {idx+1}"
            work_prompts[idx]["text"] = prompt_edit.get("1.0", "end").strip()
            # Refresh that row's display text
            listbox.delete(idx)
            listbox.insert(idx, f"{idx+1}. {work_prompts[idx]['name']}")
            listbox.selection_set(idx)

        def _on_listbox_select(_event=None):
            sel = listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            p = work_prompts[idx]
            prompt_name_var.set(p["name"])
            prompt_edit.delete("1.0", "end")
            prompt_edit.insert("1.0", p["text"])

        listbox.bind("<<ListboxSelect>>", _on_listbox_select)

        def _add_prompt():
            _save_current_to_work()
            n = len(work_prompts) + 1
            work_prompts.append({"name": f"Prompt {n}", "text": ""})
            _refresh_listbox()
            listbox.selection_clear(0, "end")
            listbox.selection_set("end")
            listbox.see("end")
            _on_listbox_select()
            prompt_name_entry.focus_set()
            prompt_name_entry.select_range(0, "end")

        def _remove_prompt():
            sel = listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            work_prompts.pop(idx)
            _refresh_listbox()
            if work_prompts:
                new_idx = min(idx, len(work_prompts) - 1)
                listbox.selection_set(new_idx)
                _on_listbox_select()
            else:
                prompt_name_var.set("")
                prompt_edit.delete("1.0", "end")

        def _move_up():
            sel = listbox.curselection()
            if not sel or sel[0] == 0:
                return
            _save_current_to_work()
            idx = sel[0]
            work_prompts[idx - 1], work_prompts[idx] = work_prompts[idx], work_prompts[idx - 1]
            _refresh_listbox()
            listbox.selection_set(idx - 1)
            listbox.see(idx - 1)

        def _move_down():
            sel = listbox.curselection()
            if not sel or sel[0] >= len(work_prompts) - 1:
                return
            _save_current_to_work()
            idx = sel[0]
            work_prompts[idx + 1], work_prompts[idx] = work_prompts[idx], work_prompts[idx + 1]
            _refresh_listbox()
            listbox.selection_set(idx + 1)
            listbox.see(idx + 1)

        # --- Toolbar buttons ---
        tb = tk.Frame(dlg)
        tb.pack(fill="x", padx=12, pady=(4, 2))
        tk.Button(tb, text="+ Add", command=_add_prompt).pack(side="left", padx=(0, 4))
        tk.Button(tb, text="- Remove", command=_remove_prompt).pack(side="left", padx=(0, 4))
        tk.Button(tb, text="↑ Up", command=_move_up).pack(side="left", padx=(0, 4))
        tk.Button(tb, text="↓ Down", command=_move_down).pack(side="left")

        # --- Save / Cancel ---
        def _do_save():
            sname = set_name_var.get().strip()
            if not sname:
                messagebox.showwarning("Name required",
                                      "Please enter a name for this prompt set.",
                                      parent=dlg)
                return
            _save_current_to_work()

            if existing_set:
                existing_set["name"] = sname
                existing_set["prompts"] = work_prompts
            else:
                # Check for duplicate name
                for s in self._prompt_sets:
                    if s["name"] == sname:
                        s["prompts"] = work_prompts
                        break
                else:
                    self._prompt_sets.append({"name": sname, "prompts": work_prompts})

            save_prompt_sets(self._prompt_sets)
            self._refresh_prompt_set_combo()
            self._prompt_set_var.set(sname)
            self._update_set_info()
            dlg.destroy()

        bottom = tk.Frame(dlg)
        bottom.pack(fill="x", padx=12, pady=(6, 12))
        tk.Button(bottom, text="Save Set", width=12, command=_do_save).pack(side="right", padx=(6, 0))
        tk.Button(bottom, text="Cancel", width=12, command=dlg.destroy).pack(side="right")

        # Populate
        _refresh_listbox()
        if work_prompts:
            listbox.selection_set(0)
            _on_listbox_select()

        # Centre the dialog
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 720) // 2
        y = self.winfo_y() + (self.winfo_height() - 480) // 2
        dlg.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # Saved prompts (individual — existing feature)
    # ------------------------------------------------------------------

    def _refresh_prompt_combo(self):
        """Sync the combobox values with self._prompts."""
        names = [p["name"] for p in self._prompts]
        self._prompt_combo["values"] = names
        if self._saved_prompt_var.get() not in names:
            self._saved_prompt_var.set("")

    def _on_prompt_selected(self, _event=None):
        """Load the selected saved prompt into the text area."""
        name = self._saved_prompt_var.get()
        for p in self._prompts:
            if p["name"] == name:
                self.prompt_text.delete("1.0", "end")
                self.prompt_text.insert("1.0", p["text"])
                break

    def _save_prompt(self):
        """Ask for a name and save the current prompt text."""
        current_text = self.prompt_text.get("1.0", "end").strip()
        if not current_text:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Save Prompt")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self)

        tk.Label(dlg, text="Name for this prompt:").pack(padx=16, pady=(14, 4))
        name_var = tk.StringVar()

        if self._saved_prompt_var.get():
            name_var.set(self._saved_prompt_var.get())

        name_entry = tk.Entry(dlg, textvariable=name_var, width=36)
        name_entry.pack(padx=16)
        name_entry.select_range(0, "end")
        name_entry.focus_set()

        def _confirm():
            name = name_var.get().strip()
            if not name:
                return
            for p in self._prompts:
                if p["name"] == name:
                    p["text"] = current_text
                    break
            else:
                self._prompts.append({"name": name, "text": current_text})
            save_prompts(self._prompts)
            self._refresh_prompt_combo()
            self._saved_prompt_var.set(name)
            dlg.destroy()

        btn_row = tk.Frame(dlg)
        btn_row.pack(pady=12)
        tk.Button(btn_row, text="Save", width=10, command=_confirm).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", width=10, command=dlg.destroy).pack(side="left", padx=6)

        dlg.bind("<Return>", lambda _e: _confirm())
        dlg.bind("<Escape>", lambda _e: dlg.destroy())

        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_reqwidth()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_reqheight()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _delete_prompt(self):
        """Delete the currently selected saved prompt after confirmation."""
        name = self._saved_prompt_var.get()
        if not name:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Delete Prompt")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self)

        tk.Label(dlg, text=f'Delete "{name}"?').pack(padx=20, pady=(16, 4))
        tk.Label(dlg, text="This cannot be undone.", fg="gray",
                 font=("TkDefaultFont", 8)).pack(padx=20, pady=(0, 12))

        def _confirm():
            self._prompts = [p for p in self._prompts if p["name"] != name]
            save_prompts(self._prompts)
            self._refresh_prompt_combo()
            dlg.destroy()

        btn_row = tk.Frame(dlg)
        btn_row.pack(pady=(0, 12))
        tk.Button(btn_row, text="Delete", width=10, command=_confirm).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", width=10, command=dlg.destroy).pack(side="left", padx=6)

        dlg.bind("<Return>", lambda _e: _confirm())
        dlg.bind("<Escape>", lambda _e: dlg.destroy())

        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_reqwidth()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_reqheight()) // 2
        dlg.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        """Populate widgets from saved config and OS keychain."""
        c = self._config

        # Restore provider first so we know which keychain slot to read.
        saved_provider = c.get("provider")
        if saved_provider in PROVIDERS:
            self.provider_var.set(saved_provider)
        provider = self.provider_var.get()

        # API key priority: env var > OS keychain (if opted in) > empty.
        # The env var was already populated for DEFAULT_PROVIDER in _build_ui;
        # if the saved provider is different, refresh from its env var.
        if saved_provider and saved_provider != DEFAULT_PROVIDER:
            self.api_key_var.set(os.environ.get(_API_KEY_ENV_VARS[provider], ""))
        if c.get(_SAVE_API_KEY):
            self.save_api_key_var.set(True)
            if not self.api_key_var.get():
                self.api_key_var.set(keychain_load_api_key(provider))

        if c.get("input_dir"):
            self.input_var.set(c["input_dir"])
        if c.get("output_dir"):
            self.output_var.set(c["output_dir"])
        if c.get("model"):
            self.model_var.set(c["model"])
        if c.get("aspect_ratio") in ASPECT_RATIOS:
            self.aspect_ratio_var.set(c["aspect_ratio"])
        if c.get("image_size") in IMAGE_SIZES:
            self.image_size_var.set(c["image_size"])
        if c.get("openai_size") in OPENAI_SIZES:
            self.openai_size_var.set(c["openai_size"])
        if c.get("openai_quality") in OPENAI_QUALITIES:
            self.openai_quality_var.set(c["openai_quality"])
        if "overwrite" in c:
            self.overwrite_var.set(c["overwrite"])
        if c.get("run_mode") in RUN_MODES:
            self.run_mode_var.set(c["run_mode"])

        # Now apply the provider's UI changes (model list, label swaps, etc.).
        # Suppress the API-key auto-swap since we just populated it above.
        self._suppress_api_key_swap = True
        try:
            self._on_provider_change()
        finally:
            self._suppress_api_key_swap = False

        if not _KEYRING_AVAILABLE:
            self.save_api_key_var.set(False)
            self.after(200, self._log,
                       "NOTE: 'keyring' package not found — "
                       "run 'pip install keyring' to enable secure API key saving.")

        # Populate the saved-prompts combobox
        self._refresh_prompt_combo()

        # Populate the prompt-set combobox and restore selection
        self._refresh_prompt_set_combo()
        saved_set = c.get("selected_prompt_set", _SINGLE_PROMPT)
        set_names = [s["name"] for s in self._prompt_sets]
        if saved_set in set_names:
            self._prompt_set_var.set(saved_set)
        else:
            self._prompt_set_var.set(_SINGLE_PROMPT)
        self._update_set_info()
        self._set_prompt_editor_enabled(self._prompt_set_var.get() == _SINGLE_PROMPT)

        # Check for a pending batch job from a previous session
        batch = load_batch_state()
        if batch and batch.get("job_name"):
            self.resume_btn.configure(state="normal")
            self.after(200, self._log,
                       f"Pending batch job found: {batch['job_name']}\n"
                       f"  Output folder: {batch.get('output_dir', '?')}\n"
                       f"  Items: {len(batch.get('key_map', {}))}\n"
                       "Click 'Resume Batch' to reconnect and download results.")

    def _save_state(self):
        """Persist settings to config.json; API key goes to OS keychain only."""
        save_api = self.save_api_key_var.get()
        provider = self.provider_var.get()

        if save_api:
            keychain_save_api_key(self.api_key_var.get(), provider)
        else:
            keychain_delete_api_key(provider)

        save_config({
            "provider":             provider,
            "input_dir":            self.input_var.get(),
            "output_dir":           self.output_var.get(),
            "model":                self.model_var.get(),
            "aspect_ratio":         self.aspect_ratio_var.get(),
            "image_size":           self.image_size_var.get(),
            "openai_size":          self.openai_size_var.get(),
            "openai_quality":       self.openai_quality_var.get(),
            "overwrite":            self.overwrite_var.get(),
            _SAVE_API_KEY:          save_api,
            "selected_prompt_set":  self._prompt_set_var.get(),
            "run_mode":             self.run_mode_var.get(),
        })

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, msg: str, color: str = "gray"):
        self.status_label.configure(text=msg, fg=color)

    @staticmethod
    def _format_eta(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _start(self):
        api_key = self.api_key_var.get().strip()
        input_dir = Path(self.input_var.get().strip())
        output_dir = Path(self.output_var.get().strip())
        model = self.model_var.get().strip()
        provider = self.provider_var.get()
        aspect_ratio = self.aspect_ratio_var.get()
        image_size = self.image_size_var.get()
        openai_size = self.openai_size_var.get()
        openai_quality = self.openai_quality_var.get()

        if not api_key:
            self._log("ERROR: API key is required.")
            return
        if provider == "OpenAI" and not _OPENAI_AVAILABLE:
            self._log("ERROR: OpenAI SDK not installed. Run: pip install openai")
            return
        if not input_dir or not input_dir.is_dir():
            self._log("ERROR: Select a valid input folder.")
            return
        if not output_dir:
            self._log("ERROR: Select an output folder.")
            return

        # Build prompt list: [(suffix, text), ...]
        selected_set = self._prompt_set_var.get()
        if selected_set != _SINGLE_PROMPT:
            # Find the set
            prompt_set = None
            for s in self._prompt_sets:
                if s["name"] == selected_set:
                    prompt_set = s
                    break
            if not prompt_set or not prompt_set.get("prompts"):
                self._log("ERROR: Selected prompt set is empty. Add prompts via Edit Set.")
                return
            prompts = [
                (str(i), p["text"])
                for i, p in enumerate(prompt_set["prompts"], 1)
            ]
            self._log(f'Using prompt set "{selected_set}" ({len(prompts)} prompts)')
        else:
            # Single prompt from text area
            prompt_text = self.prompt_text.get("1.0", "end").strip()
            if not prompt_text:
                self._log("ERROR: Prompt cannot be empty.")
                return
            prompts = [("mockup", prompt_text)]

        # --- Cost estimate & confirmation ---
        run_mode = self.run_mode_var.get()
        is_batch = run_mode.startswith("Batch")

        # Defense-in-depth: the UI already locks Real-time when OpenAI is selected,
        # but make sure no edge case slips through.
        if is_batch and provider == "OpenAI":
            self._log("ERROR: Batch mode is Gemini-only. Switch to Real-time.")
            return

        images = collect_images(input_dir)
        num_images = len(images)
        if num_images == 0:
            self._log("No supported images found in the input folder.")
            return

        num_prompts = len(prompts)
        total_generations = num_images * num_prompts

        # Look up per-image cost; fall back for unknown model/size combos.
        if provider == "OpenAI":
            cost_key = f"{openai_size}/{openai_quality}"
            size_label = f"{openai_size} · {openai_quality}"
        else:
            cost_key = image_size
            size_label = image_size

        model_costs = COST_PER_IMAGE.get(model)
        unit_cost = model_costs.get(cost_key) if model_costs else None
        if unit_cost is None and model_costs:
            unit_cost = max(model_costs.values())
        if unit_cost is not None:
            if is_batch:
                unit_cost *= 0.5  # Gemini batch discount
            estimated_total = unit_cost * total_generations
            cost_line = (
                f"Estimated cost: {total_generations} image(s) "
                f"× ${unit_cost:.3f} = ${estimated_total:.2f}"
            )
            if is_batch:
                cost_line += "  (50% batch discount)"
        else:
            cost_line = (
                f"Total generations: {total_generations} "
                f"(cost unknown — custom model or pricing not on file)"
            )

        mode_label = "Batch" if is_batch else "Real-time"
        confirm = messagebox.askyesno(
            "Cost estimate",
            f"Provider: {provider}\n"
            f"Mode: {mode_label}\n"
            f"Model: {model}\n"
            f"Output: {size_label}\n"
            f"Images: {num_images}  ×  Prompts: {num_prompts}\n\n"
            f"{cost_line}\n\n"
            f"Continue?",
        )
        if not confirm:
            self._log("Run cancelled by user.")
            return

        self._save_state()
        self._stop_event.clear()
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_var.set(0)

        if is_batch:
            thread = threading.Thread(
                target=self._run_batch_worker,
                args=(api_key, model, prompts, input_dir, output_dir,
                      self.overwrite_var.get(), aspect_ratio, image_size),
                daemon=True,
            )
        else:
            thread = threading.Thread(
                target=self._run_worker,
                args=(api_key, provider, model, prompts, input_dir, output_dir,
                      self.overwrite_var.get(), aspect_ratio, image_size,
                      openai_size, openai_quality),
                daemon=True,
            )
        thread.start()

    def _resume_batch(self):
        """Resume a previously submitted batch job from saved state."""
        state = load_batch_state()
        if not state or not state.get("job_name"):
            self._log("No pending batch job found.")
            return

        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("API Key Required",
                                   "Please enter your API key before resuming.")
            return

        job_name = state["job_name"]
        key_map = state.get("key_map", {})
        output_dir = Path(state.get("output_dir", "."))
        skipped = state.get("skipped", 0)

        self._log(f"\nResuming batch job: {job_name}")
        self._log(f"  Output folder: {output_dir}")
        self._log(f"  Items: {len(key_map)}")

        self._stop_event.clear()
        self.run_btn.configure(state="disabled")
        self.resume_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_var.set(30)  # start at 30% (upload+submit already done)

        thread = threading.Thread(
            target=self._resume_batch_worker,
            args=(api_key, job_name, key_map, output_dir, skipped),
            daemon=True,
        )
        thread.start()

    def _resume_batch_worker(self, api_key, job_name, key_map, output_dir, skipped):
        """Background thread: poll a previously submitted batch job and download results."""
        client = genai.Client(api_key=api_key)
        self._batch_job_name = job_name

        # --- Poll for completion ---
        self.after(0, self._set_status, f"Reconnected — polling {job_name}…", "blue")
        self.after(0, self._log, "Polling for completion…")
        poll_failures = 0
        state = None

        while True:
            if self._stop_event.is_set():
                try:
                    client.batches.cancel(name=self._batch_job_name)
                    self.after(0, self._log, "Batch cancellation requested.")
                except Exception:
                    pass
                self.after(0, self._finish, 0, skipped, len(key_map), output_dir)
                return

            try:
                batch_job = client.batches.get(name=self._batch_job_name)
                state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
                poll_failures = 0
            except Exception as exc:
                poll_failures += 1
                self.after(0, self._log,
                           f"  Poll error ({poll_failures}): "
                           f"{sanitize_error(str(exc))}")
                if poll_failures >= 10:
                    self.after(0, self._log,
                               "Too many consecutive poll failures — aborting.\n"
                               "The batch job may still be running. "
                               "Try 'Resume Batch' again later.")
                    self.after(0, self._finish, 0, skipped, len(key_map), output_dir)
                    return
                for _ in range(BATCH_POLL_INTERVAL):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                continue

            self.after(0, self._set_status,
                       f"Batch status: {state}", "blue")
            self.after(0, self._log, f"  Status: {state}")

            if state in BATCH_TERMINAL_STATES:
                break

            for _ in range(BATCH_POLL_INTERVAL):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        # --- Handle terminal state ---
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            reason = state.replace("JOB_STATE_", "").lower()
            self.after(0, self._log, f"\nBatch job {reason}.")
            if state == "JOB_STATE_FAILED":
                try:
                    err = getattr(batch_job, 'error', None)
                    if err:
                        self.after(0, self._log, f"  Error: {err}")
                except Exception:
                    pass
            self.after(0, self._finish, 0, skipped, len(key_map), output_dir)
            return

        # --- Download and process results ---
        self.after(0, self._set_status, "Downloading results…", "blue")
        self.after(0, self._log, "\nDownloading results…")
        try:
            result_file = batch_job.dest.file_name
            result_bytes = client.files.download(file=result_file)
            if not isinstance(result_bytes, bytes):
                result_bytes = result_bytes.read()
        except Exception as exc:
            self.after(0, self._log,
                       f"ERROR downloading results: {sanitize_error(str(exc))}")
            self.after(0, self._finish, 0, skipped, len(key_map), output_dir)
            return

        # key_map values are strings (from JSON); convert to Path
        key_to_output = {k: Path(v) for k, v in key_map.items()}
        succeeded = failed = 0

        lines = result_bytes.decode("utf-8").strip().split("\n")
        total_results = len(lines)
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                self.after(0, self._log, f"  Malformed result line: {exc}")
                failed += 1
                continue

            key = entry.get("key", "unknown")
            out_path = key_to_output.get(key)

            if "error" in entry:
                self.after(0, self._log,
                           f"  FAILED [{key}]: {entry['error']}")
                failed += 1
                continue

            try:
                parts = entry["response"]["candidates"][0]["content"]["parts"]
                saved = False
                for part in parts:
                    inline = part.get("inlineData")
                    if inline and inline.get("mimeType", "").startswith("image/"):
                        img_data = base64.b64decode(inline["data"])
                        img = Image.open(io.BytesIO(img_data))
                        if out_path:
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            img.save(out_path)
                            self.after(0, self._log,
                                       f"  Saved: {out_path.name}")
                        succeeded += 1
                        saved = True
                        break
                if not saved:
                    self.after(0, self._log,
                               f"  WARNING [{key}]: No image in response")
                    failed += 1
            except Exception as exc:
                self.after(0, self._log,
                           f"  ERROR [{key}]: {sanitize_error(str(exc))}")
                failed += 1

            self.after(0, self.progress_var.set,
                       30 + (i / total_results * 70))

        self.after(0, self._finish, succeeded, skipped, failed, output_dir)

    def _request_stop(self):
        self._stop_event.set()
        if self._batch_job_name:
            self._set_status("Cancelling batch job…", "orange")
        else:
            self._set_status("Stopping after current image…", "orange")

    def _finish(self, succeeded: int, skipped: int, failed: int, output_dir: Path):
        self._batch_job_name = None
        clear_batch_state()
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.resume_btn.configure(state="disabled")
        self._log(
            f"\nDone.  Succeeded: {succeeded}  |  Skipped: {skipped}  |  Failed: {failed}"
        )
        self._log(f"Output folder: {output_dir}")
        self._set_status(
            f"Done — {succeeded} succeeded, {skipped} skipped, {failed} failed",
            "green" if failed == 0 else "red",
        )

    # ------------------------------------------------------------------
    # Worker (runs in a background thread)
    # ------------------------------------------------------------------

    def _run_worker(self, api_key, provider, model, prompts, input_dir, output_dir,
                    overwrite, aspect_ratio, image_size,
                    openai_size=DEFAULT_OPENAI_SIZE,
                    openai_quality=DEFAULT_OPENAI_QUALITY):
        """Process images through one or more prompts.

        *prompts* is a list of (suffix, prompt_text) tuples.
        Output files are named {stem}_{suffix}.png.
        Provider-aware: dispatches to Gemini or OpenAI based on *provider*.
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.after(0, self._log, f"ERROR creating output folder: {exc}")
            self.after(0, self._finish, 0, 0, 1, output_dir)
            return

        images = collect_images(input_dir)
        total_images = len(images)
        total_prompts = len(prompts)
        total_work = total_images * total_prompts

        self.after(0, self._log, f"Found {total_images} image(s) in {input_dir}")
        if provider == "OpenAI":
            self.after(0, self._log,
                       f"Provider: OpenAI  |  Model: {model}  "
                       f"|  Size: {openai_size}  |  Quality: {openai_quality}")
        else:
            self.after(0, self._log,
                       f"Provider: Gemini  |  Model: {model}  "
                       f"|  Aspect ratio: {aspect_ratio}  |  Resolution: {image_size}")
        if total_prompts > 1:
            self.after(0, self._log, f"Running {total_prompts} prompts per image "
                                     f"({total_work} total generations)\n")
        else:
            self.after(0, self._log, "")

        if total_images == 0:
            self.after(0, self._finish, 0, 0, 0, output_dir)
            return

        if provider == "OpenAI":
            client = OpenAI(api_key=api_key)
        else:
            client = genai.Client(api_key=api_key)
        succeeded = skipped = failed = 0
        completed = 0
        api_call_times: list[float] = []  # durations of actual API calls
        eta_str = ""  # current ETA display string, updated after each item
        consecutive_503s = 0  # circuit breaker: consecutive items that failed with 503

        for img_idx, image_path in enumerate(images, start=1):
            if self._stop_event.is_set():
                break

            for p_idx, (suffix, prompt_text) in enumerate(prompts):
                if self._stop_event.is_set():
                    break

                output_path = output_dir / f"{image_path.stem}_{suffix}.png"

                if output_path.exists() and not overwrite:
                    self.after(0, self._log,
                               f"[{completed+1:>3}/{total_work}] SKIP  "
                               f"{image_path.name} → {output_path.name}  (already done)")
                    skipped += 1
                    completed += 1
                    self.after(0, self.progress_var.set, completed / total_work * 100)
                    continue

                label = image_path.name
                if total_prompts > 1:
                    label += f"  [prompt {suffix}]"
                self.after(0, self._log,
                           f"[{completed+1:>3}/{total_work}] Processing  {label} …")
                status = f"Processing {completed+1}/{total_work}: {label}"
                if eta_str:
                    status += f"  —  ~{eta_str} remaining"
                self.after(0, self._set_status, status, "blue")

                attempt = 0
                item_503 = False  # did this item exhaust retries on 503?
                while attempt < MAX_RETRIES:
                    try:
                        t0 = time.monotonic()
                        if provider == "OpenAI":
                            ok = process_image_openai(
                                client, model, prompt_text, image_path,
                                output_path, openai_size, openai_quality)
                        else:
                            ok = process_image(
                                client, model, prompt_text, image_path,
                                output_path, aspect_ratio, image_size)
                        api_call_times.append(time.monotonic() - t0)
                        if ok:
                            self.after(0, self._log,
                                       f"          saved → {output_path.name}")
                            succeeded += 1
                        else:
                            self.after(0, self._log,
                                       "          WARNING: API returned no image part")
                            failed += 1
                        consecutive_503s = 0  # reset on any successful API call
                        break

                    except _ServiceUnavailable as su:
                        attempt += 1
                        if attempt >= MAX_RETRIES:
                            self.after(0, self._log,
                                       f"          ERROR: Model unavailable (503) "
                                       f"after {MAX_RETRIES} attempts — skipping.")
                            self.after(0, self._log,
                                       f"          Last error: {su.detail}")
                            failed += 1
                            item_503 = True
                            break
                        base = min(30 * (2 ** (attempt - 1)), 300)
                        wait = int(base + random.uniform(0, base * 0.5))
                        self.after(0, self._log,
                                   f"          503: {su.detail}")
                        self.after(0, self._log,
                                   f"          Waiting {wait}s "
                                   f"(retry {attempt}/{MAX_RETRIES-1})…")
                        for _ in range(wait):
                            if self._stop_event.is_set():
                                break
                            time.sleep(1)

                    except _RateLimitRetry as rl:
                        attempt += 1
                        self.after(0, self._log,
                                   f"          Rate limited. Waiting {rl.wait}s "
                                   f"(retry {rl.attempt}/{MAX_RETRIES-1})…")
                        for _ in range(rl.wait):
                            if self._stop_event.is_set():
                                break
                            time.sleep(1)

                    except Exception as exc:
                        self.after(0, self._log,
                                   f"          ERROR: {sanitize_error(str(exc))}")
                        failed += 1
                        break

                # Circuit breaker: stop if too many consecutive 503 failures
                if item_503:
                    consecutive_503s += 1
                    if consecutive_503s >= CIRCUIT_BREAKER_THRESHOLD:
                        self.after(0, self._log,
                                   f"\n*** {CIRCUIT_BREAKER_THRESHOLD} consecutive "
                                   f"items failed with 503 — service appears "
                                   f"degraded. Stopping run. ***")
                        self.after(0, self._log,
                                   "Try again later when the model is less busy.")
                        self.after(0, self._finish, succeeded, skipped, failed,
                                   output_dir)
                        return

                completed += 1
                self.after(0, self.progress_var.set, completed / total_work * 100)

                # Update ETA estimate (shown in status bar on next iteration)
                remaining = total_work - completed
                if api_call_times and remaining > 0:
                    avg_time = sum(api_call_times) / len(api_call_times)
                    eta_secs = remaining * (avg_time + REQUEST_DELAY)
                    eta_str = self._format_eta(eta_secs)
                elif remaining == 0:
                    eta_str = ""

                # Delay between API calls (not after the very last one)
                if not self._stop_event.is_set() and completed < total_work:
                    time.sleep(REQUEST_DELAY)

        self.after(0, self._finish, succeeded, skipped, failed, output_dir)

    # ------------------------------------------------------------------
    # Batch worker (runs in a background thread)
    # ------------------------------------------------------------------

    def _run_batch_worker(self, api_key, model, prompts, input_dir, output_dir,
                          overwrite, aspect_ratio, image_size):
        """Submit all work as a single Gemini Batch API job."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.after(0, self._log, f"ERROR creating output folder: {exc}")
            self.after(0, self._finish, 0, 0, 1, output_dir)
            return

        images = collect_images(input_dir)
        total_images = len(images)
        total_prompts = len(prompts)

        self.after(0, self._log, f"Found {total_images} image(s) in {input_dir}")
        self.after(0, self._log,
                   f"Model: {model}  |  Aspect ratio: {aspect_ratio}  "
                   f"|  Resolution: {image_size}  |  Mode: BATCH")

        if total_images == 0:
            self.after(0, self._finish, 0, 0, 0, output_dir)
            return

        # Build work items, applying skip logic
        work_items = []  # (key, image_path, prompt_text, output_path)
        skipped = 0
        for image_path in images:
            for suffix, prompt_text in prompts:
                output_path = output_dir / f"{image_path.stem}_{suffix}.png"
                if output_path.exists() and not overwrite:
                    skipped += 1
                    continue
                key = f"{image_path.stem}___{suffix}"
                work_items.append((key, image_path, prompt_text, output_path))

        if skipped:
            self.after(0, self._log, f"Skipped {skipped} already-done item(s).")
        if not work_items:
            self.after(0, self._log, "All items already done — nothing to submit.")
            self.after(0, self._finish, 0, skipped, 0, output_dir)
            return

        self.after(0, self._log,
                   f"Submitting {len(work_items)} generation(s) as a batch job…\n")

        client = genai.Client(api_key=api_key)

        # --- Phase 1: Upload input images to File API ---
        unique_images = list({item[1]: None for item in work_items}.keys())
        uploaded_files = {}  # image_path -> uploaded file object
        self.after(0, self._set_status,
                   f"Uploading images to File API… (0/{len(unique_images)})", "blue")
        for i, img_path in enumerate(unique_images, 1):
            if self._stop_event.is_set():
                self.after(0, self._log, "Cancelled during upload.")
                self.after(0, self._finish, 0, skipped, 0, output_dir)
                return
            try:
                mime = MIME_MAP.get(img_path.suffix.lower(), "image/jpeg")
                f = client.files.upload(
                    file=str(img_path),
                    config=types.UploadFileConfig(
                        display_name=img_path.name,
                        mime_type=mime,
                    ),
                )
                uploaded_files[img_path] = f
                self.after(0, self._set_status,
                           f"Uploading images to File API… ({i}/{len(unique_images)})",
                           "blue")
                self.after(0, self.progress_var.set,
                           i / len(unique_images) * 25)  # 0-25% for uploads
            except Exception as exc:
                self.after(0, self._log,
                           f"ERROR uploading {img_path.name}: "
                           f"{sanitize_error(str(exc))}")
                self.after(0, self._finish, 0, skipped, len(work_items), output_dir)
                return

        self.after(0, self._log,
                   f"Uploaded {len(uploaded_files)} image(s) to File API.")

        # --- Phase 2: Build and upload JSONL ---
        self.after(0, self._set_status, "Building and uploading JSONL…", "blue")
        try:
            jsonl_lines = []
            for key, image_path, prompt_text, _ in work_items:
                file_obj = uploaded_files[image_path]
                mime = MIME_MAP.get(image_path.suffix.lower(), "image/jpeg")
                line = {
                    "key": key,
                    "request": {
                        "contents": [{
                            "parts": [
                                {"fileData": {
                                    "fileUri": file_obj.uri,
                                    "mimeType": mime,
                                }},
                                {"text": prompt_text},
                            ]
                        }],
                        "generationConfig": {
                            "responseModalities": ["TEXT", "IMAGE"],
                            "imageConfig": {
                                "aspectRatio": aspect_ratio,
                                "imageSize": image_size,
                            },
                        },
                    },
                }
                jsonl_lines.append(json.dumps(line))

            jsonl_content = "\n".join(jsonl_lines) + "\n"

            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
            tmp.write(jsonl_content)
            tmp.close()

            uploaded_jsonl = client.files.upload(
                file=tmp.name,
                config=types.UploadFileConfig(
                    display_name="batch-requests.jsonl",
                    mime_type="jsonl",
                ),
            )
            os.unlink(tmp.name)
            self.after(0, self._log,
                       f"Uploaded JSONL ({len(jsonl_lines)} requests).")

        except Exception as exc:
            self.after(0, self._log,
                       f"ERROR building/uploading JSONL: "
                       f"{sanitize_error(str(exc))}")
            self.after(0, self._finish, 0, skipped, len(work_items), output_dir)
            return

        # --- Phase 3: Create batch job ---
        self.after(0, self._set_status, "Creating batch job…", "blue")
        try:
            batch_job = client.batches.create(
                model=model,
                src=uploaded_jsonl.name,
                config={"display_name": f"bulk-image-gen-{int(time.time())}"},
            )
            self._batch_job_name = batch_job.name
            # Persist job state so it survives app restarts
            key_map = {key: str(out_path) for key, _, _, out_path in work_items}
            save_batch_state({
                "job_name": batch_job.name,
                "model": model,
                "output_dir": str(output_dir),
                "key_map": key_map,
                "skipped": skipped,
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            self.after(0, self._log,
                       f"Batch job created: {batch_job.name}")
            self.after(0, self._log,
                       "Job state saved — you can safely close the app and "
                       "resume later via 'Resume Batch'.")
        except Exception as exc:
            self.after(0, self._log,
                       f"ERROR creating batch job: {sanitize_error(str(exc))}")
            self.after(0, self._finish, 0, skipped, len(work_items), output_dir)
            return

        self.after(0, self.progress_var.set, 30)

        # --- Phase 4: Poll for completion ---
        self.after(0, self._log, "Polling for completion…")
        poll_failures = 0
        while True:
            if self._stop_event.is_set():
                try:
                    client.batches.cancel(name=self._batch_job_name)
                    self.after(0, self._log, "Batch cancellation requested.")
                except Exception:
                    pass
                self.after(0, self._finish, 0, skipped, len(work_items), output_dir)
                return

            try:
                batch_job = client.batches.get(name=self._batch_job_name)
                state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
                poll_failures = 0
            except Exception as exc:
                poll_failures += 1
                self.after(0, self._log,
                           f"  Poll error ({poll_failures}): "
                           f"{sanitize_error(str(exc))}")
                if poll_failures >= 10:
                    self.after(0, self._log,
                               "Too many consecutive poll failures — aborting.")
                    self.after(0, self._finish, 0, skipped, len(work_items),
                               output_dir)
                    return
                for _ in range(BATCH_POLL_INTERVAL):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                continue

            self.after(0, self._set_status,
                       f"Batch status: {state}", "blue")
            self.after(0, self._log, f"  Status: {state}")

            if state in BATCH_TERMINAL_STATES:
                break

            # Sleep in small increments so Stop is responsive
            for _ in range(BATCH_POLL_INTERVAL):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        # --- Phase 5: Handle terminal state ---
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            reason = state.replace("JOB_STATE_", "").lower()
            self.after(0, self._log, f"\nBatch job {reason}.")
            if state == "JOB_STATE_FAILED":
                # Try to get error details
                try:
                    err = getattr(batch_job, 'error', None)
                    if err:
                        self.after(0, self._log, f"  Error: {err}")
                except Exception:
                    pass
            self.after(0, self._finish, 0, skipped, len(work_items), output_dir)
            return

        # --- Phase 6: Download and process results ---
        self.after(0, self._set_status, "Downloading results…", "blue")
        self.after(0, self._log, "\nDownloading results…")
        try:
            result_file = batch_job.dest.file_name
            result_bytes = client.files.download(file=result_file)
            if not isinstance(result_bytes, bytes):
                result_bytes = result_bytes.read()
        except Exception as exc:
            self.after(0, self._log,
                       f"ERROR downloading results: {sanitize_error(str(exc))}")
            self.after(0, self._finish, 0, skipped, len(work_items), output_dir)
            return

        # Build key -> output_path mapping
        key_to_output = {key: out_path for key, _, _, out_path in work_items}
        succeeded = failed = 0

        lines = result_bytes.decode("utf-8").strip().split("\n")
        total_results = len(lines)
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                self.after(0, self._log, f"  Malformed result line: {exc}")
                failed += 1
                continue

            key = entry.get("key", "unknown")
            out_path = key_to_output.get(key)

            if "error" in entry:
                self.after(0, self._log,
                           f"  FAILED [{key}]: {entry['error']}")
                failed += 1
                continue

            try:
                parts = entry["response"]["candidates"][0]["content"]["parts"]
                saved = False
                for part in parts:
                    inline = part.get("inlineData")
                    if inline and inline.get("mimeType", "").startswith("image/"):
                        img_data = base64.b64decode(inline["data"])
                        img = Image.open(io.BytesIO(img_data))
                        if out_path:
                            img.save(out_path)
                            self.after(0, self._log,
                                       f"  Saved: {out_path.name}")
                        succeeded += 1
                        saved = True
                        break
                if not saved:
                    self.after(0, self._log,
                               f"  WARNING [{key}]: No image in response")
                    failed += 1
            except Exception as exc:
                self.after(0, self._log,
                           f"  ERROR [{key}]: {sanitize_error(str(exc))}")
                failed += 1

            self.after(0, self.progress_var.set,
                       30 + (i / total_results * 70))  # 30-100%

        # --- Cleanup: delete uploaded files ---
        try:
            for file_obj in uploaded_files.values():
                client.files.delete(name=file_obj.name)
            client.files.delete(name=uploaded_jsonl.name)
        except Exception:
            pass  # best-effort cleanup

        self.after(0, self._finish, succeeded, skipped, failed, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = App()
    app.protocol("WM_DELETE_WINDOW", lambda: (app._save_state(), app.destroy()))
    app.mainloop()


if __name__ == "__main__":
    main()
