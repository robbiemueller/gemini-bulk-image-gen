"""
generate_mockups.py

Opens a GUI to select an input folder, output folder, and prompt, then sends
each image to the Gemini API and saves the resulting mockup PNGs.

Requirements:
    Billing must be enabled on your Google AI Studio / Google Cloud account.
    Image-generation models have no free-tier quota (limit = 0).

Setup:
    1. Enable billing: https://aistudio.google.com/plan_information
    2. Set your API key:
           Windows (PowerShell): $env:GOOGLE_API_KEY = "your_key_here"
           macOS/Linux:          export GOOGLE_API_KEY="your_key_here"
    3. Run:
           python generate_mockups.py
"""

import io
import os
import re
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from pathlib import Path
import platform

from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}

MODELS = [
    "gemini-3-pro-image-preview",    # highest quality, 4K support
    "gemini-2.5-flash-image",        # faster, cheaper
]
DEFAULT_MODEL = MODELS[0]

ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "2:3", "3:2", "21:9"]
DEFAULT_ASPECT_RATIO = "1:1"

IMAGE_SIZES = ["1K", "2K", "4K"]
DEFAULT_IMAGE_SIZE = "1K"

DEFAULT_PROMPT = (
    "take the print from this image and create a square aspect ratio hero mockup photo "
    "of 3 of these prints inside of different landscape orientation colored oak frames. "
    "one should be a walnut oak frame, the other should be a natural wood oak frame, and "
    "the 3rd should be a bright white oak frame. the frames should be overlapping each "
    "other sitting on a wood floor leaning against a neutral colored wall"
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

# Retry settings for rate-limit (429) errors
MAX_RETRIES = 5
RETRY_BASE_DELAY = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(folder: Path) -> list[Path]:
    """Return all supported image files in folder (non-recursive)."""
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


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
            if "429" in msg and attempt < MAX_RETRIES:
                wait = extract_retry_delay(msg)
                raise _RateLimitRetry(wait, attempt)
            else:
                raise


class _RateLimitRetry(Exception):
    def __init__(self, wait: int, attempt: int):
        self.wait = wait
        self.attempt = attempt


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gemini Bulk Image Generator")
        self.resizable(True, True)
        self._stop_event = threading.Event()
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # --- API key ---
        tk.Label(self, text="API Key:", anchor="w").grid(row=0, column=0, sticky="w", **pad)
        self.api_key_var = tk.StringVar(value=os.environ.get("GOOGLE_API_KEY", ""))
        api_entry = tk.Entry(self, textvariable=self.api_key_var, show="*", width=50)
        api_entry.grid(row=0, column=1, columnspan=2, sticky="ew", **pad)

        # --- Model ---
        tk.Label(self, text="Model:", anchor="w").grid(row=1, column=0, sticky="w", **pad)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        model_frame = tk.Frame(self)
        model_frame.grid(row=1, column=1, columnspan=2, sticky="ew", **pad)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                   values=MODELS, width=38)
        model_combo.pack(side="left")
        tk.Label(model_frame, text="  (or type a custom model ID)", fg="gray",
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

        tk.Label(options_frame, text="Aspect ratio:").pack(side="left")
        self.aspect_ratio_var = tk.StringVar(value=DEFAULT_ASPECT_RATIO)
        ttk.Combobox(options_frame, textvariable=self.aspect_ratio_var,
                     values=ASPECT_RATIOS, width=6, state="readonly").pack(side="left", padx=(4, 16))

        tk.Label(options_frame, text="Resolution:").pack(side="left")
        self.image_size_var = tk.StringVar(value=DEFAULT_IMAGE_SIZE)
        ttk.Combobox(options_frame, textvariable=self.image_size_var,
                     values=IMAGE_SIZES, width=4, state="readonly").pack(side="left", padx=(4, 0))
        tk.Label(options_frame, text="(4K: pro model only)", fg="gray",
                 font=("TkDefaultFont", 8)).pack(side="left", padx=(6, 0))

        # --- Prompt ---
        tk.Label(self, text="Prompt:", anchor="nw").grid(row=5, column=0, sticky="nw", **pad)
        self.prompt_text = scrolledtext.ScrolledText(self, width=60, height=6, wrap=tk.WORD)
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)
        self.prompt_text.grid(row=5, column=1, columnspan=2, sticky="ew", **pad)

        # --- Progress ---
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky="ew", **pad)

        self.status_label = tk.Label(self, text="", anchor="w", fg="gray")
        self.status_label.grid(row=7, column=0, columnspan=3, sticky="w", **pad)

        # --- Log ---
        self.log = scrolledtext.ScrolledText(self, width=70, height=12, state="disabled",
                                             wrap=tk.WORD, bg="#1e1e1e", fg="#d4d4d4",
                                             font=("TkFixedFont", 9))
        self.log.grid(row=8, column=0, columnspan=3, sticky="nsew", **pad)

        # --- Buttons ---
        # ttk.Button respects native theming on all platforms (tk.Button ignores bg/fg on macOS)
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=10)
        style = ttk.Style()
        style.configure("Run.TButton", foreground="white", background="#0078d7")
        self.run_btn = ttk.Button(btn_frame, text="Run", width=14,
                                  style="Run.TButton", command=self._start)
        self.run_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(btn_frame, text="Stop", width=14,
                                   state="disabled", command=self._request_stop)
        self.stop_btn.pack(side="left", padx=6)

        # Make columns/rows resize gracefully
        self.columnconfigure(1, weight=1)
        self.rowconfigure(8, weight=1)

    # ------------------------------------------------------------------
    # Folder pickers
    # ------------------------------------------------------------------

    def _pick_input(self):
        folder = filedialog.askdirectory(title="Select input folder")
        if folder:
            self.input_var.set(folder)
            if not self.output_var.get():
                self.output_var.set(str(Path(folder) / "gemini"))

    def _pick_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_var.set(folder)

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

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _start(self):
        api_key = self.api_key_var.get().strip()
        input_dir = Path(self.input_var.get().strip())
        output_dir = Path(self.output_var.get().strip())
        prompt = self.prompt_text.get("1.0", "end").strip()
        model = self.model_var.get().strip()
        aspect_ratio = self.aspect_ratio_var.get()
        image_size = self.image_size_var.get()

        if not api_key:
            self._log("ERROR: API key is required.")
            return
        if not input_dir or not input_dir.is_dir():
            self._log("ERROR: Select a valid input folder.")
            return
        if not output_dir:
            self._log("ERROR: Select an output folder.")
            return
        if not prompt:
            self._log("ERROR: Prompt cannot be empty.")
            return

        self._stop_event.clear()
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_var.set(0)

        thread = threading.Thread(
            target=self._run_worker,
            args=(api_key, model, prompt, input_dir, output_dir,
                  self.overwrite_var.get(), aspect_ratio, image_size),
            daemon=True,
        )
        thread.start()

    def _request_stop(self):
        self._stop_event.set()
        self._set_status("Stopping after current image…", "orange")

    def _finish(self, succeeded: int, skipped: int, failed: int, output_dir: Path):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
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

    def _run_worker(self, api_key, model, prompt, input_dir, output_dir, overwrite,
                    aspect_ratio, image_size):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.after(0, self._log, f"ERROR creating output folder: {exc}")
            self.after(0, self._finish, 0, 0, 1, output_dir)
            return

        images = collect_images(input_dir)
        total = len(images)
        self.after(0, self._log, f"Found {total} image(s) in {input_dir}")
        self.after(0, self._log,
                   f"Model: {model}  |  Aspect ratio: {aspect_ratio}  |  Resolution: {image_size}\n")

        if total == 0:
            self.after(0, self._finish, 0, 0, 0, output_dir)
            return

        client = genai.Client(api_key=api_key)
        succeeded = skipped = failed = 0

        for idx, image_path in enumerate(images, start=1):
            if self._stop_event.is_set():
                break

            output_path = output_dir / f"{image_path.stem}_mockup.png"

            if output_path.exists() and not overwrite:
                self.after(0, self._log,
                           f"[{idx:>3}/{total}] SKIP  {image_path.name}  (already done)")
                skipped += 1
                self.after(0, self.progress_var.set, idx / total * 100)
                continue

            self.after(0, self._log,
                       f"[{idx:>3}/{total}] Processing  {image_path.name} …")
            self.after(0, self._set_status,
                       f"Processing {idx}/{total}: {image_path.name}", "blue")

            attempt = 0
            while attempt < MAX_RETRIES:
                try:
                    ok = process_image(client, model, prompt, image_path, output_path,
                                      aspect_ratio, image_size)
                    if ok:
                        self.after(0, self._log,
                                   f"          saved → {output_path.name}")
                        succeeded += 1
                    else:
                        self.after(0, self._log,
                                   "          WARNING: API returned no image part")
                        failed += 1
                    break

                except _RateLimitRetry as rl:
                    attempt += 1
                    self.after(0, self._log,
                               f"          Rate limited. Waiting {rl.wait}s "
                               f"(retry {rl.attempt}/{MAX_RETRIES-1})…")
                    # Sleep in small increments so Stop works promptly
                    for _ in range(rl.wait):
                        if self._stop_event.is_set():
                            break
                        time.sleep(1)

                except Exception as exc:
                    self.after(0, self._log, f"          ERROR: {exc}")
                    failed += 1
                    break

            self.after(0, self.progress_var.set, idx / total * 100)

            if not self._stop_event.is_set() and idx < total:
                time.sleep(REQUEST_DELAY)

        self.after(0, self._finish, succeeded, skipped, failed, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
