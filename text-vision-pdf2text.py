import os
import sys
import shutil
import signal
import tempfile
import atexit

# ── Temp-file handling ─────────────────────────────────────────────────────────
# Everything this app writes (Gradio's upload/download copies, PDF page images,
# saved responses) goes into one folder per run, deleted on exit. A forced kill
# skips that, so each launch also deletes folders left by runs that are gone.
APP_TEMP_ROOT = os.path.join(tempfile.gettempdir(), "evren-pdf2text")
RUN_DIR       = os.path.join(APP_TEMP_ROOT, f"run-{os.getpid()}")
os.makedirs(RUN_DIR, exist_ok=True)
# Gradio reads this at import time, so it must be set before `import gradio`
os.environ["GRADIO_TEMP_DIR"] = os.path.join(RUN_DIR, "gradio")

def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return bool(ok) and exit_code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)  # POSIX only: signal 0 just checks existence
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

def folder_stats(path):
    files = [os.path.join(root, f) for root, _, names in os.walk(path) for f in names]
    size  = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    return len(files), size / 1_048_576

def sweep_stale_runs():
    for name in os.listdir(APP_TEMP_ROOT):
        path = os.path.join(APP_TEMP_ROOT, name)
        pid  = name.removeprefix("run-")
        if path == RUN_DIR or not name.startswith("run-") or not pid.isdigit():
            continue
        if not pid_alive(int(pid)):
            n_files, mb = folder_stats(path)
            shutil.rmtree(path, ignore_errors=True)
            print(f" ✔ Deleted leftovers of an earlier run: {path} ({n_files} files, {mb:.1f} MB)")

def cleanup_temp_files():
    n_files, mb = folder_stats(RUN_DIR)
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    status = "✔ Deleted" if not os.path.exists(RUN_DIR) else "❌ Could not fully delete"
    print(f"\n{status} temp folder: {RUN_DIR} ({n_files} files, {mb:.1f} MB)")

def exit_on_signal(signum, frame):
    sys.exit(0)  # raises SystemExit, so atexit cleanup runs

sweep_stale_runs()
atexit.register(cleanup_temp_files)
signal.signal(signal.SIGTERM, exit_on_signal)
if hasattr(signal, "SIGBREAK"):  # Windows: Ctrl-Break
    signal.signal(signal.SIGBREAK, exit_on_signal)

import gradio as gr
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import time
from pdf2image import convert_from_path

# ── Config loaders ─────────────────────────────────────────────────────────────
def load_text_file(filepath, fallback=""):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using default.")
        return fallback
    if not text:
        print(f"Warning: '{filepath}' is empty. Using default.")
        return fallback
    return text

def load_models(filepath, fallback=None):
    fallback = fallback or ["auto"]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            models = [l.strip() for l in f if l.strip()]
            return models if models else fallback
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using default model list.")
        return fallback

# ── EVREN client ───────────────────────────────────────────────────────────────
EVREN_URL = os.environ.get("EVREN_URL", "https://evren-llmapi.ssyz.org.tr/v1")
EVREN_KEY = os.environ.get("EVREN_API_KEY")
if not EVREN_KEY:
    raise SystemExit("EVREN_API_KEY environment variable is not set.")

client = OpenAI(base_url=EVREN_URL, api_key=EVREN_KEY)

FALLBACK_CONTEXT     = 131072
FALLBACK_MAX_OUTPUT  = 16384
MIN_MAX_TOKENS       = 2048   # docs: reasoning models share max_tokens between thinking and answer

def model_info(vision=False, context=FALLBACK_CONTEXT,
               max_output=FALLBACK_MAX_OUTPUT, reasoning_effort=False):
    return {"vision": vision, "context": context, "max_output": max_output,
            "reasoning_effort": reasoning_effort}

def fetch_evren_models():
    """Return {model_id: info} for chat-capable models from /v1/models, or None on failure."""
    try:
        infos = {}
        for m in client.models.list().data:
            d = m.model_dump()
            if d.get("task", "chat") not in ("chat", "vision_chat"):
                continue  # skip embedding / rerank / ocr / asr / decision models
            caps = d.get("capabilities") or {}
            infos[m.id] = model_info(
                vision=bool(caps.get("vision")) or "image" in (d.get("modalities") or []),
                context=d.get("context_length") or FALLBACK_CONTEXT,
                max_output=d.get("default_max_output_tokens") or FALLBACK_MAX_OUTPUT,
                reasoning_effort=bool(caps.get("reasoning_effort")),
            )
        return infos or None
    except Exception as e:
        print(f"Warning: could not fetch model list from EVREN ({e}). Using models.txt files.")
        return None

# ── Load config ────────────────────────────────────────────────────────────────
MODEL_INFO = fetch_evren_models()
if not MODEL_INFO:
    vision_set = set(load_models("models-image.txt"))
    MODEL_INFO = {m: model_info(vision=m in vision_set)
                  for m in dict.fromkeys(load_models("models.txt") + list(vision_set))}

def info_for(model_id):
    return MODEL_INFO.get(model_id) or model_info()

# Vision models first, then text-only
combined_models = ([m for m in MODEL_INFO if MODEL_INFO[m]["vision"]] +
                   [m for m in MODEL_INFO if not MODEL_INFO[m]["vision"]])

default_model         = combined_models[0] if combined_models else None
initial_system_prompt = load_text_file("system-prompt.txt", "You are a helpful assistant.")

# ── PDF / image helpers ────────────────────────────────────────────────────────
MAX_IMAGE_DIMENSION = 2240
PDF_DPI = 200

def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    # pdf2image writes page files to disk; keep them inside this run's folder
    pdf_out_dir = tempfile.mkdtemp(prefix="pdf_extract_", dir=RUN_DIR)
    pages = convert_from_path(pdf_path, dpi=PDF_DPI, output_folder=pdf_out_dir)
    images = []
    for page in pages:
        if page.mode in ("RGBA", "P"):
            page = page.convert("RGB")
        if max(page.size) > MAX_IMAGE_DIMENSION:
            page.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        images.append(page)
    return images

def file_to_images(file_obj_list) -> list[Image.Image]:
    if not file_obj_list:
        return []

    # If a single file object somehow slips through, wrap it in a list
    if not isinstance(file_obj_list, list):
        file_obj_list = [file_obj_list]

    all_images = []
    for file_obj in file_obj_list:
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            all_images.extend(pdf_to_images(path))
        else:
            with Image.open(path) as img:
                img = img.copy()
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                if max(img.size) > MAX_IMAGE_DIMENSION:
                    img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                all_images.append(img)
                
    return all_images

def pil_to_b64_jpeg(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Reasoning helpers ──────────────────────────────────────────────────────────
# Models that think by default but reject `reasoning_effort`; only an on/off
# chat-template switch works (gemma-4 reports thinking=false yet still thinks).
TEMPLATE_TOGGLE_FAMILIES = ("gemma-4",)

def uses_template_toggle(model_choice: str) -> bool:
    return any(f in model_choice.lower() for f in TEMPLATE_TOGGLE_FAMILIES)

def has_thinking_toggle(model_choice: str) -> bool:
    return ("levels" in profile_for(model_choice)
            or info_for(model_choice)["reasoning_effort"] or uses_template_toggle(model_choice))

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P       = 1.0

# Per-family settings from official model cards. Models without an entry get
# off/on (if they have a thinking toggle) and default sampling. Keys are matched
# as substrings of the model id.
MODEL_PROFILES = {
    # https://huggingface.co/Qwen/Qwen3.8-Flash-Next — xhigh is the default.
    # EVREN passes levels through unmapped (except high→xhigh); the model rejects
    # minimal/max. Measured via EVREN (2026-10-14) the levels don't order
    # consistently (low thought more than xhigh on math, less on a word puzzle),
    # but xhigh was always correct while low and off failed often. "off" sends
    # reasoning_effort="none" (same as the card's enable_thinking=False).
    "qwen3.8": {
        "levels":   ["off", "low", "medium", "xhigh"],
        "sampling": {"on":  {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
                     "off": {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "presence_penalty": 1.5}},
    },
    # Z.ai docs: GLM-5.3 always thinks; reasoning_effort low / high / max
    # (max is the default). No "off": EVREN's "none" doesn't stop thinking, it
    # merges the thinking into the answer text with no separator. Measured via
    # EVREN (2026-10-14): low < high < max (= default); EVREN seems to map
    # minimal→low, medium→high, xhigh→max.
    "glm-5.3": {
        "levels": ["low", "high", "max"],
    },
}

def profile_for(model_choice: str) -> dict:
    return next((p for f, p in MODEL_PROFILES.items() if f in model_choice.lower()), {})

def thinking_levels(model_choice: str) -> list[str]:
    return profile_for(model_choice).get("levels", ["off", "on"])

def default_thinking(model_choice: str) -> str:
    return thinking_levels(model_choice)[-1]

def profile_sampling(model_choice: str, thinking: str) -> dict | None:
    sampling = profile_for(model_choice).get("sampling")
    return sampling and sampling["off" if thinking == "off" else "on"]

def sampling_info(model_choice: str, thinking: str) -> str:
    s = profile_sampling(model_choice, thinking)
    if s:
        return (f"Model card: top_p {s['top_p']} · top_k {s['top_k']} · "
                f"presence_penalty {s['presence_penalty']} (set automatically)")
    return f"top_p {DEFAULT_TOP_P}"

def build_reasoning_params(model_choice: str, thinking: str) -> dict:
    """Extra kwargs for chat.completions.create controlling reasoning.

    /v1/models advertises reasoning_effort support (mimo lists none..xhigh), but
    measured on deepseek, qwen3.8, glm and mimo the levels don't scale thinking
    (glm: low/high shorter than medium); only "none" reliably turns it off. So
    models get off/on unless a MODEL_PROFILES entry lists the model card's levels.
    """
    if not has_thinking_toggle(model_choice) or thinking not in thinking_levels(model_choice):
        return {}  # no switch, or a value carried over from another model: server default

    if info_for(model_choice)["reasoning_effort"]:
        if thinking == "off":
            return {"reasoning_effort": "none"}
        if thinking == "on":
            return {}  # server default, which thinks
        return {"reasoning_effort": thinking}

    if uses_template_toggle(model_choice):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": thinking != "off"}}}

    return {}

def build_sampling_params(model_choice: str, thinking: str, temperature: float) -> dict:
    """temperature comes from the UI slider (pre-set from the model card when known)."""
    s = profile_sampling(model_choice, thinking)
    if not s:
        return {"temperature": temperature, "top_p": DEFAULT_TOP_P}
    return {"temperature": temperature, "top_p": s["top_p"],
            "presence_penalty": s["presence_penalty"], "extra_body": {"top_k": s["top_k"]}}

# ── Core chat function ─────────────────────────────────────────────────────────
def chat(message, history, api_history, model_choice, instructions,
         temperature, max_tokens, thinking, pages_per_request, input_file):

    initial_dl = gr.update(visible=False)

    if not message.strip() and not input_file:
        yield history, api_history, "", "*No reasoning generated yet...*", initial_dl
        return

    # ── Convert upload → list of PIL images ───────────────────────────────────
    is_vision = info_for(model_choice)["vision"]
    if input_file and not is_vision:
        gr.Warning(f"{model_choice} cannot read images; attached files were ignored.")
        if not message.strip():
            yield history, api_history, gr.update(), "*No reasoning generated yet...*", initial_dl
            return

    # Clear the textbox right away (message is already captured); PDF
    # conversion and the first tokens can take a while
    yield history, api_history, "", "*Waiting for the model...*", initial_dl

    images = file_to_images(input_file) if is_vision else []

    # ── Build user content ────────────────────────────────────────────────────
    # api_history holds text only: page images go into this turn's request(s)
    # alone and api_history gets a placeholder instead, so follow-ups don't
    # resend them and the history stays valid after switching to a text-only model.
    prompt_text = message or "Describe the content of these images in detail."
    if images:
        n_pages = len(images)
        user_content_display = f"📎 [Attached {n_pages} image(s)/page(s)] {message}".strip()
        user_content_api = (f"[{n_pages} image(s)/page(s) were attached here; "
                            f"their content is in the following reply.]\n{prompt_text}")
    else:
        user_content_display = message
        user_content_api     = message

    # ── Assemble messages for the API ─────────────────────────────────────────
    base_messages = []
    if instructions.strip():
        base_messages.append({"role": "system", "content": instructions})
    base_messages += [{"role": m["role"], "content": m["content"]}
                      for m in api_history if m["content"]]

    if images:
        step    = max(1, int(pages_per_request))
        batches = [(i, images[i:i + step]) for i in range(0, len(images), step)]
        requests = [
            base_messages + [{"role": "user", "content": [
                {"type":      "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{pil_to_b64_jpeg(img)}"}}
                for img in batch
            ] + [{"type": "text", "text": prompt_text}]}]
            for _, batch in batches
        ]
        labels = [f"Pages {i + 1}–{i + len(batch)}" if len(batch) > 1 else f"Page {i + 1}"
                  for i, batch in batches]
    else:
        requests = [base_messages + [{"role": "user", "content": message}]]
        labels   = [None]

    # Update both the UI history and the hidden API history
    history = history + [
        {"role": "user",      "content": user_content_display},
        {"role": "assistant", "content": ""},
    ]
    api_history = api_history + [
        {"role": "user",      "content": user_content_api},
        {"role": "assistant", "content": ""},
    ]

    # ── API calls (one per page batch) ────────────────────────────────────────
    full_content      = ""
    reasoning_content = ""
    try:
        reasoning_params = build_reasoning_params(model_choice, thinking)
        sampling_params  = build_sampling_params(model_choice, thinking, temperature)
        extra_body = {**reasoning_params.pop("extra_body", {}),
                      **sampling_params.pop("extra_body", {})}
        common_params = {
            "model":      model_choice,
            "max_tokens": int(max_tokens),
            "stream":     True,
            **sampling_params,
            **reasoning_params,
        }
        if extra_body:
            common_params["extra_body"] = extra_body

        last_yield_time  = time.time()
        flush_interval_s = 0.04

        for idx, (messages, label) in enumerate(zip(requests, labels)):
            if len(requests) > 1:
                if idx:
                    full_content += "\n\n---\n\n"
                reasoning_content += f"\n\n**{label}**\n\n"

            completion = client.chat.completions.create(messages=messages, **common_params)

            for chunk in completion:
                if not chunk.choices:
                    continue

                delta        = chunk.choices[0].delta
                new_content  = getattr(delta, "content",   None) or None
                new_reasoning= (getattr(delta, "reasoning", None)
                                or getattr(delta, "reasoning_content", None) or None)

                if new_content is not None:
                    full_content += new_content
                    history[-1]["content"] = full_content
                    api_history[-1]["content"] = full_content  # Keep API history synced

                if new_reasoning is not None:
                    reasoning_content += new_reasoning

                if time.time() - last_yield_time >= flush_interval_s:
                    last_yield_time = time.time()
                    thoughts = reasoning_content or ("*No separate reasoning for this answer.*"
                                                     if full_content else "*Waiting for the model...*")
                    yield history, api_history, gr.update(), thoughts, initial_dl

        # ── Save last response to temp file ───────────────────────────────────
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".md", mode="w", encoding="utf-8", dir=RUN_DIR
        ) as tf:
            tf.write(full_content)
            output_filepath = tf.name

        yield (history, api_history, "", reasoning_content or "*No separate reasoning for this answer.*",
               gr.update(visible=True, value=output_filepath))

    except Exception as e:
        error_text = f"❌ An error occurred: {e}"
        history[-1]["content"] = f"{full_content}\n\n{error_text}" if full_content else error_text
        # Never feed the error text back to the model: keep partial output, or drop the turn
        if full_content:
            api_history[-1]["content"] = full_content
        else:
            api_history = api_history[:-2]
        yield history, api_history, "", reasoning_content or f"An error occurred: {e}", initial_dl


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="💬 EVREN Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot (Powered by EVREN LLM API)")
    
    # Conversation history sent to the API: text only (page images are sent
    # with their own turn and stored here as a placeholder)
    api_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(height=500, buttons=["copy"])

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type a message...", scale=4, show_label=False
                )
                send_btn = gr.Button("Send", scale=1)

            input_file = gr.File(
                label="📎 Attach Image(s) or PDF(s) (cleared after each send)",
                file_types=["image", ".pdf"],
                file_count="multiple",
                visible=info_for(default_model)["vision"],
            )
            pages_per_request = gr.Slider(
                1, 32, value=1, step=1,
                label="Pages per request",
                info="1 = transcribe page by page · raise for questions spanning several pages",
                visible=info_for(default_model)["vision"],
            )

            with gr.Row():
                stop_btn     = gr.Button("Stop",  scale=1)
                clear_btn    = gr.Button("Clear Chat", scale=1)
                download_btn = gr.DownloadButton(
                    "⬇️ Download Last Response", visible=False, scale=3
                )

        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                label="Choose a Model",
                choices=combined_models,
                value=default_model,
            )

            thinking = gr.Radio(
                thinking_levels(default_model),
                value=default_thinking(default_model),
                label="🧠 Reasoning",
                info="off = answers directly · otherwise the model thinks before answering",
                visible=has_thinking_toggle(default_model),
            )

            instructions = gr.Textbox(
                label="System Instructions",
                value=initial_system_prompt,
                lines=3,
            )
            initial_sampling = profile_sampling(default_model, default_thinking(default_model)) or {}
            temperature = gr.Slider(
                0.0, 2.0, value=initial_sampling.get("temperature", DEFAULT_TEMPERATURE),
                step=0.1, label="Temperature",
                info=sampling_info(default_model, default_thinking(default_model)),
            )
            default_info = info_for(default_model)
            max_tokens = gr.Slider(
                MIN_MAX_TOKENS, default_info["context"],
                value=min(default_info["max_output"], default_info["context"]),
                step=256, label="Max Tokens",
                info="Shared by thinking and answer; limit = model context length",
            )
            # gr.Markdown doesn't render its label (container=False), so use a heading
            gr.Markdown("### 🧠 Model Thoughts")
            thoughts_box = gr.Markdown(value="*Reasoning will appear here...*")

    def temperature_update(model_name, thinking_value):
        s = profile_sampling(model_name, thinking_value)
        return gr.update(value=s["temperature"] if s else DEFAULT_TEMPERATURE,
                         info=sampling_info(model_name, thinking_value))

    def on_model_change(model_name, current_max_tokens):
        info   = info_for(model_name)
        levels = thinking_levels(model_name)
        # Always start from the model's default: a level chosen for another model
        # (e.g. "off" or "low") can hurt accuracy badly on this one
        new_thinking = default_thinking(model_name)
        return (
            gr.update(visible=info["vision"]),
            gr.update(visible=info["vision"]),
            gr.update(choices=levels, value=new_thinking, visible=has_thinking_toggle(model_name)),
            gr.update(maximum=info["context"],
                      value=max(MIN_MAX_TOKENS, min(int(current_max_tokens), info["context"]))),
            temperature_update(model_name, new_thinking),
        )

    model_choice.change(
        fn=on_model_change,
        inputs=[model_choice, max_tokens],
        outputs=[input_file, pages_per_request, thinking, max_tokens, temperature],
        queue=False,
    )

    # Model-card sampling differs between thinking and non-thinking mode
    thinking.input(
        fn=temperature_update,
        inputs=[model_choice, thinking],
        outputs=[temperature],
        queue=False,
    )

    # ── Wire up send / submit ──────────────────────────────────────────────────
    inputs  = [msg, chatbot_ui, api_history, model_choice, instructions,
               temperature, max_tokens, thinking, pages_per_request, input_file]
    outputs = [chatbot_ui, api_history, msg, thoughts_box, download_btn]

    e_submit = msg.submit(chat, inputs, outputs)
    e_click  = send_btn.click(chat, inputs, outputs)

    e_submit.then(fn=lambda: gr.update(value=None), outputs=[input_file])
    e_click.then( fn=lambda: gr.update(value=None), outputs=[input_file])

    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click], queue=False)

    clear_btn.click(
        fn=lambda: (
            [],
            [], 
            "*Reasoning will appear here...*",
            gr.update(visible=False),
            gr.update(value=None),
        ),
        outputs=[chatbot_ui, api_history, thoughts_box, download_btn, input_file],
        cancels=[e_submit, e_click],
        queue=False,
    )

demo.queue()

if __name__ == "__main__":
    print("Launching Gradio interface... Press Ctrl+C to exit.")
    print("Temporary files for this session will be cleaned up automatically on exit.")
    demo.launch(theme=gr.themes.Default())