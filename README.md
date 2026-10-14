# text-vision-pdf2text.py

A Gradio chat app for the [EVREN LLM API](https://evren-llmapi.ssyz.org.tr) (OpenAI-compatible). Chat with any EVREN text model, or attach images and PDFs to a vision model, for example to transcribe (OCR) a PDF page by page.

## Features

- **Live model list.** Models and their capabilities (vision, context length, reasoning support) come from EVREN's `/v1/models`. If that call fails, `models.txt` and `models-image.txt` are used instead.
- **Images and PDFs.** Attach several images and/or PDFs at once. Each PDF page becomes an image, and pages are sent in batches you choose (*Pages per request*).
- **Streaming answers** with the model's reasoning shown in a separate *Model Thoughts* panel.
- **Reasoning switch** for each model: off/on, `low`/`medium`/`xhigh` for qwen3.8, or `low`/`high`/`max` for glm-5.3 (see [Reasoning controls](#reasoning-controls)). Switching models resets it to that model's default.
- **Recommended sampling settings from the model card** for qwen3.8, applied automatically. All other models use the defaults.
- **Download** the last answer as a `.md` file.
- **Temp-file cleanup.** Everything the app writes is deleted when it stops.

## Requirements

- Python 3.10+
- Python packages: `gradio` (6.x), `openai`, `pillow`, `pdf2image`
- [Poppler](https://poppler.freedesktop.org/), which `pdf2image` needs to read PDFs:
  - Windows: download a [Poppler build](https://github.com/oschwartz10612/poppler-windows/releases) and add its `bin` folder to `PATH`
  - Debian/Ubuntu: `sudo apt install poppler-utils`
  - macOS: `brew install poppler`
- An EVREN API key (`evren_llm_...`) from the EVREN `/api-keys` page

### First-time EVREN setup

Each EVREN account has to accept the terms of use once before any chat call works. Without it, every request fails with `403 terms_not_accepted`.

```bash
curl https://evren-llmapi.ssyz.org.tr/v1/terms/status -H "X-API-Key: $EVREN_API_KEY"   # note current_version
curl -X POST https://evren-llmapi.ssyz.org.tr/v1/terms/accept \
     -H "X-API-Key: $EVREN_API_KEY" -H "Content-Type: application/json" \
     -d '{"version": 1}'                                                                  # use current_version
```

## Running

Set your API key:

```bash
export EVREN_API_KEY="evren_llm_..."          # Linux / macOS
$env:EVREN_API_KEY = "evren_llm_..."          # Windows PowerShell
```

Start the app with [uv](https://docs.astral.sh/uv/):

```bash
uv run --with gradio --with openai --with pillow --with pdf2image text-vision-pdf2text.py
```

Or install the packages into a virtual environment and run it with Python:

```bash
pip install gradio openai pillow pdf2image
python text-vision-pdf2text.py
```

Then open http://127.0.0.1:7860. If that port is taken, Gradio picks the next free one and prints it.

| Environment variable | Required | Default |
|---|---|---|
| `EVREN_API_KEY` | yes | — |
| `EVREN_URL` | no | `https://evren-llmapi.ssyz.org.tr/v1` |

## Files

| File | Purpose |
|---|---|
| `text-vision-pdf2text.py` | The app |
| `system-prompt.txt` | Optional. Starting text for the *System Instructions* box. If the file is missing or empty, the box starts with "You are a helpful assistant." |
| `models.txt` | Fallback list of models, one per line, used only if `/v1/models` can't be reached |
| `models-image.txt` | Fallback list of the vision models, one per line |

## Usage

1. **Choose a model.** Vision models are listed first. The attachment box and *Pages per request* appear only for vision models.
2. **Attach files (optional).** Images or PDFs, several at once. The attachment box is cleared after each send.
3. **Type a message** (for example "Transcribe this document"). If you send files without a message, the model is asked to describe their content.
4. **Read the answer** as it streams in. Download it with **⬇️ Download Last Response**.

### Pages per request

PDF pages are rendered at 200 DPI, scaled down to at most 2240 px on the longest side, and sent as JPEG. *Pages per request* sets how many pages go into one API call:

- **1 (default): use this for OCR / transcription.** Each page gets the full *Max Tokens* budget, and the model doesn't skip or summarize pages. A 16-page PDF becomes 16 calls, sent one after another. The answers are joined with `---` lines between them.
- **Higher:** use this for questions about the whole document ("summarize this PDF", tables that run across pages). Each call only sees its own pages.

Page images are sent only with the turn they were attached to. In the conversation history they're replaced by a short note, so follow-up questions don't resend the PDF. You can also switch to a text-only model and keep asking about the transcription.

### Other settings

- **Max Tokens:** the limit for each call. Thinking and the answer both count toward it. The slider goes up to the selected model's context length and doesn't go below 2048.
- **Temperature:** defaults to 1.0. For qwen3.8 it is set from the model card when you choose the model or change the reasoning setting. You can still adjust it afterwards.
- **Stop** cancels the current answer. **Clear Chat** resets the conversation.

## Reasoning controls

EVREN accepts `reasoning_effort` values `none`, `minimal`, `low`, `medium`, `high`, `xhigh` and `max`, and how it handles them depends on the model. Levels are offered only where the model's own documentation lists them. Results tested through EVREN in October 2026:

| Model | Options | What is sent |
|---|---|---|
| deepseek-v4.1-flash, deepseek-v4-flash, mimo-v2.6-pro | off / on | off → `reasoning_effort="none"`; on → nothing (the model thinks by default) |
| glm-5.3 | low / high / max | `reasoning_effort` as documented by Z.ai (default `max`); no off, see below |
| qwen3.8-flash-next | off / low / medium / xhigh | off → `reasoning_effort="none"`; levels as documented on the [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (default `xhigh`) |
| gemma-4-31b | off / on | `chat_template_kwargs.enable_thinking` (gemma rejects `reasoning_effort`) |
| qwen3-vl-30b, auto | — | no thinking control |

Things to know:

- **Use qwen3.8 at `xhigh` (the default).** EVREN passes qwen's levels straight to the model, treating only `high` as `xhigh`; the model rejects `minimal` and `max`. How much qwen thinks at each level varied from task to task, but `xhigh` answered every test question correctly. `low` and thinking off failed often: on "How many days in the week have the letter d in them?" both went 0/3, claiming Tuesday has no "d".
- **Thinking off can lower accuracy** on any model. Leave it on for anything that needs careful reasoning. Turning it off makes plain transcription faster.
- **glm-5.3 always thinks.** Z.ai's documentation says its thinking can't be disabled. Through EVREN, "off" makes glm merge its thinking into the answer with no separator, so the app offers only glm's levels. Measured: `low` < `high` < `max`, and `max` is the default. On easy questions `low` and `high` think about the same amount.

### qwen3.8 sampling

When qwen3.8-flash-next is selected, the app uses the model card's recommended settings:

| Mode | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| thinking (low / medium / xhigh) | 1.0 | 0.95 | 20 | 0.0 |
| off | 0.7 | 0.80 | 20 | 1.5 |

All other models use temperature 1.0 and `top_p` 1.0, with no `top_k` or `presence_penalty`. To give another model its own settings, add an entry to `MODEL_PROFILES` in the script.

## Temporary files

Each run writes only to `<system temp>/evren-pdf2text/run-<pid>/`: uploads, Gradio's download copies, PDF page images and saved answers.

- **Normal stop** (Ctrl-C, Ctrl-Break, or `SIGTERM` on Linux/macOS): the folder is deleted, and the app prints how many files and MB it removed.
- **Forced kill** (Task Manager, `kill -9`, and possibly closing the terminal window): cleanup may not run. The next launch deletes folders left by runs that are no longer running. Two copies of the app can run at the same time without deleting each other's files.

## Notes

- EVREN's free daily quota is 15M tokens per account, across all models. Page images add up quickly, so check your remaining quota with `GET /v1/quota`.
- EVREN's dedicated OCR models (`dots-ocr`, `deepseek-ocr-2`) work only through `/v1/ocr`, not the chat endpoint, so this app doesn't list them.
- The model fleet and its behavior can change. If a model is added or changes, test its reasoning settings again before relying on the tables above.

## License

Released under the [MIT License](https://opensource.org/licenses/MIT): you may use, copy, modify, merge, publish, distribute, sublicense and sell copies of this software, provided the copyright notice and permission notice below are included. The software comes with no warranty.

```
MIT License

Copyright (c) 2026 zakcali

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
