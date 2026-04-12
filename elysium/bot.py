"""
Elysium — autonomous Gemini-powered Telegram group bot.
No slash commands. Gemini decides what tools to call based on conversation.
Reads all group messages to learn speech style.
Responds only when mentioned, tagged, or replied to.
"""

import os
import re
import asyncio
import logging
import hashlib
import time
from collections import deque
from pathlib import Path
from io import BytesIO

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode
import html
import google.generativeai as genai
from google.generativeai import types as genai_types

# ── Config ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
BOT_NAME        = "elysium"
BOT_USERNAME    = None

WORKSPACE       = Path("/workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

TIMEOUT_SECONDS  = 60
MSG_BUFFER_SIZE  = 300

genai.configure(api_key=GEMINI_API_KEY)
CHAT_MODEL  = "gemini-3.1-flash-lite-preview"
IMAGE_MODEL = "gemini-3.1-flash-image-preview"

_group_msgs:      dict[int, deque]             = {}
_sessions:        dict[int, genai.ChatSession] = {}
_pending_runs:    dict[str, tuple[str, str]]   = {}
_pending_outputs: dict[str, str]               = {}
_file_context:    dict[int, list[dict]]        = {}

# Interactive process sessions: chat_id -> InteractiveSession
_interactive:     dict[int, "InteractiveSession"] = {}
_chat_locks:      dict[int, asyncio.Lock]         = {}   # one request at a time per chat


def _get_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in _chat_locks:
        _chat_locks[chat_id] = asyncio.Lock()
    return _chat_locks[chat_id]

OUTPUT_SHORT_LIMIT = 400
OUTPUT_LINES_LIMIT = 12
INPUT_WAIT_TIMEOUT = 3.0   # seconds to wait for more output before assuming process wants input


class InteractiveSession:
    def __init__(self, proc: asyncio.subprocess.Process, cmd: str):
        self.proc       = proc
        self.cmd        = cmd
        self.output_buf = ""          # accumulated output so far
        self.input_event = asyncio.Event()
        self.pending_input: str | None = None

    def feed(self, text: str):
        self.pending_input = text + "\n"
        self.input_event.set()

# ── Tool definitions for Gemini function calling ──────────────────────────────

TOOLS = [
    genai_types.Tool(function_declarations=[
        genai_types.FunctionDeclaration(
            name="execute_shell",
            description=(
                "Run any shell command or script inside the Docker workspace. "
                "Use for: running code, installing packages, file operations, system tasks, "
                "executing uploaded scripts, compiling, anything terminal-related."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        ),
        genai_types.FunctionDeclaration(
            name="generate_image",
            description="Generate an image from a text description.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Detailed image description"}
                },
                "required": ["prompt"]
            }
        ),
        genai_types.FunctionDeclaration(
            name="list_files",
            description="List files in the workspace directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Subdirectory path (optional)"}
                }
            }
        ),
        genai_types.FunctionDeclaration(
            name="read_file",
            description="Read the contents of a file in the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename to read"}
                },
                "required": ["filename"]
            }
        ),
        genai_types.FunctionDeclaration(
            name="write_file",
            description="Write content to a file in the workspace. Use for saving scripts, configs, or any text.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename to write"},
                    "content":  {"type": "string", "description": "File content"}
                },
                "required": ["filename", "content"]
            }
        ),
        genai_types.FunctionDeclaration(
            name="send_file",
            description="Send a file from the workspace back to the Telegram chat.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename to send"}
                },
                "required": ["filename"]
            }
        ),
        genai_types.FunctionDeclaration(
            name="web_search",
            description=(
                "Search the web using DuckDuckGo. Use this for any question about current events, "
                "facts, people, places, news, or anything that benefits from up-to-date information. "
                "Always prefer this over writing a scraper."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query":      {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results (default 5, max 10)"}
                },
                "required": ["query"]
            }
        ),
    ])
]

# ── Tool execution ────────────────────────────────────────────────────────────

async def tool_execute_shell(command: str) -> str:
    """Non-interactive shell execution (for Gemini tool calls)."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=str(WORKSPACE)
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SECONDS)
        output = stdout.decode(errors="replace").strip()
        return f"exit_code={proc.returncode}\n{output[:3000] or '(no output)'}"
    except asyncio.TimeoutError:
        return f"error=timed out after {TIMEOUT_SECONDS}s"
    except Exception as e:
        return f"error={e}"


async def run_interactive(command: str, chat_id: int, status_msg, bot) -> None:
    """
    Run a command interactively.
    Streams output to Telegram. When the process blocks waiting for stdin,
    asks the user for input. Feeds replies back to the process.
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.PIPE,
        cwd=str(WORKSPACE)
    )

    session = InteractiveSession(proc, command)
    _interactive[chat_id] = session

    accumulated = ""
    full_output = ""

    async def read_chunk() -> str | None:
        """Read available output with a short timeout. Returns None on timeout."""
        try:
            chunk = await asyncio.wait_for(proc.stdout.read(512), timeout=INPUT_WAIT_TIMEOUT)
            return chunk.decode(errors="replace") if chunk else ""
        except asyncio.TimeoutError:
            return None

    try:
        while True:
            chunk = await read_chunk()

            if chunk is None:
                # Timed out — process is likely waiting for input
                if proc.returncode is not None:
                    break
                prompt_text = accumulated.strip()
                accumulated = ""
                if prompt_text:
                    full_output += prompt_text + "\n"
                    display = f"<pre>{html.escape(prompt_text[-3000:])}</pre>\n\nType your response:"
                    await status_msg.edit_text(display, parse_mode=ParseMode.HTML)
                else:
                    await status_msg.edit_text("Waiting for your input:")

                # Wait for user to reply (up to 5 minutes)
                session.input_event.clear()
                try:
                    await asyncio.wait_for(session.input_event.wait(), timeout=300)
                except asyncio.TimeoutError:
                    proc.kill()
                    await status_msg.edit_text("Timed out waiting for input. Process killed.")
                    break

                user_input = session.pending_input or "\n"
                full_output += f"[input: {user_input.strip()}]\n"
                proc.stdin.write(user_input.encode())
                await proc.stdin.drain()

                # Acknowledge input
                await status_msg.edit_text(f"Input sent: {user_input.strip()}\n\nRunning...")
                continue

            if chunk == "":
                # EOF — process done
                break

            accumulated += chunk
            full_output += chunk

        # Process finished — show final output
        await asyncio.wait_for(proc.wait(), timeout=10)
        code = proc.returncode or 0

        # Drain any remaining output
        try:
            rest = await asyncio.wait_for(proc.stdout.read(), timeout=2)
            if rest:
                accumulated += rest.decode(errors="replace")
                full_output += accumulated
        except Exception:
            pass

        body = (accumulated + full_output).strip()
        # Deduplicate (full_output may contain accumulated)
        body = full_output.strip() or accumulated.strip() or "(no output)"

        is_long = len(body) > OUTPUT_SHORT_LIMIT or len(body.splitlines()) > OUTPUT_LINES_LIMIT
        if is_long:
            oid = _make_run_id(command)
            _pending_outputs[oid] = f"$ {command}\n\n{body}\nexit_code={code}"
            preview_lines = body.splitlines()[:3]
            preview = "\n".join(preview_lines)
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("📄 Show full output", callback_data=f"out:{oid}")
            ]])
            await status_msg.edit_text(
                f"<pre>{html.escape(preview)}\n...</pre>\nexit_code={code}",
                parse_mode=ParseMode.HTML,
                reply_markup=kb
            )
        else:
            await status_msg.edit_text(
                f"<pre>{html.escape(body)}</pre>\nexit_code={code}",
                parse_mode=ParseMode.HTML
            )

    except Exception as e:
        logger.exception("run_interactive error")
        try:
            await status_msg.edit_text(f"Error: {e}")
        except Exception:
            pass
    finally:
        _interactive.pop(chat_id, None)
        try:
            proc.kill()
        except Exception:
            pass


async def tool_web_search(query: str, max_results: int = 5) -> str:
    try:
        import httpx, urllib.parse
        max_results = min(max(1, max_results), 8)
        params = {"q": query, "format": "json", "no_html": "1", "no_redirect": "1"}
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ElysiumBot/1.0)"}

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                "https://api.duckduckgo.com/",
                params=params,
                headers=headers
            )
            data = r.json()

        results = []

        # Instant answer
        if data.get("AbstractText"):
            results.append(
                f"Title: {data.get('Heading','')}\n"
                f"Summary: {data['AbstractText']}\n"
                f"URL: {data.get('AbstractURL','')}"
            )

        # Related topics
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if "Text" in topic and "FirstURL" in topic:
                results.append(
                    f"Title: {topic['Text'][:120]}\n"
                    f"URL: {topic['FirstURL']}"
                )
            elif "Topics" in topic:
                for sub in topic["Topics"]:
                    if len(results) >= max_results:
                        break
                    if "Text" in sub and "FirstURL" in sub:
                        results.append(
                            f"Title: {sub['Text'][:120]}\n"
                            f"URL: {sub['FirstURL']}"
                        )

        if not results:
            # Fallback: DuckDuckGo HTML search via lite endpoint
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(
                    "https://lite.duckduckgo.com/lite/",
                    params={"q": query},
                    headers=headers
                )
            from html.parser import HTMLParser

            class _P(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results, self._cur = [], {}
                    self._in_a = False
                def handle_starttag(self, tag, attrs):
                    if tag == "a":
                        d = dict(attrs)
                        href = d.get("href","")
                        if href.startswith("http") and "duckduckgo" not in href:
                            self._cur = {"url": href}
                            self._in_a = True
                def handle_data(self, data):
                    if self._in_a and data.strip():
                        self._cur["title"] = data.strip()
                def handle_endtag(self, tag):
                    if tag == "a" and self._in_a:
                        if self._cur.get("url") and self._cur.get("title"):
                            self.results.append(self._cur)
                        self._cur = {}
                        self._in_a = False

            p = _P()
            p.feed(r.text)
            for item in p.results[:max_results]:
                results.append(f"Title: {item['title']}\nURL: {item['url']}")

        return "\n\n".join(results) if results else "No results found."

    except asyncio.TimeoutError:
        return "Search timed out."
    except Exception as e:
        return f"Search error: {e}"


async def tool_generate_image(prompt: str) -> bytes | None:
    try:
        model = genai.GenerativeModel(IMAGE_MODEL)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai_types.GenerationConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                return part.inline_data.data
    except Exception as e:
        logger.error(f"image gen error: {e}")
    return None


def tool_list_files(path: str = "") -> str:
    target = (WORKSPACE / path).resolve() if path else WORKSPACE
    if not str(target).startswith(str(WORKSPACE)) or not target.exists():
        return "error=path not found"
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        return "workspace is empty"
    lines = []
    for e in entries:
        if e.is_dir():
            lines.append(f"dir: {e.name}/")
        else:
            s = e.stat().st_size
            lines.append(f"file: {e.name} ({s} bytes)")
    return "\n".join(lines)


def tool_read_file(filename: str) -> str:
    fpath = (WORKSPACE / filename).resolve()
    if not str(fpath).startswith(str(WORKSPACE)):
        return "error=path outside workspace"
    if not fpath.exists():
        return f"error=file not found: {filename}"
    try:
        return fpath.read_text(errors="replace")[:4000]
    except Exception as e:
        return f"error={e}"


def tool_write_file(filename: str, content: str) -> str:
    fpath = (WORKSPACE / filename).resolve()
    if not str(fpath).startswith(str(WORKSPACE)):
        return "error=path outside workspace"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(content, encoding="utf-8")
    return f"written {len(content)} chars to {filename}"


# ── Gemini session ────────────────────────────────────────────────────────────

def _style_snapshot(chat_id: int) -> str:
    buf = _group_msgs.get(chat_id)
    if not buf or len(buf) < 5:
        return ""
    sample = list(buf)[-60:]
    return "\n".join(f"{u}: {t}" for u, t in sample)


def _build_model(chat_id: int) -> genai.GenerativeModel:
    style = _style_snapshot(chat_id)
    style_block = ""
    if style:
        style_block = (
            "\n\nSample of how this group talks:\n---\n"
            + style +
            "\n---\nBlend into their tone and vocabulary naturally. Don't copy anyone — absorb the group vibe."
        )

    return genai.GenerativeModel(
        CHAT_MODEL,
        tools=TOOLS,
        system_instruction=(
            "You are Elysium, a sharp, autonomous AI living in this Telegram group. "
            "You run inside a Docker container at /workspace. You have full shell access.\n\n"
            "Capabilities you use proactively:\n"
            "- Search the web with web_search — use this for any question about facts, news, people, places\n"
            "- Run shell commands and scripts\n"
            "- Install packages with pip, apt, npm, etc.\n"
            "- Write scripts and tools to /workspace and run them\n"
            "- Read, write, send files\n"
            "- Generate images\n\n"
            "Rules:\n"
            "- Never use asterisks, underscores for formatting, or markdown headers — plain text only\n"
            "- For code in replies, use backtick code blocks normally — they render fine\n"
            "- Be concise and direct, no filler\n"
            "- When someone asks you to do something technical, just do it — don't explain what you're about to do, do it\n"
            "- You can write persistent helper scripts to /workspace/tools/ and reuse them later"
            + style_block
        )
    )


def get_session(chat_id: int) -> genai.ChatSession:
    if chat_id not in _sessions:
        _sessions[chat_id] = _build_model(chat_id).start_chat(history=[])
    return _sessions[chat_id]


def refresh_session(chat_id: int):
    _sessions.pop(chat_id, None)


# ── Core: send message to Gemini and handle tool calls ───────────────────────

async def gemini_respond(chat_id: int, message: str, update: Update) -> str | None:
    """
    Send message to Gemini. If it calls tools, execute them and continue.
    Returns final text reply, or None if an image was sent directly.
    """
    session = get_session(chat_id)
    tg_msg  = update.effective_message
    image_sent = False

    # Agentic loop: keep going until Gemini gives a text reply
    for _ in range(5):  # max 5 tool calls in a row before forcing a reply
        response = await asyncio.to_thread(session.send_message, message)
        part = response.candidates[0].content.parts[0]

        # Pure text reply — done
        if hasattr(part, "text") and part.text:
            return clean(part.text)

        # Function call
        if hasattr(part, "function_call") and part.function_call:
            fc   = part.function_call
            name = fc.name
            args = dict(fc.args)
            logger.info(f"Tool call: {name}({args})")

            if name == "execute_shell":
                cmd    = args["command"]
                result = await tool_execute_shell(cmd)
                lines_raw  = result.split("\n", 1)
                exit_info  = lines_raw[0]
                output_body = lines_raw[1].strip() if len(lines_raw) > 1 else ""
                out_lines   = output_body.splitlines()
                is_long     = len(output_body) > OUTPUT_SHORT_LIMIT or len(out_lines) > OUTPUT_LINES_LIMIT

                if is_long:
                    # Store full output, show summary + button
                    oid = _make_run_id(cmd)
                    _pending_outputs[oid] = f"$ {cmd}\n\n{output_body}\n{exit_info}"
                    # Ask Gemini to summarise in one line for the button label
                    preview = "\n".join(out_lines[:3])
                    display = f"<pre>{html.escape(preview)}\n...</pre>"
                    kb = InlineKeyboardMarkup([[
                        InlineKeyboardButton("📄 Show full output", callback_data=f"out:{oid}")
                    ]])
                    await tg_msg.reply_text(display, parse_mode=ParseMode.HTML, reply_markup=kb)
                else:
                    display = f"$ {html.escape(cmd)}\n\n<pre>{html.escape(output_body or '(no output)')}</pre>\n{html.escape(exit_info)}"
                    await tg_msg.reply_text(display[:4090], parse_mode=ParseMode.HTML)

                message = f"function_response: {name}\nresult: {result[:1000]}"

            elif name == "generate_image":
                prompt    = args["prompt"]
                img_bytes = await tool_generate_image(prompt)
                if img_bytes:
                    await tg_msg.reply_photo(photo=BytesIO(img_bytes), caption=prompt[:200])
                    image_sent = True
                    message = f"function_response: {name}\nresult: image sent successfully"
                else:
                    message = f"function_response: {name}\nresult: failed to generate image"

            elif name == "list_files":
                result  = tool_list_files(args.get("path", ""))
                message = f"function_response: {name}\nresult: {result}"

            elif name == "read_file":
                result  = tool_read_file(args["filename"])
                message = f"function_response: {name}\nresult: {result}"

            elif name == "write_file":
                result   = tool_write_file(args["filename"], args["content"])
                fname    = args["filename"]
                fpath    = WORKSPACE / fname
                ext      = fpath.suffix.lower().lstrip(".")
                run_langs = {"py", "sh", "js", "ts", "rb", "go"}

                # Offer run button if it's a script
                if ext in run_langs and fpath.exists():
                    run_cmd = _run_cmd_for(fname, ext)
                    rid     = _make_run_id(fname)
                    _pending_runs[rid] = (fname, run_cmd)
                    kb = InlineKeyboardMarkup([[
                        InlineKeyboardButton(f"▶  Run  ({ext})", callback_data=f"run:{rid}")
                    ]])
                    with fpath.open("rb") as f:
                        await tg_msg.reply_document(
                            document=InputFile(f, filename=fname),
                            caption=fname,
                            reply_markup=kb
                        )
                message = f"function_response: {name}\nresult: {result}"

            elif name == "send_file":
                fname = args["filename"]
                fpath = (WORKSPACE / fname).resolve()
                if fpath.exists() and str(fpath).startswith(str(WORKSPACE)):
                    ext      = fpath.suffix.lower().lstrip(".")
                    run_langs = {"py", "sh", "js", "ts", "rb", "go"}
                    kb = None
                    if ext in run_langs:
                        run_cmd = _run_cmd_for(fname, ext)
                        rid     = _make_run_id(fname)
                        _pending_runs[rid] = (fname, run_cmd)
                        kb = InlineKeyboardMarkup([[
                            InlineKeyboardButton(f"▶  Run  ({ext})", callback_data=f"run:{rid}")
                        ]])
                    with fpath.open("rb") as f:
                        await tg_msg.reply_document(
                            document=InputFile(f, filename=fname),
                            reply_markup=kb
                        )
                    message = f"function_response: {name}\nresult: sent {fname}"
                else:
                    message = f"function_response: {name}\nresult: file not found"
            elif name == "web_search":
                result  = await tool_web_search(args["query"], int(args.get("max_results", 5)))
                message = f"function_response: {name}\nresult: {result[:3000]}"

            else:
                message = f"function_response: {name}\nresult: unknown tool"

            continue  # back to top of loop

        break  # no text, no function call — stop

    if image_sent:
        return None  # already replied with image
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """
    Convert Gemini markdown to clean HTML for Telegram.
    Code blocks → <pre>, inline code → <code>, strip all other markdown.
    """
    # Triple backtick code blocks → <pre>
    def replace_block(m):
        code = m.group(2).strip()
        return f"<pre>{html.escape(code)}</pre>"
    text = re.sub(r'```(\w+)?\n?(.*?)```', replace_block, text, flags=re.DOTALL)

    # Inline backticks → <code>
    text = re.sub(r'`([^`]+)`', lambda m: f"<code>{html.escape(m.group(1))}</code>", text)

    # Strip bold/italic/headings
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{1,3}(.*?)_{1,3}',   r'\1', text, flags=re.DOTALL)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Escape any raw HTML tags Gemini might have output (except our own <pre>/<code>)
    # We do this by escaping < and > that aren't part of our own tags
    allowed = re.compile(r'<(/?(pre|code))>')
    parts = re.split(r'(</?(?:pre|code)>)', text)
    cleaned = []
    for part in parts:
        if allowed.fullmatch(part):
            cleaned.append(part)
        else:
            cleaned.append(part.replace('<', '&lt;').replace('>', '&gt;'))
    text = ''.join(cleaned)

    return text.strip()


async def safe_edit(msg, text: str, **kwargs):
    """Edit message with HTML, fall back to plain text if parse fails."""
    try:
        await msg.edit_text(text[:4090], parse_mode=ParseMode.HTML, **kwargs)
    except Exception:
        plain = re.sub(r'<[^>]+>', '', text)
        try:
            await msg.edit_text(plain[:4090], **kwargs)
        except Exception:
            pass


async def get_suggestions(question: str, answer: str) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Ask Gemini for follow-up questions and relevant source links.
    Returns (followups, links) — either or both can be empty if they don't make sense.
    """
    try:
        m = genai.GenerativeModel(CHAT_MODEL)
        prompt = (
            "Given this Q&A, decide if follow-up questions would be useful.\n"
            "If yes, give 2-3 short follow-up questions (max 55 chars each).\n"
            "If the answer references a specific real website or Wikipedia article, include it as a link.\n"
            "If follow-ups don't make sense (e.g. simple factual answer, casual chat, code output), return nothing.\n\n"
            "Reply in this exact format (omit sections that don't apply):\n"
            "FOLLOWUPS:\n"
            "question one\n"
            "question two\n"
            "LINKS:\n"
            "Label | https://url.com\n\n"
            f"Q: {question[:300]}\n"
            f"A: {answer[:500]}"
        )
        r = await asyncio.to_thread(m.generate_content, prompt)
        text = r.text.strip()

        followups, links = [], []

        if "FOLLOWUPS:" in text:
            fq_block = text.split("FOLLOWUPS:")[1].split("LINKS:")[0].strip()
            followups = [l.strip() for l in fq_block.splitlines() if l.strip() and len(l.strip()) <= 64][:3]

        if "LINKS:" in text:
            link_block = text.split("LINKS:")[1].strip()
            for line in link_block.splitlines():
                if "|" in line and "http" in line:
                    parts = line.split("|", 1)
                    label = parts[0].strip()
                    url   = parts[1].strip()
                    if label and url.startswith("http"):
                        links.append((label, url))
            links = links[:2]

        return followups, links
    except Exception:
        return [], []


def make_keyboard(followups: list[str], links: list[tuple[str, str]]) -> InlineKeyboardMarkup | None:
    buttons = []
    for q in followups:
        buttons.append([InlineKeyboardButton(q, callback_data=f"fq:{q}")])
    if links:
        row = [InlineKeyboardButton(f"🔗 {name}", url=url) for name, url in links]
        buttons.append(row)
    return InlineKeyboardMarkup(buttons) if buttons else None


def _make_run_id(filename: str) -> str:
    return hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:12]


def _run_cmd_for(fname: str, ext: str) -> str:
    return {
        "py": f"python3 {fname}", "js": f"node {fname}",
        "ts": f"npx ts-node {fname}", "sh": f"bash {fname}",
        "rb": f"ruby {fname}", "go": f"go run {fname}",
    }.get(ext, f"bash {fname}")


def is_group(update: Update) -> bool:
    return update.effective_chat.type in ("group", "supergroup")


def should_respond(update: Update) -> bool:
    msg = update.effective_message
    if not msg:
        return False
    # Reply to bot's message
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if msg.reply_to_message.from_user.username and \
           msg.reply_to_message.from_user.username.lower() == (BOT_USERNAME or "").lower():
            return True
    text = (msg.text or msg.caption or "").lower()
    if BOT_USERNAME and f"@{BOT_USERNAME.lower()}" in text:
        return True
    if BOT_NAME in text:
        return True
    return False


def record_message(chat_id: int, username: str, text: str):
    if chat_id not in _group_msgs:
        _group_msgs[chat_id] = deque(maxlen=MSG_BUFFER_SIZE)
    _group_msgs[chat_id].append((username or "user", text[:200]))


# ── Message handlers ──────────────────────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg  = update.effective_message
    user = update.effective_user
    cid  = update.effective_chat.id
    text = (msg.text or "").strip()

    if not text or not user or user.is_bot:
        return

    username = user.username or user.first_name or "user"

    # If there's an interactive process waiting for input in this chat, feed it
    session = _interactive.get(cid)
    if session and session.proc.returncode is None:
        session.feed(text)
        return

    # Always record for style learning
    clean_text = re.sub(rf'@?{BOT_NAME}\b', '', text, flags=re.IGNORECASE).strip()
    if clean_text:
        record_message(cid, username, clean_text)

    # Groups: only respond when addressed
    if is_group(update) and not should_respond(update):
        return

    # Strip bot mention/name for the query
    query = re.sub(rf'@{re.escape(BOT_USERNAME or "")}\b', '', text, flags=re.IGNORECASE)
    query = re.sub(rf'\belysium\b', '', query, flags=re.IGNORECASE).strip()
    if not query:
        query = text

    # Include context about any file this message is replying to
    extra = ""
    if msg.reply_to_message:
        rt = msg.reply_to_message
        if rt.document:
            fname = rt.document.file_name or "file"
            fpath = WORKSPACE / fname
            if fpath.exists():
                extra = f"\n[User is replying to a file: {fname} — it's in the workspace]"
        elif rt.text:
            extra = f"\n[User is replying to: {rt.text[:200]}]"

    # Refresh style every 50 messages
    buf = _group_msgs.get(cid)
    if buf and len(buf) % 50 == 0:
        refresh_session(cid)

    thinking = await msg.reply_text("...")

    async with _get_lock(cid):
        try:
            full_q = f"[{username}]: {query}{extra}"
            reply  = await gemini_respond(cid, full_q, update)
            if reply:
                followups, links = await get_suggestions(query, reply)
                kb = make_keyboard(followups, links)
                await safe_edit(thinking, reply, reply_markup=kb)
            else:
                await thinking.delete()
        except Exception as e:
            logger.exception("handle_message error")
        await thinking.edit_text(f"Error: {e}")


async def handle_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg   = update.effective_message
    user  = update.effective_user
    cid   = update.effective_chat.id
    doc   = msg.document
    photo = msg.photo

    if not doc and not photo:
        return

    if doc:
        tg_file = await doc.get_file()
        fname   = doc.file_name or f"file_{doc.file_id}"
    else:
        tg_file = await photo[-1].get_file()
        fname   = f"photo_{photo[-1].file_id}.jpg"

    dest = WORKSPACE / fname
    await tg_file.download_to_drive(str(dest))
    size = dest.stat().st_size
    ext  = dest.suffix.lower().lstrip(".")

    logger.info(f"Saved file: {fname} ({size} bytes)")

    # If there's a caption that addresses the bot — treat it as a query about the file
    caption = (msg.caption or "").strip()
    if caption and (BOT_NAME in caption.lower() or
                    (BOT_USERNAME and f"@{BOT_USERNAME.lower()}" in caption.lower())):
        username = (user.username or user.first_name or "user") if user else "user"
        query    = re.sub(rf'@?{BOT_NAME}\b', '', caption, flags=re.IGNORECASE)
        query    = re.sub(rf'@{re.escape(BOT_USERNAME or "")}\b', '', query, flags=re.IGNORECASE).strip()
        thinking = await msg.reply_text("...")
        try:
            full_q = f"[{username}]: {query or 'help with this file'}\n[File saved to workspace: {fname}]"
            reply  = await gemini_respond(cid, full_q, update)
            if reply:
                await safe_edit(thinking, reply)
            else:
                await thinking.delete()
        except Exception as e:
            await thinking.edit_text(f"Error: {e}")
        return

    # Script files — show run button silently
    run_langs = {"py", "sh", "js", "ts", "rb", "go"}
    if ext in run_langs:
        run_cmd = _run_cmd_for(fname, ext)
        rid     = _make_run_id(fname)
        _pending_runs[rid] = (fname, run_cmd)
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"▶  Run  ({ext})", callback_data=f"run:{rid}")
        ]])
        await msg.reply_text(f"Saved {fname}", reply_markup=kb)
    else:
        await msg.reply_text(f"Saved {fname}  ({size} bytes)")


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data  = query.data
    cid   = query.message.chat.id

    if data.startswith("fq:"):
        question = data[3:]
        username = query.from_user.username or query.from_user.first_name or "user"
        msg = await query.message.reply_text("...")

        # Build a fake Update-like context so gemini_respond can reply to this message
        class _FakeUpdate:
            effective_message = query.message
            effective_chat    = query.message.chat
            effective_user    = query.from_user

        async with _get_lock(cid):
            try:
                reply = await gemini_respond(cid, f"[{username}]: {question}", _FakeUpdate())
                if reply:
                    followups, links = await get_suggestions(question, reply)
                    kb = make_keyboard(followups, links)
                    await safe_edit(msg, reply, reply_markup=kb)
                else:
                    await msg.delete()
            except Exception as e:
                await msg.edit_text(f"Error: {e}")
        return

    if data.startswith("out:"):
        # Show full stored output
        oid    = data[4:]
        output = _pending_outputs.get(oid)
        if not output:
            await query.message.reply_text("Output expired.")
            return
        # Send in chunks if needed
        chunks = [output[i:i+4000] for i in range(0, min(len(output), 12000), 4000)]
        for i, chunk in enumerate(chunks):
            await query.message.reply_text(
                f"<pre>{html.escape(chunk)}</pre>",
                parse_mode=ParseMode.HTML
            )
        return

    if data.startswith("run:"):
        rid   = data[4:]
        entry = _pending_runs.get(rid)
        if not entry:
            await query.message.reply_text("Expired — re-upload the script.")
            return
        fname, run_cmd = entry
        fpath = WORKSPACE / fname
        if not fpath.exists():
            await query.message.reply_text(f"File not found: {fname}")
            return

        # Kill any existing interactive session in this chat
        old = _interactive.get(cid)
        if old:
            try: old.proc.kill()
            except Exception: pass
            _interactive.pop(cid, None)

        status = await query.message.reply_text(f"$ {run_cmd}")
        asyncio.create_task(run_interactive(run_cmd, cid, status, ctx.bot))


# ── Boot ──────────────────────────────────────────────────────────────────────

async def post_init(app: Application):
    global BOT_USERNAME
    me = await app.bot.get_me()
    BOT_USERNAME = me.username
    logger.info(f"Elysium online as @{BOT_USERNAME}")
    await app.bot.delete_my_commands()


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Elysium online.\n\n"
        "Mention me or reply to me to chat.\n"
        "I can run code, write scripts, generate images, manage files — just ask.\n\n"
        "/start — this message\n"
        "/clear — wipe my memory for this chat"
    )


async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    _sessions.pop(cid, None)
    _group_msgs.pop(cid, None)
    old = _interactive.get(cid)
    if old:
        try: old.proc.kill()
        except Exception: pass
        _interactive.pop(cid, None)
    await update.message.reply_text("Memory wiped.")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.Document.ALL | filters.PHOTO, handle_file))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))

    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
