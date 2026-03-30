import os
import re
import sys
import json
import uuid
import asyncio
import logging
import sqlite3
import subprocess
import traceback
import random
import ast
from pathlib import Path
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS 

from telegram import Update, ReactionTypeEmoji, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import TelegramError, BadRequest
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)

# 2026 Unified Google GenAI SDK
from google import genai
from google.genai import types

# ==========================================
# 1. SETUP & LOGGING
# ==========================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
PLUGIN_DIR = Path("plugins")
PLUGIN_DIR.mkdir(exist_ok=True)
DB_FILE = "bot_data.db"

# ==========================================
# 2. DATABASE ENGINE
# ==========================================
db_lock = asyncio.Lock()

async def execute_db_write(query: str, params: tuple = ()):
    async with db_lock:
        def _write():
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute(query, params)
            last_id = c.lastrowid
            conn.commit()
            conn.close()
            return last_id
        return await asyncio.to_thread(_write)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS whitelist (id INTEGER PRIMARY KEY, type TEXT, approved INTEGER)')
    c.execute('CREATE TABLE IF NOT EXISTS api_keys (id INTEGER PRIMARY KEY AUTOINCREMENT, api_key TEXT UNIQUE, is_active INTEGER DEFAULT 1)')
    c.execute('''
        CREATE TABLE IF NOT EXISTS routines (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, name TEXT, 
            type TEXT, schedule_interval INTEGER, payload TEXT, is_active INTEGER DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 3. CORE AI MANAGERS (2026 REFACTORED)
# ==========================================
class BotState:
    def __init__(self):
        self.owner_id = self._load_setting("owner_id")
        self.memory = {} 
        self.memory_enabled = {}
        self.memory_locks = defaultdict(asyncio.Lock) 
        self.thoughts = {} 
        
    def _load_setting(self, key):
        conn = sqlite3.connect(DB_FILE)
        row = conn.cursor().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        conn.close()
        return row[0] if row else None

    async def set_owner(self, user_id):
        self.owner_id = str(user_id)
        await execute_db_write("REPLACE INTO settings (key, value) VALUES (?, ?)", ("owner_id", self.owner_id))

class KeyRotationManager:
    def __init__(self):
        self.last_used_key = None
        self.keys = []
        self.lock = asyncio.Lock()
        self.refresh_keys()

    def refresh_keys(self):
        conn = sqlite3.connect(DB_FILE)
        self.keys = [row[0] for row in conn.cursor().execute("SELECT api_key FROM api_keys WHERE is_active=1").fetchall()]
        conn.close()

    def _get_next_key(self):
        if not self.keys: raise Exception("No API keys found. Use /addkey")
        available = [k for k in self.keys if k != self.last_used_key] if len(self.keys) > 1 else self.keys
        selected = random.choice(available)
        self.last_used_key = selected
        return selected

    async def execute(self, model_id, contents):
        self.refresh_keys()
        attempts = len(self.keys)
        async with self.lock:
            for _ in range(attempts):
                current_key = self._get_next_key()
                client = genai.Client(api_key=current_key)
                try:
                    response = await asyncio.to_thread(client.models.generate_content, model=model_id, contents=contents)
                    await asyncio.sleep(random.uniform(1.5, 3.0))
                    return response
                except Exception as e:
                    if "429" in str(e) or "ResourceExhausted" in str(e): continue
                    raise e
            raise Exception("All API keys rate-limited.")

class AIManager:
    def __init__(self):
        self.models = {
            "pro": "nano-banana-pro-preview", 
            "flash": "gemini-flash-latest", 
            "lite": "gemini-flash-lite-latest"
        }

    async def summarize_context(self, history):
        prompt = f"Summarize this conversation concisely:\n{str(history)}"
        res = await key_manager.execute(self.models["lite"], prompt)
        return [{"role": "user", "parts": [{"text": "Context summary: " + res.text}]}, 
                {"role": "model", "parts": [{"text": "Understood."}]}]

class InternalPlugins:
    @staticmethod
    async def web_scraper(args: dict):
        query, url = args.get("query"), args.get("url")
        def _scrape():
            def fetch_url_text(target_url):
                try:
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                    response = requests.get(target_url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for s in soup(["script", "style"]): s.extract()
                    return soup.get_text(separator=' ', strip=True)[:8000]
                except Exception as e: return f"Error: {e}"
            if url: return {"source": url, "content": fetch_url_text(url)}
            if query:
                with DDGS() as ddgs: results = [r for r in ddgs.text(query, max_results=3)]
                return {"query": query, "results": [{"title": r['title'], "url": r['href'], "content": fetch_url_text(r['href'])[:2000]} for r in results]}
            return {"error": "Missing params"}
        return await asyncio.to_thread(_scrape)

class PluginManager:
    @staticmethod
    async def run_plugin(plugin_name: str, args: dict):
        if plugin_name == "web_scraper": return await InternalPlugins.web_scraper(args)
        plugin_path = PLUGIN_DIR / f"{plugin_name}.py"
        if not plugin_path.exists(): return {"error": "Plugin file not found."}
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(plugin_path), json.dumps(args),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
        return json.loads(stdout.decode()) if stdout else {"error": stderr.decode()}

    @staticmethod
    async def develop_plugin(prompt: str):
        sys_inst = "Output JSON ONLY: {'filename': 'name', 'dependencies': ['pkg'], 'code': '...'}"
        res = await key_manager.execute(ai_manager.models["pro"], sys_inst + "\n" + prompt)
        try:
            raw = re.search(r'(\{.*\})', res.text, re.DOTALL).group().replace("'", '"')
            data = json.loads(raw)
            if data.get("dependencies"):
                await asyncio.to_thread(lambda: [subprocess.run([sys.executable, "-m", "pip", "install", d]) for d in data["dependencies"]])
            with open(PLUGIN_DIR / f"{data['filename']}.py", "w") as f: f.write(data["code"])
            return {"status": "success", "plugin": data['filename']}
        except Exception as e: return {"error": f"Build failed: {e}"}

# Instances
bot_state = BotState()
key_manager = KeyRotationManager()
ai_manager = AIManager()

# ==========================================
# 4. TELEGRAM UTILITIES (HTML EDITION)
# ==========================================
async def send_smart_reply(msg_obj, text: str, reply_markup=None):
    MAX_LEN = 4000
    try:
        if len(text) <= MAX_LEN:
            # SWITCHED TO HTML PARSER
            return await msg_obj.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
        for i in range(0, len(text), MAX_LEN):
            chunk = text[i:i+MAX_LEN]
            markup = reply_markup if i+MAX_LEN >= len(text) else None
            await msg_obj.reply_text(chunk, reply_markup=markup, parse_mode=ParseMode.HTML)
    except BadRequest as e:
        logger.warning(f"HTML Parsing failed: {e}. Falling back to plain text.")
        if len(text) <= MAX_LEN: return await msg_obj.reply_text(text, reply_markup=reply_markup)
        for i in range(0, len(text), MAX_LEN):
            await msg_obj.reply_text(text[i:i+MAX_LEN], reply_markup=reply_markup if i+MAX_LEN >= len(text) else None)

async def execute_routine(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    chat_id, payload, r_type = job.data['chat_id'], job.data['payload'], job.data['type']
    try:
        if r_type == 'simple':
            await context.bot.send_message(chat_id=chat_id, text=payload, parse_mode=ParseMode.HTML)
        else:
            res = await key_manager.execute(ai_manager.models["flash"], f"TASK: {payload}")
            if res.text: await context.bot.send_message(chat_id=chat_id, text=res.text, parse_mode=ParseMode.HTML)
    except: pass

async def load_routines_on_startup(app: Application):
    conn = sqlite3.connect(DB_FILE)
    rows = conn.cursor().execute("SELECT id, chat_id, type, schedule_interval, payload FROM routines WHERE is_active=1").fetchall()
    conn.close()
    for r_id, cid, rtyp, interval, payl in rows:
        app.job_queue.run_repeating(execute_routine, interval=interval, data={'chat_id': cid, 'payload': payl, 'type': rtyp}, name=f"routine_{r_id}")

# ==========================================
# 5. THE ORCHESTRATOR (CORE ENGINE)
# ==========================================
async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not await check_access(update): return
    chat_id, user_id = msg.chat_id, update.effective_user.id
    
    if not key_manager.keys:
        if str(user_id) == bot_state.owner_id: await msg.reply_text("🛑 No API keys. Use /addkey <key>.")
        return

    # Handle Voice
    audio_content = None
    if msg.voice:
        status_msg = await msg.reply_text("🎧 Listening...")
        file = await context.bot.get_file(msg.voice.file_id)
        path = f"v_{msg.voice.file_id}.ogg"
        await file.download_to_drive(path)
        client = genai.Client(api_key=key_manager.keys[0])
        upload = await asyncio.to_thread(client.files.upload, file=path)
        audio_content = types.Content(parts=[types.Part(file_data=types.FileData(file_uri=upload.uri, mime_type=upload.mime_type))])
        os.remove(path)
        await status_msg.delete()

    user_text = msg.text or "[Attachment Processing]"
    current_level = "lite"
    decision, status_msg = None, None
    thinking_enabled = False
    accumulated_thoughts = []

    try:
        await msg.set_reaction(reaction=ReactionTypeEmoji('👀'))
        
        while current_level:
            model_id = ai_manager.models[current_level]
            btn_inst = 'For buttons, append <buttons>[[{"text":"Btn","url":"link"}]]</buttons> to "text".'
            
            # Instructed to use HTML for raw replies
            if current_level == "lite":
                sys_route = f'Output VALID JSON ONLY. Chat: {{"action": "reply", "text": "html formatted text"}}. Scrape: {{"action": "run_plugin", "plugin_name": "web_scraper", "args": {{"query": "search"}}}}. Complex: {{"action": "escalate", "thinking_required": true}}. {btn_inst}'
            else:
                sys_route = f'JSON ONLY. Level: {current_level.upper()}. REQUIRED: "thought_summary" (3 words) and "thought_process". Action: "reply", "run_plugin", or "build_plugin". {btn_inst}'

            parts = [types.Part(text=sys_route + "\nUser: " + user_text)]
            if audio_content: parts.extend(audio_content.parts)
            
            res = await key_manager.execute(model_id, [types.Content(parts=parts)])
            
            # --- GREEDY JSON PARSING ---
            try:
                if not res.text: raise ValueError("Empty output from AI")
                raw_text = res.text.replace("```json", "").replace("```", "").strip()
                match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
                if match:
                    cleaned_json = match.group(1).replace("'", '"')
                    try: decision = json.loads(cleaned_json)
                    except: decision = ast.literal_eval(cleaned_json)
                else: raise ValueError("No JSON found")
            except: 
                decision = {"action": "reply", "text": res.text if res.text else "API blocked response."}

            # --- LIVE THINKING UI (HTML FORMATTED) ---
            if decision.get('thought_process'):
                summary = decision.get('thought_summary', 'Thinking').strip()
                # Used HTML tags instead of markdown for safe parsing
                accumulated_thoughts.append(f"<b>[{current_level.upper()}] - {summary}</b>\n<i>{decision['thought_process']}</i>")
                think_text = f"🧠 <b>Thinking:</b> <i>{summary}...</i>"
                if not status_msg: status_msg = await msg.reply_text(think_text, parse_mode=ParseMode.HTML)
                else:
                    try: await status_msg.edit_text(think_text, parse_mode=ParseMode.HTML)
                    except: pass
                await asyncio.sleep(0.5)

            if decision.get('action') == 'escalate':
                thinking_enabled = decision.get('thinking_required', False)
                current_level = "flash" if current_level == "lite" else "pro" if current_level == "flash" else None
            else: break

        # Execution
        plugin_output = None
        if decision.get('action') in ['run_plugin', 'build_plugin']:
            if not status_msg: status_msg = await msg.reply_text("⚡ Processing...")
            if decision['action'] == 'run_plugin':
                plugin_output = await PluginManager.run_plugin(decision['plugin_name'], decision.get('args', {}))
            else:
                plugin_output = await PluginManager.develop_plugin(decision.get('instructions', ''))
        
        if status_msg:
            if decision.get('action') == 'reply': await status_msg.delete()
            else: await status_msg.edit_text("✅ Background Task Complete.")

        # --- FORMATTING FINAL REPLY ---
        if decision.get('action') == 'reply':
            final_text = decision.get('text', 'No text provided.')
        else:
            # HTML Formatting instructions
            sys_inst = (
                'Format the raw data into a clean, readable response using Telegram HTML tags '
                '(<b>bold</b>, <i>italic</i>, <code>code</code>). Use the • character for bullet points. '
                'DO NOT use markdown like ** or *. '
                'To add buttons, append <buttons>[[{"text":"Btn","url":"link"}]]</buttons> at the end.'
            )
            parts = [types.Part(text=f"SYSTEM: {sys_inst}\nHistory: {bot_state.memory.get(chat_id, [])}\nRaw Data: {plugin_output}\nUser: {user_text}")]
            if audio_content: parts.extend(audio_content.parts)
            
            format_res = await key_manager.execute(ai_manager.models["lite"], [types.Content(parts=parts)])
            final_text = format_res.text if format_res.text else str(plugin_output)
        
        reply_markup = None
        btn_match = re.search(r'<buttons>(.*?)</buttons>', final_text, re.DOTALL)
        if btn_match:
            try:
                kb = [[InlineKeyboardButton(**b) for b in r] for r in json.loads(btn_match.group(1))]
                reply_markup = InlineKeyboardMarkup(kb)
                final_text = final_text.replace(btn_match.group(0), "").strip()
            except: pass

        if accumulated_thoughts:
            t_id = str(uuid.uuid4())[:8]
            bot_state.thoughts[t_id] = "\n\n".join(accumulated_thoughts)
            btn = InlineKeyboardButton("🧠 View Thinking", callback_data=f"thought_{t_id}")
            if reply_markup: 
                new_kb = list(reply_markup.inline_keyboard)
                new_kb.append([btn])
                reply_markup = InlineKeyboardMarkup(new_kb)
            else: reply_markup = InlineKeyboardMarkup([[btn]])

        await send_smart_reply(msg, final_text, reply_markup=reply_markup)
        
        async with bot_state.memory_locks[chat_id]:
            if bot_state.memory_enabled.get(chat_id, True): 
                mem = bot_state.memory.setdefault(chat_id, [])
                mem.append({"role": "user", "parts": [user_text]})

    except Exception as e:
        logger.error(traceback.format_exc())
        await msg.reply_text(f"⚠️ Orchestrator Error: {e}")
    finally:
        try: await msg.set_reaction(reaction=[])
        except: pass

# ==========================================
# 6. COMMANDS & UI
# ==========================================
async def check_access(update: Update) -> bool:
    uid = update.effective_user.id
    if bot_state.owner_id is None: await bot_state.set_owner(uid); return True
    return str(uid) == bot_state.owner_id

async def add_key_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_access(update): return
    if context.args: 
        await key_manager.add_key(context.args[0])
        await update.message.reply_text("✅ API Key Integrated.")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if q.data.startswith("thought_"):
        t_id = q.data.split("_")[-1]
        text = bot_state.thoughts.get(t_id, "Memory Expired.")
        try:
            # Send thoughts using HTML parsing
            await context.bot.send_message(
                chat_id=q.message.chat_id, 
                text=f"🧠 <b>Thinking Log:</b>\n\n{text}", 
                parse_mode=ParseMode.HTML,
                reply_to_message_id=q.message.message_id
            )
        except Exception as e:
            logger.error(f"Failed to send thoughts: {e}")
            await context.bot.send_message(
                chat_id=q.message.chat_id, 
                text=f"🧠 Internal Monologue:\n\n{text}", 
                reply_to_message_id=q.message.message_id
            )
    await q.answer()

def main():
    while True:
        try:
            app = Application.builder().token(TELEGRAM_TOKEN).post_init(load_routines_on_startup).build()
            app.add_handler(CommandHandler("addkey", add_key_command))
            app.add_handler(CallbackQueryHandler(callback_handler))
            app.add_handler(MessageHandler(filters.TEXT | filters.VOICE | filters.Document.ALL, process_message))
            logger.info("Bot Online.")
            app.run_polling(drop_pending_updates=True)
            break
        except Exception as e:
            logger.error(f"Kernel Panic: {e}")
            import time; time.sleep(5)

if __name__ == "__main__":
    main()
