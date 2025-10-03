# gui_app_pro_ui_plus.py — Professional UI++ (bcrypt + TOTP ready)
# - Aligns GUI auth with backend changes:
#     • bcrypt verification (auto-upgrade legacy plaintext -> bcrypt)
#     • OTP methods: SMS / Email / Authenticator (TOTP)
# - Keeps your original architecture (ThemeManager, Router, Login/Signup/App)
# - Still includes: settings persistence, menu, status bar, toast, tooltips,
#   sortable/filterable tables, safe_call wrappers, logging.

import os, json, time, traceback
import builtins
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image, ImageTk
import qrcode
from datetime import datetime, timedelta

LOCK_STATE_FILE = "login_lock_state.json"
MAX_ATTEMPTS = 5
LOCK_MINUTES = 5


# ==== Import your backend (must contain bcrypt-aware login helpers & get_totp_secret) ====
import intell as backend
from crypto_utils import encrypt_text, decrypt_text, load_key

# New: crypto for GUI-side checks/upgrade (matches backend changes)
import bcrypt
import pyotp

APP_NAME = "Colony Bank – Admin Portal"
APP_MIN_W, APP_MIN_H = 1200, 760
SETTINGS_FILE = "app_settings.json"

# --- Logging (console + file) ---
LOG_LEVEL = logging.INFO
logger = logging.getLogger("ColonyBankGUI")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    ch = logging.StreamHandler(); ch.setLevel(LOG_LEVEL)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    fh = RotatingFileHandler("gui_app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(ch); logger.addHandler(fh)
logger.info("==== Starting ColonyBank GUI ====")

# =========================================================
#                        UTILITIES
# =========================================================

def safe_call(fn):
    """Decorator to catch/log exceptions from UI callbacks without crashing the app."""
    def _wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            logger.exception("UI callback error")
            messagebox.showerror("Unexpected Error", f"{e}\n\nSee gui_app.log for details.")
    return _wrap

class ToolTip:
    def __init__(self, widget, text: str, delay_ms: int = 500):
        self.widget, self.text, self.delay_ms = widget, text, delay_ms
        self.tip = None; self._id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._unschedule)

    def _schedule(self, _):
        self._id = self.widget.after(self.delay_ms, self._show)

    def _unschedule(self, _):
        if self._id: self.widget.after_cancel(self._id); self._id = None
        self._hide()

    def _show(self):
        if self.tip: return
        try:
            x,y,cx,cy = self.widget.bbox("insert")
        except Exception:
            x=y=cy=0
        x += self.widget.winfo_rootx() + 24; y = self.widget.winfo_rooty() + cy + 24
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True); tw.wm_geometry(f"+{x}+{y}")
        frm = ttk.Frame(tw, padding=(8,6), style="Card.TFrame"); frm.pack()
        ttk.Label(frm, text=self.text, style="Muted.TLabel").pack()

    def _hide(self):
        if self.tip: self.tip.destroy(); self.tip = None

class Toast:
    """Simple transient notification in bottom-right."""
    def __init__(self, root: tk.Tk, text: str, ms: int = 1800):
        tw = tk.Toplevel(root)
        tw.overrideredirect(True)
        tw.attributes("-topmost", True)
        root.update_idletasks()
        x = root.winfo_rootx() + root.winfo_width() - 360
        y = root.winfo_rooty() + root.winfo_height() - 120
        tw.geometry(f"340x60+{x}+{y}")
        frm = ttk.Frame(tw, padding=10, style="Card.TFrame"); frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=text).pack(anchor="w")
        tw.after(ms, tw.destroy)

# Settings persistence -------------------------------------------------------

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Settings corrupted; using defaults")
    return {"dark": False, "geometry": f"{APP_MIN_W}x{APP_MIN_H}+120+60", "last_section": "Accounts"}

def save_settings(data: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save settings: {e}")

def ask_int(title, prompt):
    val = simpledialog.askstring(title, prompt)
    if val is None: return None
    try: return int(val)
    except: messagebox.showerror("Invalid input", "Please enter a valid integer."); return None

# =========================================================
#                        THEME ENGINE
# =========================================================
class ThemeManager:
    def __init__(self, root: tk.Tk, dark_default=False):
        self.root = root
        self.style = ttk.Style(root)
        base = "clam" if "clam" in self.style.theme_names() else self.style.theme_use()
        self.style.theme_use(base)
        try: self.root.call("tk", "scaling", 1.2)
        except tk.TclError: pass
        self.font_base = ("Segoe UI", 10)
        self.font_small = ("Segoe UI", 9)
        self.font_title = ("Segoe UI", 16, "bold")
        self.root.option_add("*Font", self.font_base)
        self.apply(light=not dark_default)

    def apply(self, light: bool):
        self.dark = not light
        if light:
            BG, SUR, PRI, TXT, MUT, BDR, HOV, SEL, STRIPE = (
                "#F5F7FA","#FFFFFF","#2563EB","#0F172A","#475569","#E2E8F0","#1D4ED8","#DBEAFE","#F8FAFC")
        else:
            BG, SUR, PRI, TXT, MUT, BDR, HOV, SEL, STRIPE = (
                "#0B1220","#101827","#4F8CFF","#E5EAF3","#9BA4B5","#1F2A44","#71A6FF","#1B2A4B","#0E1627")
        s = self.style
        s.configure("App.TFrame", background=BG)
        s.configure("Card.TFrame", background=SUR, borderwidth=1, relief="solid")
        s.configure("TLabel", background=SUR, foreground=TXT)
        s.configure("Muted.TLabel", background=SUR, foreground=MUT)
        s.configure("Title.TLabel", background=SUR, foreground=TXT, font=self.font_title)
        s.configure("TEntry", fieldbackground=SUR, background=SUR, foreground=TXT)
        s.configure("TButton", padding=(14,8), background=SUR, foreground=TXT, borderwidth=1)
        s.configure("Accent.TButton", padding=(16,10), background=PRI, foreground="#FFFFFF")
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(18,10), background=SUR, foreground=MUT)
        s.configure("Treeview", background=SUR, fieldbackground=SUR, foreground=TXT, borderwidth=0, rowheight=28)
        s.configure("Treeview.Heading", background=SUR, foreground=MUT, relief="flat", padding=(10,8))
        s.configure("Link.TButton", padding=(2,0), relief="flat", background=SUR, foreground=PRI)
        s.map("TButton", background=[("active", SEL)])
        s.map("Accent.TButton", background=[("active", HOV)])
        s.map("TNotebook.Tab", background=[("selected", SEL)], foreground=[("selected", TXT)])
        self.colors = dict(BG=BG, SUR=SUR, PRI=PRI, TXT=TXT, MUT=MUT, BDR=BDR, HOV=HOV, SEL=SEL, STRIPE=STRIPE)
        self.root.configure(bg=BG)

# =========================================================
#                 DATA VIEWS (sorting + filtering)
# =========================================================
class SortableTree(ttk.Treeview):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._sort_reverse = {}
        for col in self["columns"]:
            self.heading(col, command=lambda c=col: self._sort_by(c))

    def _sort_by(self, col):
        items = [(self.set(k, col), k) for k in self.get_children("")]
        def cast(v):
            try: return float(v)
            except: return v
        items.sort(key=lambda t: cast(t[0]), reverse=self._sort_reverse.get(col, False))
        for i, (_, k) in enumerate(items):
            self.move(k, "", i)
        self._sort_reverse[col] = not self._sort_reverse.get(col, False)

class SearchEntry(ttk.Entry):
    def __init__(self, master, placeholder: str = "Search…", on_change=None, **k):
        super().__init__(master, **k)
        self.placeholder = placeholder
        self._on_change = on_change
        self._ph = True
        self.insert(0, placeholder)
        self.configure(foreground="#9BA4B5")
        self.bind("<FocusIn>", self._clear_ph)
        self.bind("<FocusOut>", self._set_ph)
        self.bind("<KeyRelease>", self._changed)

    def _clear_ph(self, _):
        if self._ph:
            self.delete(0, "end"); self._ph = False
            self.configure(foreground="")

    def _set_ph(self, _):
        if not self.get():
            self._ph = True
            self.configure(foreground="#9BA4B5")
            self.insert(0, self.placeholder)

    def _changed(self, _):
        if self._ph: return
        if callable(self._on_change): self._on_change(self.get())

# =========================================================
#            AUTH HELPERS (bcrypt + upgrade + totp)
# =========================================================
def _is_bcrypt_hash(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("$2a$") or s.startswith("$2b$") or s.startswith("$2y$"))

def _rewrite_user_to_bcrypt(username: str, new_bcrypt_hash: str):
    """
    GUI-side update (mirrors backend). Rewrites the user's line in CREDENTIALS_FILE
    to store bcrypt hash (still encrypted per line).
    """
    key = load_key()
    with open(backend.CREDENTIALS_FILE, "r", encoding="utf-8") as fin:
        enc_lines = [ln.rstrip("\n") for ln in fin.readlines()]

    new_lines, replaced = [], False
    for enc in enc_lines:
        if not enc:
            new_lines.append(enc); continue
        try:
            dec = decrypt_text(enc, key)
            parts = [p.strip() for p in dec.split(",")]
            if len(parts) >= 2 and parts[0] == username and not replaced:
                trailing = parts[2:] if len(parts) > 2 else []
                dec_new = ",".join([username, new_bcrypt_hash] + trailing)
                new_lines.append(encrypt_text(dec_new, key))
                replaced = True
            else:
                new_lines.append(enc)
        except Exception:
            new_lines.append(enc)

    if not replaced:
        new_lines.append(encrypt_text(f"{username},{new_bcrypt_hash}", key))

    with open(backend.CREDENTIALS_FILE, "w", encoding="utf-8") as fout:
        for ln in new_lines: fout.write(ln + "\n")

def _verify_password_gui(username: str, password: str) -> bool:
    """
    Verifies username/password against credentials file.
    Supports legacy plaintext and auto-upgrades to bcrypt on success.
    """
    if not os.path.exists(backend.CREDENTIALS_FILE): return False
    key = load_key()
    with open(backend.CREDENTIALS_FILE, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        try:
            dec = decrypt_text(line, key)
            parts = [p.strip() for p in dec.split(",")]
            if len(parts) < 2: continue
            u, secret = parts[0], parts[1]
            if u != username: continue

            if _is_bcrypt_hash(secret):
                ok = bcrypt.checkpw(password.encode(), secret.encode())
                if ok: return True
            else:
                # legacy plaintext
                if password == secret:
                    # upgrade to bcrypt
                    salt = bcrypt.gensalt()
                    new_hash = bcrypt.hashpw(password.encode(), salt).decode()
                    _rewrite_user_to_bcrypt(username, new_hash)
                    return True
        except Exception:
            continue
    return False
def _load_lock_state():
    try:
        if os.path.exists(LOCK_STATE_FILE):
            with open(LOCK_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}  # { username: {"count": int, "until": "ISO8601" or ""} }

def _save_lock_state(state: dict):
    try:
        with open(LOCK_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def _is_locked(username: str):
    state = _load_lock_state()
    rec = state.get(username)
    if not rec: return False, None
    until_s = rec.get("until") or ""
    if not until_s: return False, None
    try:
        until = datetime.fromisoformat(until_s)
        if datetime.now() < until:
            return True, until
        else:
            # lock expired → reset
            rec["count"] = 0
            rec["until"] = ""
            state[username] = rec
            _save_lock_state(state)
            return False, None
    except Exception:
        return False, None

def _note_password_failure(username: str):
    state = _load_lock_state()
    rec = state.get(username, {"count": 0, "until": ""})
    rec["count"] = int(rec.get("count", 0)) + 1
    locked = False
    remaining = max(0, MAX_ATTEMPTS - rec["count"])
    if rec["count"] >= MAX_ATTEMPTS:
        rec["until"] = (datetime.now() + timedelta(minutes=LOCK_MINUTES)).isoformat(timespec="seconds")
        locked = True
        remaining = 0
    state[username] = rec
    _save_lock_state(state)
    return locked, remaining, rec.get("until")

def _reset_password_failures(username: str):
    state = _load_lock_state()
    if username in state:
        state[username] = {"count": 0, "until": ""}
        _save_lock_state(state)

# =========================================================
#            SEPARATE LOGIN & SIGNUP PAGES
# =========================================================
class LoginScreen(ttk.Frame):
    def __init__(self, master, router, theme: ThemeManager):
        super().__init__(master, style="App.TFrame")
        self.router, self.theme = router, theme
        self.otp_expected = None
        self.otp_method = tk.StringVar(value="sms")  # "sms" | "email" | "totp"

        center = ttk.Frame(self, style="App.TFrame"); center.pack(fill="both", expand=True)
        card = ttk.Frame(center, padding=24, style="Card.TFrame")
        card.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(card, text=APP_NAME, style="Title.TLabel").grid(row=0, column=0, columnspan=3, pady=(0,6))
        ttk.Label(card, text="Administrator sign-in with 2-step verification", style="Muted.TLabel").grid(row=1, column=0, columnspan=3, pady=(0,16))

        r = 2
        ttk.Label(card, text="Username").grid(row=r, column=0, sticky="e", padx=6, pady=6)
        self.user_e = ttk.Entry(card, width=28); self.user_e.grid(row=r, column=1, columnspan=2, sticky="w", pady=6)
        r += 1
        ttk.Label(card, text="Password").grid(row=r, column=0, sticky="e", padx=6, pady=6)
        self.pass_e = ttk.Entry(card, width=28, show="*"); self.pass_e.grid(row=r, column=1, columnspan=2, sticky="w", pady=6)

        r += 1
        ttk.Label(card, text="OTP Method").grid(row=r, column=0, sticky="e", padx=6, pady=6)
        m = ttk.Frame(card, style="Card.TFrame"); m.grid(row=r, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(m, text="SMS", value="sms", variable=self.otp_method).pack(side="left")
        ttk.Radiobutton(m, text="Email", value="email", variable=self.otp_method).pack(side="left", padx=(12,0))
        ttk.Radiobutton(m, text="Authenticator (TOTP)", value="totp", variable=self.otp_method).pack(side="left", padx=(12,0))

        r += 1
        self.dest_label = ttk.Label(card, text="Phone (+E.164)")
        self.dest_label.grid(row=r, column=0, sticky="e", padx=6, pady=6)
        self.dest_e = ttk.Entry(card, width=28); self.dest_e.grid(row=r, column=1, columnspan=2, sticky="w", pady=6)

        r += 1
        btns = ttk.Frame(card, style="Card.TFrame"); btns.grid(row=r, column=0, columnspan=3, pady=(10,4))
        so = ttk.Button(btns, text="Send OTP / Prepare TOTP", command=self.send_otp_or_prepare_totp); so.pack(side="left", padx=6)
        ttk.Button(btns, text="Verify & Login", style="Accent.TButton", command=self.verify_and_login).pack(side="left", padx=6)
        ToolTip(so, "Validate credentials then send OTP (SMS/Email) or prepare TOTP")

        r += 1
        ttk.Label(card, text="Enter OTP / TOTP",).grid(row=r, column=0, sticky="e", padx=6, pady=(6,6))
        self.otp_e = ttk.Entry(card, width=16); self.otp_e.grid(row=r, column=1, sticky="w", pady=(6,6))
        # --- QR preview area (hidden until TOTP is prepared) ---
        r += 1
        self.qr_frame = ttk.Frame(card, style="Card.TFrame")
        self.qr_frame.grid(row=r, column=0, columnspan=3, pady=(10, 4))
        self.qr_title = ttk.Label(self.qr_frame, text="", style="Muted.TLabel")
        self.qr_title.pack()
        self.qr_label = ttk.Label(self.qr_frame)   # will hold the QR image
        self.qr_label.pack(pady=6)
        self.qr_hint = ttk.Label(self.qr_frame, text="", style="Muted.TLabel")
        self.qr_hint.pack()
        self.qr_frame.grid_remove()  # start hidden
        self._qr_photo = None        # keep a reference so the image isn't GC'd


        # Footer: go to signup
        r += 1
        footer = ttk.Frame(card, style="Card.TFrame"); footer.grid(row=r, column=0, columnspan=3, pady=(14,0))
        ttk.Label(footer, text="New admin?", style="Muted.TLabel").pack(side="left")
        ttk.Button(footer, text="Create an account", style="Link.TButton", command=self.router.to_signup).pack(side="left", padx=(6,0))

        self.otp_method.trace_add("write", lambda *_: self._update_dest_label())
        self._update_dest_label()

        # Keyboard UX
        self.user_e.focus_set()
        self.bind_all("<Return>", lambda e: self.verify_and_login())
        self.otp_e.bind("<Escape>", lambda e: self.otp_e.delete(0, "end"))

    def _update_dest_label(self):
        method = self.otp_method.get()
        if method == "sms":
            self.dest_label.config(text="Phone (+E.164)")
            self.dest_e.delete(0, "end"); self.dest_e.insert(0, "+91")
            self.dest_e.configure(state="normal")
            if hasattr(self, "qr_frame"): self.qr_frame.grid_remove()

        elif method == "email":
            self.dest_label.config(text="Email")
            self.dest_e.delete(0, "end")
            self.dest_e.configure(state="normal")
            if hasattr(self, "qr_frame"): self.qr_frame.grid_remove()

        else:
            self.dest_label.config(text="(No destination needed for TOTP)")
            self.dest_e.delete(0, "end")
            self.dest_e.configure(state="disabled")

    # --- Send OTP or prepare TOTP using backend helpers ---
    @safe_call
    def send_otp_or_prepare_totp(self):
        user = self.user_e.get().strip(); pwd = self.pass_e.get().strip()
        if not user or not pwd:
            messagebox.showwarning("Login", "Enter username and password first."); return
        if not os.path.exists(backend.CREDENTIALS_FILE):
            messagebox.showwarning("Login", "No credentials file found. Sign up first."); return
        # 1) Check if user is currently locked
        locked, until = _is_locked(user)
        if locked:
            messagebox.showerror("Login Locked", f"Too many failed attempts. Try again after {until.strftime('%H:%M:%S')}.")
            return
        # bcrypt-aware verification (auto-upgrades legacy)

        if not _verify_password_gui(user, pwd):
            locked_now, remaining, until_iso = _note_password_failure(user)
            if locked_now:
                messagebox.showerror("Login Locked", f"Too many failed attempts. Locked for {LOCK_MINUTES} minutes.")
            else:
                messagebox.showerror("Invalid Credentials", f"Wrong username or password. Attempts left: {remaining}")
            return
         # 3) Password OK → reset failure counter for this user
        _reset_password_failures(user)

        method = self.otp_method.get()

        if method == "totp":
            # ensure TOTP secret exists; show user instructions
            try:
                secret = backend.get_totp_secret(user)  # backend helper you added
                uri = f"otpauth://totp/ColonyBank:{user}?secret={secret}&issuer=ColonyBank"
                qr_img = qrcode.make(uri)
                os.makedirs("totp_secrets", exist_ok=True)
                qr_path = os.path.join("totp_secrets", f"{user}_qr.png")
                qr_img.save(qr_path)
                disp_img = qr_img.resize((220, 220))
                self._qr_photo = ImageTk.PhotoImage(disp_img)
                self.qr_label.configure(image=self._qr_photo)
                self.qr_title.configure(text=f"Scan QR for {user}")
                self.qr_hint.configure(text=f"Or add key manually: {secret}")
                self.qr_frame.grid()
                # Sentinel to indicate we're in TOTP path; no random OTP expected
                self.otp_expected = "__TOTP__"
                Toast(self, "TOTP ready. Scan the QR and enter the 6-digit code.")
            except Exception as e:
                logger.exception("TOTP prepare failed")
                messagebox.showerror("TOTP", f"Failed to prepare TOTP: {e}")
                self.otp_expected = None
            return

        # Random OTP for SMS/Email
        import random
        self.otp_expected = str(random.randint(100000, 999999))
        dest = self.dest_e.get().strip()

        if method == "sms":
            if not dest.startswith("+"):
                messagebox.showerror("SMS OTP", "Use E.164 format, e.g., +91XXXXXXXXXX"); self.otp_expected = None; return
            try:
                backend.send_sms_otp(dest, self.otp_expected); Toast(self, "OTP sent via SMS")
            except Exception as e:
                logger.exception("SMS OTP send failed"); messagebox.showerror("SMS OTP", f"Failed to send SMS: {e}"); self.otp_expected = None
        else:  # email
            if "@" not in dest:
                messagebox.showerror("Email OTP", "Enter a valid email address."); self.otp_expected = None; return
            try:
                backend.send_email_otp(dest, self.otp_expected); Toast(self, "OTP sent via Email")
            except Exception as e:
                logger.exception("Email OTP send failed"); messagebox.showerror("Email OTP", f"Failed to send email: {e}"); self.otp_expected = None

    @safe_call
    def verify_and_login(self):
        user = self.user_e.get().strip(); pwd = self.pass_e.get().strip()
        if not user or not pwd:
            messagebox.showwarning("Login", "Enter username and password first."); return

        # Re-verify password to avoid TOCTOU
        if not _verify_password_gui(user, pwd):
            messagebox.showerror("Login", "Invalid username or password."); return

        method = self.otp_method.get()
        code = self.otp_e.get().strip()

        if method == "totp":
            try:
                secret = backend.get_totp_secret(user)
                totp = pyotp.TOTP(secret)
                if totp.verify(code, valid_window=1):
                    Toast(self, "TOTP verified. Welcome!")
                    self.router.to_app()
                else:
                    messagebox.showerror("TOTP", "Invalid TOTP code.")
            except Exception as e:
                logger.exception("TOTP verify failed")
                messagebox.showerror("TOTP", f"Verification failed: {e}")
            return

        # SMS/Email path
        if not self.otp_expected:
            messagebox.showwarning("OTP", "Click 'Send OTP' first."); return
        if code == self.otp_expected:
            Toast(self, "OTP verified. Welcome!")
            self.router.to_app()
        else:
            messagebox.showerror("OTP", "Invalid OTP.")

class SignupScreen(ttk.Frame):
    def __init__(self, master, router, theme: ThemeManager):
        super().__init__(master, style="App.TFrame")
        self.router, self.theme = router, theme
        self.show_pw = tk.BooleanVar(value=False)

        center = ttk.Frame(self, style="App.TFrame"); center.pack(fill="both", expand=True)
        card = ttk.Frame(center, padding=24, style="Card.TFrame")
        card.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(card, text="Create Admin Account", style="Title.TLabel").grid(row=0, column=0, columnspan=3, pady=(0,8))
        ttk.Label(card, text="Strong passwords recommended: 8+ chars with upper/lower/digit/symbol.", style="Muted.TLabel").grid(row=1, column=0, columnspan=3, pady=(0,14))

        r = 2
        ttk.Label(card, text="Username").grid(row=r, column=0, sticky="e", padx=6, pady=6)
        self.user_e = ttk.Entry(card, width=28); self.user_e.grid(row=r, column=1, columnspan=2, sticky="w", pady=6)
        r += 1
        ttk.Label(card, text="Password").grid(row=r, column=0, sticky="e", padx=6, pady=6)
        self.pass_e = ttk.Entry(card, width=28, show="*"); self.pass_e.grid(row=r, column=1, sticky="w", pady=6)
        chk = ttk.Checkbutton(card, text="Show", variable=self.show_pw, command=self._toggle_pw); chk.grid(row=r, column=2, sticky="w")
        ToolTip(chk, "Toggle password visibility")
        r += 1
        ttk.Label(card, text="Confirm Password").grid(row=r, column=0, sticky="e", padx=6, pady=6)
        self.conf_e = ttk.Entry(card, width=28, show="*"); self.conf_e.grid(row=r, column=1, columnspan=2, sticky="w", pady=6)

        r += 1
        ttk.Button(card, text="Create Account", style="Accent.TButton", command=self.register_admin).grid(row=r, column=0, columnspan=3, pady=(12,6))
        r += 1
        footer = ttk.Frame(card, style="Card.TFrame"); footer.grid(row=r, column=0, columnspan=3)
        ttk.Label(footer, text="Already have an account?", style="Muted.TLabel").pack(side="left")
        ttk.Button(footer, text="Back to Login", style="Link.TButton", command=self.router.to_login).pack(side="left", padx=(6,0))

        self.user_e.focus_set()
        self.bind_all("<Return>", lambda e: self.register_admin())

    def _toggle_pw(self):
        show = "" if self.show_pw.get() else "*"
        self.pass_e.config(show=show); self.conf_e.config(show=show)

    def _validate(self, user: str, pw: str, cpw: str) -> bool:
        if not user or not pw or not cpw:
            messagebox.showwarning("Sign Up", "All fields are required."); return False
        if pw != cpw:
            messagebox.showerror("Sign Up", "Passwords do not match."); return False
        ok_len = len(pw) >= 8
        has_up = any(c.isupper() for c in pw)
        has_lo = any(c.islower() for c in pw)
        has_di = any(c.isdigit() for c in pw)
        has_sy = any(c in "@$#%&*!?_-" for c in pw)
        if not (ok_len and has_up and has_lo and has_di and has_sy):
            messagebox.showwarning("Weak Password", "Use 8+ chars with upper/lower/digit/symbol.")
            return False
        return True

    @safe_call
    def register_admin(self):
        user = self.user_e.get().strip(); pwd = self.pass_e.get().strip(); cpw = self.conf_e.get().strip()
        if not self._validate(user, pwd, cpw): return

        # Store bcrypt hash (still encrypt the line with your file-level scheme)
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(pwd.encode(), salt).decode()

        key = load_key()
        enc = encrypt_text(f"{user},{hashed}", key)

        os.makedirs(os.path.dirname(backend.CREDENTIALS_FILE) or ".", exist_ok=True)
        if not os.path.exists(backend.CREDENTIALS_FILE): open(backend.CREDENTIALS_FILE, "w").close()
        with open(backend.CREDENTIALS_FILE, "a", encoding="utf-8") as f:
            f.write(enc + "\n")

        Toast(self, "Admin registered. You can now log in.")
        self.router.to_login()

# =========================================================
#                     APP (after login)
# =========================================================
class AppScreen(ttk.Frame):
    def __init__(self, master, on_logout, theme: ThemeManager, last_section: str = "Accounts"):
        super().__init__(master, style="App.TFrame")
        self.on_logout = on_logout; self.theme = theme

        menubar = tk.Menu(self)
        appm = tk.Menu(menubar, tearoff=0)
        appm.add_command(label="Export (Ctrl+E)", command=self._export_shortcut)
        appm.add_separator()
        appm.add_command(label="Logout (Ctrl+L)", command=self.on_logout)
        appm.add_command(label="Quit (Ctrl+Q)", command=self._quit)
        menubar.add_cascade(label="App", menu=appm)

        viewm = tk.Menu(menubar, tearoff=0)
        viewm.add_command(label="Toggle Dark (Ctrl+T)", command=self._toggle_theme)
        menubar.add_cascade(label="View", menu=viewm)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="Shortcuts", command=self._show_shortcuts)
        menubar.add_cascade(label="Help", menu=helpm)
        self.winfo_toplevel().config(menu=menubar)

        top = ttk.Frame(self, padding=(16,12), style="Card.TFrame"); top.pack(fill="x")
        ttk.Label(top, text=APP_NAME, style="Title.TLabel").pack(side="left")
        right = ttk.Frame(top, style="Card.TFrame"); right.pack(side="right")
        self.dark_var = tk.BooleanVar(value=theme.dark)
        dm = ttk.Checkbutton(right, text="Dark mode", variable=self.dark_var, command=self._toggle_theme)
        dm.pack(side="left", padx=(0,10)); ToolTip(dm, "Switch theme")
        lo = ttk.Button(right, text="Logout", command=self.on_logout); lo.pack(side="left")
        ToolTip(lo, "Return to Login screen")

        main = ttk.Frame(self, style="App.TFrame"); main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0); main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(main, width=240, padding=12, style="Card.TFrame")
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(12,6), pady=(12,12))
        ttk.Label(sidebar, text="Navigation").pack(anchor="w", pady=(0,8))
        self.section = tk.StringVar(value=last_section)
        for name in ("Accounts","Transactions","Analytics","Export"):
            ttk.Radiobutton(sidebar, text=name, value=name, variable=self.section, command=self._switch_section).pack(anchor="w", pady=4)

        content = ttk.Frame(main, padding=12, style="App.TFrame")
        content.grid(row=0, column=1, sticky="nsew", padx=(6,12), pady=(12,12))
        content.columnconfigure(0, weight=1); content.rowconfigure(0, weight=1)

        self.frames = {
            "Accounts": AccountsFrame(content, theme=self.theme),
            "Transactions": TransactionsFrame(content, theme=self.theme),
            "Analytics": AnalyticsFrame(content, theme=self.theme),
            "Export": ExportFrame(content, theme=self.theme),
        }
        for f in self.frames.values(): f.grid(row=0, column=0, sticky="nsew")
        self._show(self.section.get())

        self.status = tk.StringVar(value="Ready")
        sb = ttk.Frame(self, style="Card.TFrame"); sb.pack(fill="x")
        ttk.Label(sb, textvariable=self.status, style="Muted.TLabel").pack(side="left", padx=8)

        root = self.winfo_toplevel()
        root.bind("<Control-q>", lambda e: self._quit())
        root.bind("<Control-l>", lambda e: self.on_logout())
        root.bind("<Control-t>", lambda e: self._toggle_theme())
        root.bind("<Control-e>", lambda e: self._export_shortcut())

    def _show_shortcuts(self):
        messagebox.showinfo("Shortcuts",
                            "Ctrl+T Toggle theme\nCtrl+L Logout\nCtrl+Q Quit\nCtrl+E Export\nCtrl+F Focus table search")

    def _toggle_theme(self):
        want_dark = not (not self.dark_var.get()) if isinstance(self.dark_var.get(), bool) else self.dark_var.get()
        self.dark_var.set(want_dark)
        self.theme.apply(light=not want_dark)
        acc = self.frames.get("Accounts")
        if isinstance(acc, AccountsFrame): acc.apply_stripes()
        Toast(self, "Theme updated")

    def _show(self, name):
        for n,f in self.frames.items(): f.grid_remove()
        self.frames[name].grid(); self.section.set(name)
        if hasattr(self, "status"):
            self.status.set(f"Viewing: {name}")

    def _switch_section(self):
        self._show(self.section.get())

    def _export_shortcut(self):
        frame = self.frames.get("Export")
        if frame: frame.trigger_export()

    def _quit(self):
        self.winfo_toplevel().event_generate("<<AppQuit>>")

# ----------------------- Section Frames -----------------------
class AccountsFrame(ttk.Frame):
    def __init__(self, master, theme: ThemeManager):
        super().__init__(master, style="App.TFrame"); self.theme = theme
        top = ttk.Frame(self, padding=(8,8), style="Card.TFrame"); top.pack(fill="x")
        ttk.Button(top, text="Create Account", style="Accent.TButton", command=self._create).pack(side="left", padx=4)
        ttk.Button(top, text="Refresh", command=self._refresh).pack(side="left", padx=4)
        ttk.Button(top, text="View Balance", command=self._balance).pack(side="left", padx=4)
        ttk.Button(top, text="Modify", command=self._modify).pack(side="left", padx=4)
        ttk.Button(top, text="Delete", command=self._delete).pack(side="left", padx=4)
        search = SearchEntry(top, placeholder="Search name/type/acc…", on_change=self._filter, width=32)
        search.pack(side="right"); ToolTip(search, "Type to filter rows. Ctrl+F to focus.")
        self.bind_all("<Control-f>", lambda e: (search.focus_set(), search.icursor("end")))

        table_card = ttk.Frame(self, padding=10, style="Card.TFrame"); table_card.pack(fill="both", expand=True, pady=10)
        cols = ("AccNo","Name","Type","Balance")
        self.tree = SortableTree(table_card, columns=cols, show="headings", height=18)
        for c in cols: self.tree.heading(c, text=c); self.tree.column(c, anchor="center", width=180)
        self.tree.pack(fill="both", expand=True)
        sy = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=sy.set); sy.pack(side="right", fill="y")
        self.tree.bind("<Button-3>", self._context_menu)
        self._cache = []
        self.apply_stripes(); self._refresh()

    def apply_stripes(self):
        self.tree.tag_configure("odd", background=self.theme.colors["STRIPE"])  
        self.tree.tag_configure("even", background="")

    def _load(self):
        accounts = backend.readAccountsCSV()
        rows = [(a.accNo, a.name, a.type, a.deposit) for a in accounts]
        return rows

    def _populate(self, rows):
        self.tree.delete(*self.tree.get_children())
        alt = False
        for r in rows:
            self.tree.insert("", "end", values=r, tags=("odd" if alt else "even",))
            alt = not alt

    @safe_call
    def _refresh(self):
        rows = self._load(); self._cache = rows; self._populate(rows)

    def _filter(self, text: str):
        t = text.lower().strip()
        if not t:
            self._populate(self._cache); return
        def match(r):
            return any(t in str(x).lower() for x in r)
        self._populate([r for r in self._cache if match(r)])

    def _context_menu(self, e):
        iid = self.tree.identify_row(e.y)
        if not iid: return
        self.tree.selection_set(iid)
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="View Balance", command=self._balance)
        menu.add_command(label="Modify", command=self._modify)
        menu.add_command(label="Delete", command=self._delete)
        menu.tk_popup(e.x_root, e.y_root)

    def _create(self):
        create_account_gui(self.tree, self.theme); self._refresh()

    def _balance(self):
        balance_gui()

    def _modify(self):
        modify_gui(self.tree); self._refresh()

    def _delete(self):
        if messagebox.askyesno("Delete", "Are you sure you want to delete this account?"):
            delete_gui(self.tree); self._refresh()

class TransactionsFrame(ttk.Frame):
    def __init__(self, master, theme: ThemeManager):
        super().__init__(master, style="App.TFrame"); self.theme = theme
        box = ttk.Frame(self, padding=12, style="Card.TFrame"); box.pack(pady=10)
        ttk.Button(box, text="Deposit", width=22, command=deposit_gui).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(box, text="Withdraw", width=22, command=withdraw_gui).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(box, text="View All", width=22, command=lambda: view_all_txn_gui(theme=self.theme)).grid(row=1, column=0, padx=6, pady=6)
        ttk.Button(box, text="View by Account", width=22, command=lambda: view_txn_by_acc_gui(theme=self.theme)).grid(row=1, column=1, padx=6, pady=6)

class AnalyticsFrame(ttk.Frame):
    def __init__(self, master, theme: ThemeManager):
        super().__init__(master, style="App.TFrame"); self.theme = theme
        a = ttk.Frame(self, padding=12, style="Card.TFrame"); a.pack(pady=10)
        ttk.Button(a, text="Train Anomaly Model", width=28, command=train_model_gui).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(a, text="View Flagged Transactions", width=28, command=lambda: view_flagged_gui(theme=self.theme)).grid(row=0, column=1, padx=6, pady=6)

class ExportFrame(ttk.Frame):
    def __init__(self, master, theme: ThemeManager):
        super().__init__(master, style="App.TFrame"); self.theme = theme
        ex = ttk.Frame(self, padding=12, style="Card.TFrame"); ex.pack(pady=10)
        ttk.Button(ex, text="Export Decrypted Accounts (Password Protected)", width=48, command=self.trigger_export).grid(row=0, column=0, padx=8, pady=10)
        ttk.Button(ex, text="Export PDF Statement (Monthly / Custom)",width=48,command=self.trigger_export_pdf).grid(row=1, column=0, padx=8, pady=10)


    @safe_call
    def trigger_export(self):
        export_excel_gui()
    
    @safe_call
    def trigger_export_pdf(self):
    # Ask for account number
        acc = ask_int("PDF Statement", "Enter account number:")
        if acc is None:
            return

    # Ask for period: either YYYY-MM or "YYYY-MM-DD to YYYY-MM-DD"
        mode = simpledialog.askstring(
            "PDF Statement",
            "Enter period:\n"
            "- For month: YYYY-MM (e.g., 2025-09)\n"
            "- Or custom: YYYY-MM-DD to YYYY-MM-DD"
        )
        if not mode:
            return
        mode = mode.strip()

        try:
            if "to" in mode:
                start, end = [p.strip() for p in mode.split("to", 1)]
                pdf_path = backend.export_statement_pdf(acc, start, end)
            else:
            # Treat as month (YYYY-MM) or a single day (YYYY-MM-DD)
                pdf_path = backend.export_statement_pdf(acc, mode)

            messagebox.showinfo("Statement", f"Saved:\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Statement", f"Failed: {e}")

# =========================================================
#                  BACKEND-DRIVING FUNCTIONS
# =========================================================

@safe_call
def show_df(df, title="Data", theme: ThemeManager | None = None):
    win = tk.Toplevel(); win.title(title); win.geometry("960x560")
    if theme: win.configure(bg=theme.colors["BG"])
    frm = ttk.Frame(win, padding=10, style="Card.TFrame"); frm.pack(fill="both", expand=True)
    if df is None or df.empty:
        ttk.Label(frm, text="No data.", style="Muted.TLabel").pack(pady=10); return
    cols = list(df.columns)
    tv = SortableTree(frm, columns=cols, show="headings", height=18)
    for c in cols: tv.heading(c, text=c); tv.column(c, width=160, anchor="center")
    for _, row in df.iterrows(): tv.insert("", "end", values=[row[c] for c in cols])
    tv.pack(fill="both", expand=True)
    sy = ttk.Scrollbar(frm, orient="vertical", command=tv.yview)
    tv.configure(yscroll=sy.set); sy.pack(side="right", fill="y")

@safe_call
def create_account_gui(tree, theme: ThemeManager | None = None):
    acc_no = ask_int("New Account", "Enter account number:");
    if acc_no is None: return
    id_path = filedialog.askopenfilename(title="Select ID image (for OCR)", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
    if not id_path: messagebox.showwarning("Create Account", "ID image required."); return
    tmp_acc = backend.Account(); id_text = tmp_acc.extract_name_from_id(id_path)
    name = simpledialog.askstring("Create Account", "Enter account holder's name (must match ID):")
    if not name: return
    import difflib, shutil
    lines = [ln.strip() for ln in id_text.splitlines() if ln.strip()]
    candidates = [ln for ln in lines if any((w and w[0].isupper()) for w in ln.split())]
    best, score = None, 0.0
    for c in candidates:
        s = difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio()
        if s > score: best, score = c, s
    if score < 0.6:
        messagebox.showerror("Name Mismatch", f"No good match on ID.\nBest: '{best}' ({score:.2f})"); return
    messagebox.showinfo("Name Match", f"Matched with: '{best}' ({score:.2f})")
    os.makedirs("aadhaarcards", exist_ok=True)
    aadhaar_save = os.path.join("aadhaarcards", f"{name.replace(' ','_')}_aadhaar.jpg"); shutil.copy(id_path, aadhaar_save)
    pass_path = filedialog.askopenfilename(title="Select passport-size photo", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
    if not pass_path: messagebox.showwarning("Create Account", "Passport photo required."); return
    os.makedirs("passport_size_photos", exist_ok=True)
    pass_save = os.path.join("passport_size_photos", f"{name.replace(' ','_')}_passport.jpg"); shutil.copy(pass_path, pass_save)
    try:
        face_detector, embedder = backend.load_face_models()
        ref_img = backend.cv2.imread(pass_save)
        ref_emb = backend.get_face_embedding(ref_img, face_detector, embedder)
        if ref_emb is None: messagebox.showerror("Face", "No face detected in passport image."); return
        import numpy as np
        cap = backend.cv2.VideoCapture(0, backend.cv2.CAP_DSHOW) if hasattr(backend.cv2, "CAP_DSHOW") else backend.cv2.VideoCapture(0)
        for _ in range(5): cap.read()
        match_found = False
        backend.cv2.namedWindow("Live Face Verification", backend.cv2.WINDOW_NORMAL)
        backend.cv2.moveWindow("Live Face Verification", 60, 60)
        while True:
            ok, frame = cap.read()
            if not ok: break
            emb = backend.get_face_embedding(frame, face_detector, embedder)
            if emb is not None:
                dist = float(np.linalg.norm(ref_emb - emb))
                backend.cv2.putText(frame, f"Distance: {dist:.2f}", (10, 30), backend.cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                if dist < 0.6:
                    match_found = True
                    os.makedirs("admin_images", exist_ok=True)
                    backend.cv2.imwrite(os.path.join("admin_images", f"{name.replace(' ','_')}.jpg"), frame)
                    break
            backend.cv2.imshow("Live Face Verification", frame)
            if (backend.cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
        cap.release(); backend.cv2.destroyAllWindows()
        if not match_found: messagebox.showerror("Face", "Live face does not match passport photo."); return
    except Exception as e:
        messagebox.showerror("Face Error", f"Face verification failed: {e}"); return

    acc_type = (simpledialog.askstring("Account Type", "Enter type: C or S") or "").strip().upper()
    if acc_type not in ("C","S"): messagebox.showerror("Type", "Must be C (Current) or S (Saving)."); return
    deposit = ask_int("Initial Amount", "Enter initial amount (>=500 S, >=1000 C):")
    if deposit is None: return
    if (acc_type == "S" and deposit < 500) or (acc_type == "C" and deposit < 1000): messagebox.showerror("Amount", "Insufficient initial amount."); return

    accounts = backend.readAccountsCSV()
    if any(a.accNo == acc_no for a in accounts): messagebox.showerror("Duplicate", "Account number already exists."); return

    acc = backend.Account(); acc.accNo, acc.name, acc.type, acc.deposit = acc_no, name, acc_type, deposit
    accounts.append(acc); backend.writeAccountsCSV(accounts); backend.log_transaction(acc_no, "create", deposit)
    messagebox.showinfo("Account", "Account created successfully.")

@safe_call
def deposit_gui():
    num = ask_int("Deposit", "Enter account number:");
    if num is None: return
    amt = ask_int("Deposit", "Enter amount to deposit:");
    if amt is None: return
    accounts = backend.readAccountsCSV()
    for acc in accounts:
        if acc.getAccountNo() == num:
            acc.depositAmount(amt); backend.log_transaction(num, "deposit", amt); backend.writeAccountsCSV(accounts)
            messagebox.showinfo("Deposit", "Amount deposited."); return
    messagebox.showerror("Deposit", "Account not found.")

@safe_call
def withdraw_gui():
    num = ask_int("Withdraw", "Enter account number:");
    if num is None: return
    amt = ask_int("Withdraw", "Enter amount to withdraw:");
    if amt is None: return
    accounts = backend.readAccountsCSV()
    for acc in accounts:
        if acc.getAccountNo() == num:
            if acc.withdrawAmount(amt): backend.log_transaction(num, "withdraw", amt); backend.writeAccountsCSV(accounts); messagebox.showinfo("Withdraw", "Amount withdrawn.")
            else: messagebox.showerror("Withdraw", "Insufficient balance.")
            return
    messagebox.showerror("Withdraw", "Account not found.")

@safe_call
def balance_gui():
    num = ask_int("Balance", "Enter account number:");
    if num is None: return
    for acc in backend.readAccountsCSV():
        if acc.getAccountNo() == num:
            messagebox.showinfo("Balance", f"Account: {acc.accNo}\nName: {acc.name}\nType: {acc.type}\nBalance: ₹{acc.deposit}"); return
    messagebox.showerror("Balance", "Account not found.")

@safe_call
def delete_gui(tree):
    num = ask_int("Delete", "Enter account number to delete:");
    if num is None: return
    backend.deleteAccount(num); messagebox.showinfo("Delete", "Account deleted (if it existed).")

@safe_call
def modify_gui(tree):
    num = ask_int("Modify", "Enter account number to modify:");
    if num is None: return
    accounts = backend.readAccountsCSV()
    for acc in accounts:
        if acc.getAccountNo() == num:
            new_name = simpledialog.askstring("Modify", "New name:", initialvalue=acc.name) or acc.name
            new_type = (simpledialog.askstring("Modify", "New type (C/S):", initialvalue=acc.type) or acc.type).upper()
            if new_type not in ("C","S"): messagebox.showerror("Modify", "Type must be C or S."); return
            new_bal = ask_int("Modify", "New balance:");
            if new_bal is None: return
            acc.name, acc.type, acc.deposit = new_name, new_type, new_bal
            backend.writeAccountsCSV(accounts); messagebox.showinfo("Modify", "Account updated."); return
    messagebox.showerror("Modify", "Account not found.")

@safe_call
def view_all_txn_gui(theme: ThemeManager | None = None):
    if not os.path.exists("transactions.csv"): messagebox.showinfo("Transactions", "No transaction history found."); return
    df = pd.read_csv("transactions.csv")
    if df.empty: messagebox.showinfo("Transactions", "No transactions yet."); return
    show_df(df, "All Transactions", theme)

@safe_call
def view_txn_by_acc_gui(theme: ThemeManager | None = None):
    acc = ask_int("Account Transactions", "Enter account number:");
    if acc is None: return
    if not os.path.exists("transactions.csv"): messagebox.showinfo("Transactions", "No transaction history found."); return
    df = pd.read_csv("transactions.csv"); show_df(df[df["AccountNo"] == acc], f"Transactions for {acc}", theme)

@safe_call
def search_by_name_gui():
    name = simpledialog.askstring("Search", "Enter full or partial name:")
    if not name: return
    matches = [a for a in backend.readAccountsCSV() if name.lower() in a.name.lower()]
    if not matches: messagebox.showinfo("Search", "No matching names found."); return
    df = pd.DataFrame([{"AccNo":a.accNo, "Name":a.name, "Type":a.type, "Balance":a.deposit} for a in matches])
    show_df(df, f"Search results for '{name}'")

@safe_call
def train_model_gui(): backend.train_anomaly_model()

@safe_call
def view_flagged_gui(theme: ThemeManager | None = None):
    if not os.path.exists("transactions_flagged.csv"): messagebox.showinfo("Flagged", "Train the model first."); return
    df = pd.read_csv("transactions_flagged.csv")
    if df.empty: messagebox.showinfo("Flagged", "No data."); return
    show_df(df, "Flagged Transactions (anomaly = -1)", theme)

@safe_call
def export_excel_gui():
    pw = simpledialog.askstring("Export Excel", "Set password for the protected Excel file:", show="*")
    if not pw: return
    orig_input = builtins.input
    try:
        builtins.input = lambda prompt="": pw
        backend.exportAccountsToExcel()
        messagebox.showinfo("Export", "Exported: decrypted_accounts_protected.xlsx")
    finally:
        builtins.input = orig_input

# =========================================================
#                         ROUTER + SHELL
# =========================================================
class Router:
    def __init__(self, root: tk.Tk, theme: ThemeManager, last_section: str = "Accounts"):
        self.root, self.theme = root, theme
        self.login = LoginScreen(root, self, theme)
        self.signup = SignupScreen(root, self, theme)
        self.app = None
        self._last_section = last_section
        self.to_login()

    def _clear(self):
        for w in self.root.winfo_children():
            if isinstance(w, (ttk.Frame, tk.Frame)):
                try: w.pack_forget()
                except: pass
                try: w.grid_forget()
                except: pass

    def to_login(self):
        self._clear(); self.login.pack(fill="both", expand=True)

    def to_signup(self):
        self._clear(); self.signup.pack(fill="both", expand=True)

    def to_app(self):
        self._clear()
        if self.app is not None: self.app.destroy()
        self.app = AppScreen(self.root, on_logout=self.to_login, theme=self.theme, last_section=self._last_section)
        self.app.pack(fill="both", expand=True)

class Shell(tk.Tk):
    def __init__(self):
        super().__init__()
        backend.initialize_transaction_log()
        self.title(APP_NAME); self.minsize(APP_MIN_W, APP_MIN_H)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<<AppQuit>>", lambda e: self._on_close())

        st = load_settings()
        try: self.geometry(st.get("geometry", f"{APP_MIN_W}x{APP_MIN_H}+120+60"))
        except: self.geometry(f"{APP_MIN_W}x{APP_MIN_H}+120+60")

        self.theme = ThemeManager(self, dark_default=st.get("dark", False))
        self.router = Router(self, self.theme, last_section=st.get("last_section", "Accounts"))

        self.bind("<Control-q>", lambda e: self._on_close())

    def _on_close(self):
        st = {
            "dark": self.theme.dark,
            "geometry": self.geometry(),
            "last_section": getattr(getattr(self.router, "app", None), "section", tk.StringVar(value="Accounts")).get()
        }
        save_settings(st)
        self.destroy()

@safe_call
def main():
    app = Shell(); app.mainloop()

if __name__ == "__main__":
    main()
