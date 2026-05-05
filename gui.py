import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from threading import Thread, Event
from PIL import Image, ImageTk
import cv2
import os
import time
import pickle
import shutil
import requests 
import xml.etree.ElementTree as ET
from cryptography.fernet import Fernet

from capture import capture_faces
from recog import start_recognition
from train import train_model

ENCODINGS_FILE = "encodings.pickle"
DATASET_DIR = "dataset"
ADMIN_FILE = "admins.txt"
SOAP_URL = "http://jpetewebapp/jtesw_ws/jtesw_webservice.asmx"
SETTINGS_FILE = "settings.json"
FERNET_KEY = b'-_xj1UT6MLokiC2A-cd-LDp1Hj_3I06kdNCky09tr_U='  # replace with output of Fernet.generate_key()
# ----------------------------
# Color palette
# ----------------------------
BG         = "#F7F8FA"
CARD       = "#FFFFFF"
ACCENT     = "#2563EB"
ACCENT_HOV = "#1D4ED8"
DANGER     = "#EF4444"
SUCCESS    = "#22C55E"
TEXT_PRI   = "#111827"
TEXT_SEC   = "#6B7280"
BORDER     = "#E5E7EB"
CAM_BG     = "#1F2937"
Warning    = "#D97706"  


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.configure(bg=BG)
        self.root.geometry("1280x800")
        self.root.minsize(1100, 700)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        self.stop_flag = False
        self.imgtk = None
        self._recognition_running = False
        self._capture_running = False
        self._capture_event = Event()   # set when Take Photo is clicked
        self._stop_capture_event = Event()  # set when Stop is clicked during capture

        self._build_fonts()
        self._build_ui()
        self.pc_save_path = None
        self.auto_capture_enabled = False
        self._load_pc_path()
        self.refresh_dataset()
        self.current_user = None
        self.admin_list = []
        self._load_admin()
        self._set_default_permissions()
        self._mount_server()

    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------
    def _build_fonts(self):
        self.font_title  = ("Helvetica Neue", 20, "bold")
        self.font_sub    = ("Helvetica Neue", 11)
        self.font_label  = ("Helvetica Neue", 10, "bold")
        self.font_body   = ("Helvetica Neue", 10)
        self.font_mono   = ("Courier New", 9)
        self.font_btn    = ("Helvetica Neue", 10, "bold")
        self.font_badge  = ("Helvetica Neue", 9, "bold")

    # --------------------------------------------------------
    # UI build
    # --------------------------------------------------------
    def _build_ui(self):
        # ---- Header (fixed, does not scroll) ----
        header = tk.Frame(self.root, bg=CARD, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="Face Recognition System",
                 font=self.font_title, bg=CARD, fg=TEXT_PRI).pack(side="left", padx=24, pady=16)
        tk.Label(header, text="Smart AI Identification",
                 font=self.font_sub, bg=CARD, fg=TEXT_SEC).pack(side="left", padx=0, pady=16)

        self.login_btn = tk.Button(
            header,
            text="Login",
            font=self.font_btn,
            bg=ACCENT,
            fg="white",
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            command=self._login
        )
        self.login_btn.pack(side="right", padx=24)

        self.logout_btn = tk.Button(
            header,
            text="Logout",
            font=self.font_btn,
            bg=DANGER,
            fg="white",
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            command=self._logout,
            state="disabled"
        )
        self.logout_btn.pack(side="right", padx=8)

        # ⚙ Settings icon button
        self.settings_btn = tk.Button(
            header,
            text="⚙",
            font=("Helvetica Neue", 16),
            bg=CARD,
            fg=TEXT_SEC,
            relief="flat",
            padx=8,
            pady=4,
            cursor="hand2",
            command=self._open_settings
        )
        self.settings_btn.pack(side="right", padx=4)

        self.user_label = tk.Label(
            header,
            text="Not logged in",
            font=self.font_label,
            bg=CARD,
            fg=TEXT_SEC
        )
        self.user_label.pack(side="right", padx=8)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", side="top")

        # ---- Scrollable area ----
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, side="top")

        self._scroll_canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        v_scroll = ttk.Scrollbar(outer, orient="vertical",
                                 command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side="right", fill="y")
        self._scroll_canvas.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._scroll_canvas, bg=BG)
        self._inner_id = self._scroll_canvas.create_window(
            (0, 0), window=self._inner, anchor="nw"
        )

        self._inner.bind("<Configure>", self._on_inner_configure)
        self._scroll_canvas.bind("<Configure>", self._on_canvas_configure)

        self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Button-5>", self._on_mousewheel)

        # ---- Body inside inner frame ----
        body = tk.Frame(self._inner, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=16)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=False)

        right = tk.Frame(body, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(16, 0))

        self._build_camera_panel(left)
        self._build_controls(left)
        self._build_right_panel(right)

    def _on_inner_configure(self, event):
        self._scroll_canvas.configure(
            scrollregion=self._scroll_canvas.bbox("all")
        )

    def _on_canvas_configure(self, event):
        self._scroll_canvas.itemconfig(self._inner_id, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self._scroll_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._scroll_canvas.yview_scroll(1, "units")
        else:
            self._scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # --------------------------------------------------------
    # Camera panel
    # --------------------------------------------------------
    def _build_camera_panel(self, parent):
        card = self._card(parent)
        card.pack(fill="both", expand=True, pady=(0, 12))

        self._section_label(card, "Live Camera")

        cam_frame = tk.Frame(card, bg=CAM_BG)
        cam_frame.pack(fill="both", expand=True, padx=16, pady=(4, 16))

        self.canvas = tk.Canvas(cam_frame, bg=CAM_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.create_text(320, 240, text="Camera feed will appear here",
                                fill="#6B7280", font=self.font_sub, tags="placeholder")

        self.canvas.bind("<Configure>", self._on_canvas_resize)

    # --------------------------------------------------------
    # Controls panel
    # --------------------------------------------------------
    def _build_controls(self, parent):
        card = self._card(parent)
        card.pack(fill="x")

        self._section_label(card, "Controls")

        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill="x", padx=16, pady=(4, 16))

        # Name entry row
        name_row = tk.Frame(inner, bg=CARD)
        name_row.pack(fill="x", pady=(0, 12))

        tk.Label(name_row, text="NTID / Username", font=self.font_label,
                 bg=CARD, fg=TEXT_PRI).pack(anchor="w")

        entry_frame = tk.Frame(name_row, bg=BORDER, bd=0)
        entry_frame.pack(fill="x", pady=(4, 0))

        self.name_entry = tk.Entry(
            entry_frame, font=self.font_body,
            bg=CARD, fg=TEXT_PRI, relief="flat",
            insertbackground=TEXT_PRI
        )
        self.name_entry.pack(fill="x", padx=1, pady=1, ipady=8, ipadx=8)

        # ── Button grid ──────────────────────────────────────
        # Layout:
        #   Row 0: [Capture Face]  [Delete User]   ← 2 equal columns
        #   Row 1: [   Train Model             ]   ← spans 2 columns
        #   Row 2: [   Recognize Face          ]   ← spans 2 columns
        #   Row 3: [        Stop               ]   ← spans 2 columns
        # ─────────────────────────────────────────────────────
        btn_grid = tk.Frame(inner, bg=CARD)
        btn_grid.pack(fill="x")
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        # Row 0 — Capture (toggles to Take Photo) | Delete User
        self.capture_btn = self._btn(btn_grid, "Capture Face",
                                     ACCENT, self.start_capture)
        self.capture_btn.grid(row=0, column=0, padx=(0, 4), pady=4, sticky="ew")

        self.delete_btn = self._btn(btn_grid, "Delete User",
                                    DANGER, self.delete_user)
        self.delete_btn.grid(row=0, column=1, pady=4, sticky="ew")

        # Row 1 — Train Model (full width)
        self.train_btn = self._btn(btn_grid, "Train Model",
                                   ACCENT, self.start_train)
        self.train_btn.grid(row=1, column=0, columnspan=2, pady=4, sticky="ew")

        # Row 2 — Recognize Face (full width)
        self.recog_btn = self._btn(btn_grid, "Recognize Face",
                                   ACCENT, self.start_recognition_thread)
        self.recog_btn.grid(row=2, column=0, columnspan=2, pady=4, sticky="ew")

        # Row 3 — Stop (full width)
        self.stop_btn = self._btn(btn_grid, "Stop",
                                  DANGER, self.stop_all, state="disabled")
        self.stop_btn.grid(row=3, column=0, columnspan=2, pady=4, sticky="ew")

        # ----------------------------
        # PC Save Path — read only, shows value from settings
        # ----------------------------
        #settings_frame = tk.Frame(inner, bg=CARD)
        #settings_frame.pack(fill="x", pady=(12, 0))

        #tk.Label(settings_frame, text="PC Save Path (set via ⚙ Settings)",
          #       font=self.font_label, bg=CARD, fg=TEXT_SEC).pack(anchor="w")

        #self.pc_path_entry = tk.Entry(
         #   settings_frame,
         #   font=self.font_body,
          #  bg="#F3F4F6",
          #  fg=TEXT_SEC,
          #  relief="flat",
          #  insertbackground=TEXT_PRI,
           # state="readonly"
       # )
       # self.pc_path_entry.pack(fill="x", pady=(4, 6), ipady=6, ipadx=6)

    # --------------------------------------------------------
    # Right panel: dataset + log
    # --------------------------------------------------------
    def _build_right_panel(self, parent):
        ds_card = self._card(parent)
        ds_card.pack(fill="x", pady=(0, 12))

        self._section_label(ds_card, "Dataset Status")

        self.dataset_frame = tk.Frame(ds_card, bg=CARD)
        self.dataset_frame.pack(fill="x", padx=16, pady=(4, 16))

        log_card = self._card(parent)
        log_card.pack(fill="both", expand=True)

        self._section_label(log_card, "System Log")

        log_inner = tk.Frame(log_card, bg=CARD)
        log_inner.pack(fill="both", expand=True, padx=16, pady=(4, 16))

        self.log_text = tk.Text(
            log_inner, font=self.font_mono,
            bg="#F9FAFB", fg=TEXT_PRI,
            relief="flat", state="disabled",
            wrap="word", height=14,
            bd=0, highlightthickness=1,
            highlightbackground=BORDER
        )
        scroll = ttk.Scrollbar(log_inner, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)

        scroll.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        self.log_text.tag_config("info",    foreground=TEXT_PRI)
        self.log_text.tag_config("success", foreground="#15803D")
        self.log_text.tag_config("error",   foreground=DANGER)
        self.log_text.tag_config("warn",    foreground="#D97706")

    # --------------------------------------------------------
    # Dataset status refresh
    # --------------------------------------------------------
    def refresh_dataset(self):
        for widget in self.dataset_frame.winfo_children():
            widget.destroy()

        trained_users = set()
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                trained_users = set(data.get("names", []))
            except:
                pass

        if not os.path.exists(DATASET_DIR):
            tk.Label(self.dataset_frame, text="No dataset folder found.",
                     font=self.font_body, bg=CARD, fg=TEXT_SEC).pack(anchor="w")
            return

        users = [d for d in os.listdir(DATASET_DIR)
                 if os.path.isdir(os.path.join(DATASET_DIR, d))]

        if not users:
            tk.Label(self.dataset_frame, text="No users in dataset yet.",
                     font=self.font_body, bg=CARD, fg=TEXT_SEC).pack(anchor="w")
            return

        for user in sorted(users):
            user_path = os.path.join(DATASET_DIR, user)
            images = [f for f in os.listdir(user_path)
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            count = len(images)
            is_trained = user in trained_users

            row = tk.Frame(self.dataset_frame, bg=CARD, cursor="hand2")
            row.pack(fill="x", pady=2)
            row.bind("<Button-1>", lambda e, n=user: self._select_user(n))

            tk.Label(row, text=user.capitalize(), font=self.font_label,
                     bg=CARD, fg=TEXT_PRI, cursor="hand2").pack(side="left")
            row.winfo_children()[-1].bind("<Button-1>", lambda e, n=user: self._select_user(n))

            pill = tk.Label(row, text=f"{count} photos",
                            font=self.font_badge, bg="#EFF6FF", fg="#1D4ED8",
                            padx=8, pady=2, relief="flat")
            pill.pack(side="left", padx=8)

            if is_trained:
                badge = tk.Label(row, text="Trained",
                                 font=self.font_badge, bg="#DCFCE7", fg="#15803D",
                                 padx=8, pady=2)
            else:
                badge = tk.Label(row, text="Not trained",
                                 font=self.font_badge, bg="#FEE2E2", fg="#B91C1C",
                                 padx=8, pady=2)
            badge.pack(side="left")

            tk.Frame(self.dataset_frame, bg=BORDER, height=1).pack(fill="x", pady=2)

    def _select_user(self, name):
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, name)

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    def log(self, message, level="info"):
        self.log_text.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{ts}  {message}\n", level)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    # --------------------------------------------------------
    # AD SOAP validation
    # --------------------------------------------------------
    def _validate_ntid_in_ad(self, ntid: str) -> bool:
        soap = f"""<?xml version="1.0" encoding="utf-8"?>
        <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                        xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                        xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
        <soap12:Body>
            <IsUserExistsInAD xmlns="http://jpetewebapp/jtesw_ws/">
            <userName>{ntid}</userName>
            </IsUserExistsInAD>
        </soap12:Body>
        </soap12:Envelope>"""

        headers = {
            "Content-Type": "application/soap+xml; charset=utf-8",
            "SOAPAction": "http://jpetewebapp/jtesw_ws/IsUserExistsInAD"
        }

        try:
            response = requests.post(
                SOAP_URL,
                data=soap.encode("utf-8"),
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return self._parse_ad_response(response.text)

        except requests.exceptions.Timeout:
            self.log("AD request timeout.", "error")
            return False
        except requests.exceptions.RequestException as e:
            self.log(f"AD request failed: {e}", "error")
            return False
        except Exception as e:
            self.log(f"Unexpected AD error: {e}", "error")
            return False

    def _parse_ad_response(self, response: str) -> bool:
        if not response or not response.strip():
            return False
        try:
            root = ET.fromstring(response)
            for elem in root.iter():
                if "ReturnedValue" in elem.tag:
                    return elem.text.strip().lower() == "true"
            return False
        except ET.ParseError:
            self.log("Failed to parse AD response.", "error")
            return False
        except Exception as e:
            self.log(f"AD parsing error: {e}", "error")
            return False

    # --------------------------------------------------------
    # Admin setup
    # --------------------------------------------------------
    def _load_admin(self):
        self.admin_list = []
        if os.path.exists(ADMIN_FILE):
            try:
                with open(ADMIN_FILE, "r") as f:
                    lines = f.read().splitlines()
                self.admin_list = [l.strip() for l in lines if l.strip()]
            except:
                self.admin_list = []

    def _save_admin(self, ntid):
        try:
            if ntid not in self.admin_list:
                self.admin_list.append(ntid)
            with open(ADMIN_FILE, "w") as f:
                f.write("\n".join(self.admin_list))
            self.log(f"Admin registered: {ntid}", "success")
        except Exception as e:
            self.log(f"Failed to save admin: {e}", "error")

    def _remove_admin(self, ntid):
        if ntid == self.current_user:
            messagebox.showerror("Error", "You cannot remove yourself as admin.")
            return False
        if len(self.admin_list) <= 1:
            messagebox.showerror("Error", "Cannot remove the last admin.")
            return False
        try:
            self.admin_list.remove(ntid)
            with open(ADMIN_FILE, "w") as f:
                f.write("\n".join(self.admin_list))
            self.log(f"Admin removed: {ntid}", "warn")
            return True
        except Exception as e:
            self.log(f"Failed to remove admin: {e}", "error")
            return False

    # --------------------------------------------------------
    # Settings popup
    # --------------------------------------------------------
    def _open_settings(self):
        import json
        import subprocess

        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("420x520")
        win.resizable(False, False)
        win.configure(bg=CARD)
        win.grab_set()

        tk.Label(win, text="Settings", font=self.font_label,
                 bg=CARD, fg=TEXT_PRI).pack(pady=(16, 12))

        form = tk.Frame(win, bg=CARD)
        form.pack(fill="x", padx=24)

        # --- NTID field ---
        tk.Label(form, text="NTID / Username", font=self.font_body,
                 bg=CARD, fg=TEXT_SEC).pack(anchor="w")
        username_entry = tk.Entry(form, font=self.font_body, relief="flat",
                                  bg=BG, fg=TEXT_PRI)
        username_entry.pack(fill="x", pady=(2, 10), ipady=6)

        # --- Password field ---
        tk.Label(form, text="Password", font=self.font_body,
                 bg=CARD, fg=TEXT_SEC).pack(anchor="w")
        password_entry = tk.Entry(form, font=self.font_body, relief="flat",
                                  bg=BG, fg=TEXT_PRI, show="*")
        password_entry.pack(fill="x", pady=(2, 10), ipady=6)

        # --- Eye icon (press & hold to show password) ---
        eye_btn = tk.Label(form, text="Show", font=self.font_body,
                        bg=CARD, fg=TEXT_SEC, cursor="hand2")
        eye_btn.place(relx=0.95, rely=0.42, anchor="ne")

        def show_password(event=None):
            password_entry.config(show="")

        def hide_password(event=None):
            password_entry.config(show="*")

        eye_btn.bind("<ButtonPress-1>", show_password)
        eye_btn.bind("<ButtonRelease-1>", hide_password)
        eye_btn.bind("<Leave>", hide_password)

        # --- PC Server Path field ---
        tk.Label(form, text="Save Log Path (e.g. \\\\server\\folder)",
                 font=self.font_body, bg=CARD, fg=TEXT_SEC).pack(anchor="w")
        path_entry = tk.Entry(form, font=self.font_body, relief="flat",
                              bg=BG, fg=TEXT_PRI)
        path_entry.pack(fill="x", pady=(2, 16), ipady=6)

        # --- Auto-capture checkbox ---
        auto_var = tk.BooleanVar(value=self.auto_capture_enabled)
        auto_check = tk.Checkbutton(
            form,
            text="Auto-capture",
            variable=auto_var,
            font=self.font_body,
            bg=CARD,
            fg=TEXT_PRI,
            activebackground=CARD,
            selectcolor=CARD,
            anchor="w"
        )
        auto_check.pack(fill="x", pady=(0, 10))

        # --- Load existing saved values into fields ---
        settings = self._load_settings()
        if settings.get("username"):
            username_entry.insert(0, settings["username"])
        if settings.get("password"):
            try:
                f = Fernet(FERNET_KEY)
                decrypted_pwd = f.decrypt(settings["password"].encode()).decode()
                password_entry.insert(0, decrypted_pwd)
            except Exception:
                password_entry.insert(0, settings["password"])
        if settings.get("pc_save_path"):
            path_entry.insert(0, settings["pc_save_path"])

        # --- Status label inside popup ---
        status_lbl = tk.Label(win, text="", font=self.font_body,
                              bg=CARD, fg=TEXT_SEC)
        status_lbl.pack(pady=(0, 4))

        def save():
            ntid    = username_entry.get().strip()
            pwd     = password_entry.get().strip()
            unc     = path_entry.get().strip()

            if not ntid or not pwd or not unc:
                messagebox.showerror("Error", "Please fill in all three fields.", parent=win)
                return

            status_lbl.config(text="Connecting to server...", fg=Warning)
            win.update()

            linux_path = unc.replace("\\", "/")
            mount_point = "/mnt/pcshare"
            os.makedirs(mount_point, exist_ok=True)

            try:
                subprocess.run(
                    ["sudo", "umount", "-l", mount_point],
                    capture_output=True, text=True, timeout=10
                )
            except Exception:
                pass

            cmd = [
                "sudo", "mount", "-t", "cifs",
                linux_path, mount_point,
                "-o", f"username={ntid},password={pwd},domain=JABIL,vers=3.0,uid=1000,gid=1000"
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            except Exception as e:
                status_lbl.config(text="Mount error.", fg=DANGER)
                messagebox.showerror("Mount Error",
                                     f"Failed to run mount command:\n{e}", parent=win)
                return

            if result.returncode != 0:
                err_msg = result.stderr.strip() if result.stderr.strip() else "Unknown error."
                status_lbl.config(text="Connection failed.", fg=DANGER)
                messagebox.showerror(
                    "Connection Failed",
                    f"Could not connect to server.\n\nError:\n{err_msg}\n\n"
                    f"Please check your NTID, password, and server path.",
                    parent=win
                )
                return

            try:
                ls_result = subprocess.run(
                    ["ls", mount_point],
                    capture_output=True, text=True, timeout=10
                )
                if ls_result.returncode != 0:
                    status_lbl.config(text="Access denied.", fg=DANGER)
                    messagebox.showerror(
                        "Access Denied",
                        f"Mounted but cannot read folder.\n\nError:\n{ls_result.stderr.strip()}",
                        parent=win
                    )
                    return
            except Exception as e:
                status_lbl.config(text="Verification error.", fg=DANGER)
                messagebox.showerror("Error", f"Could not verify access:\n{e}", parent=win)
                return

            try:
                f       = Fernet(FERNET_KEY)
                encrypted = f.encrypt(pwd.encode()).decode()

                settings = self._load_settings()
                settings["username"]     = ntid
                settings["password"]     = encrypted
                settings["pc_save_path"]  = unc
                settings["auto_capture"]  = auto_var.get()

                with open(SETTINGS_FILE, "w") as file:
                    json.dump(settings, file, indent=4)

                self.pc_save_path = mount_point
                self.auto_capture_enabled = auto_var.get()

                self.pc_path_entry.config(state="normal")
                self.pc_path_entry.delete(0, tk.END)
                self.pc_path_entry.insert(0, unc)
                self.pc_path_entry.config(state="readonly")

                self.log(f"Settings saved. Connected to: {unc}", "success")
                status_lbl.config(text="Connected successfully!", fg="#15803D")
                messagebox.showinfo("Success",
                                    f"Connected to server successfully!\n\nPath: {unc}",
                                    parent=win)
                win.destroy()

            except Exception as e:
                self.log(f"Failed to save settings: {e}", "error")
                messagebox.showerror("Error", f"Failed to save settings:\n{e}", parent=win)

        self._btn(win, "Save & Connect", ACCENT, save).pack(pady=(0, 12))

        # ── Admin Management (only visible when admin is logged in) ──
        if self.current_user:
            tk.Frame(win, bg=BORDER, height=1).pack(fill="x", padx=24, pady=(4, 8))

            tk.Label(win, text="Admin Management", font=self.font_label,
                     bg=CARD, fg=TEXT_PRI).pack(anchor="w", padx=24)

            # --- Add Admin row ---
            add_row = tk.Frame(win, bg=CARD)
            add_row.pack(fill="x", padx=24, pady=(6, 4))

            tk.Label(add_row, text="Add Admin NTID:", font=self.font_body,
                     bg=CARD, fg=TEXT_SEC).pack(side="left")

            add_entry = tk.Entry(add_row, font=self.font_body, relief="flat",
                                 bg=BG, fg=TEXT_PRI, width=16)
            add_entry.pack(side="left", padx=(8, 8), ipady=4)

            def add_admin():
                new_ntid = add_entry.get().strip()
                if not new_ntid:
                    messagebox.showerror("Error", "Please enter an NTID.", parent=win)
                    return
                if new_ntid in self.admin_list:
                    messagebox.showinfo("Already Admin", f"'{new_ntid}' is already an admin.", parent=win)
                    return
                # Validate with AD first
                if not self._validate_ntid_in_ad(new_ntid):
                    messagebox.showerror("Invalid NTID", f"'{new_ntid}' is not a valid NTID.", parent=win)
                    return
                self._save_admin(new_ntid)
                add_entry.delete(0, tk.END)
                refresh_admin_list()
                messagebox.showinfo("Success", f"'{new_ntid}' added as admin.", parent=win)

            self._btn(add_row, "Add", SUCCESS, add_admin).pack(side="left")

            # --- Current admin list with Remove buttons ---
            tk.Label(win, text="Current Admins:", font=self.font_body,
                     bg=CARD, fg=TEXT_SEC).pack(anchor="w", padx=24, pady=(8, 2))

            list_frame = tk.Frame(win, bg=CARD)
            list_frame.pack(fill="x", padx=24)

            def refresh_admin_list():
                for w in list_frame.winfo_children():
                    w.destroy()
                for a in self.admin_list:
                    row = tk.Frame(list_frame, bg=CARD)
                    row.pack(fill="x", pady=1)

                    tag = " (you)" if a == self.current_user else ""
                    tk.Label(row, text=f"{a}{tag}", font=self.font_body,
                             bg=CARD, fg=TEXT_PRI).pack(side="left", padx=(0, 8))

                    # Cannot remove yourself or last admin
                    can_remove = (a != self.current_user and len(self.admin_list) > 1)
                    if can_remove:
                        def make_remove(ntid=a):
                            def do_remove():
                                if messagebox.askyesno("Confirm", f"Remove '{ntid}' from admins?", parent=win):
                                    if self._remove_admin(ntid):
                                        refresh_admin_list()
                            return do_remove
                        self._btn(row, "Remove", DANGER, make_remove()).pack(side="right")

            refresh_admin_list()

    # --------------------------------------------------------
    # Mount server on startup
    # --------------------------------------------------------
    def _mount_server(self):
        import subprocess
        settings = self._load_settings()
        unc_path    = settings.get("pc_save_path")
        username    = settings.get("username")
        enc_password = settings.get("password")

        if not unc_path or not username or not enc_password:
            self.log("Server mount skipped: credentials or path not set.", "warn")
            return

        try:
            f = Fernet(FERNET_KEY)
            password = f.decrypt(enc_password.encode()).decode()
        except Exception as e:
            self.log(f"Failed to decrypt password: {e}", "error")
            return

        linux_path = unc_path.replace("\\", "/")
        mount_point = "/mnt/pcshare"
        os.makedirs(mount_point, exist_ok=True)

        try:
            subprocess.run(
                ["sudo", "umount", "-l", mount_point],
                capture_output=True, text=True, timeout=10
            )
        except Exception:
            pass

        cmd = [
            "sudo", "mount", "-t", "cifs",
            linux_path, mount_point,
            "-o", f"username={username},password={password},domain=JABIL,vers=3.0,uid=1000,gid=1000"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log("Server mounted successfully.", "success")
                self.pc_save_path = mount_point
            else:
                self.log(f"Mount failed: {result.stderr}", "error")
                messagebox.showerror("Server Error",
                                     "Failed to connect to server. Running in local mode.")
        except Exception as e:
            self.log(f"Mount error: {e}", "error")
            messagebox.showerror("Server Error",
                                 "Failed to connect to server. Running in local mode.")

    # --------------------------------------------------------
    # Load settings from settings.json
    # --------------------------------------------------------
    def _load_settings(self):
        import json
        if not os.path.exists(SETTINGS_FILE):
            return {}
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}

    # --------------------------------------------------------
    # Load PC path and other settings on startup
    # --------------------------------------------------------
    def _load_pc_path(self):
        import json

        if not os.path.exists(SETTINGS_FILE):
            return

        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)

            self.pc_save_path = data.get("pc_save_path")

            if self.pc_save_path:
                self.pc_path_entry.config(state="normal")
                self.pc_path_entry.delete(0, tk.END)
                self.pc_path_entry.insert(0, self.pc_save_path)
                self.pc_path_entry.config(state="readonly")
                self.log(f"Loaded PC save path: {self.pc_save_path}", "info")

            # Load auto-capture setting
            self.auto_capture_enabled = data.get("auto_capture", False)

        except Exception as e:
            self.log(f"Failed to load settings: {e}", "error")

    # --------------------------------------------------------
    # Permissions
    # --------------------------------------------------------
    def _set_default_permissions(self):
        # Non-admin: capture, train, delete all disabled
        # Recognition always available; stop only enabled when a process runs
        self.capture_btn.config(state="disabled")
        self.train_btn.config(state="disabled")
        self.delete_btn.config(state="disabled")
        self.recog_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _enable_admin_permissions(self):
        # Admin: capture, train, delete enabled; stop still disabled until process runs
        self.capture_btn.config(state="normal")
        self.train_btn.config(state="normal")
        self.delete_btn.config(state="normal")
        self.recog_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log("Admin access granted.", "success")

    # --------------------------------------------------------
    # Login
    # --------------------------------------------------------
    def _login(self):
        ntid = simpledialog.askstring("Login", "Enter NTID:")
        if not ntid:
            return

        ntid = ntid.strip()
        self.log(f"Login attempt: {ntid}", "info")

        if not self._validate_ntid_in_ad(ntid):
            messagebox.showerror("Login Failed", "Invalid NTID.")
            self.log("Login failed: invalid NTID", "error")
            return

        if not self.admin_list:
            # No admin registered yet — first login becomes admin
            self._save_admin(ntid)
            self.current_user = ntid
            self._enable_admin_permissions()
            self.user_label.config(text=f"Logged in as: {self.current_user} (Admin)")
            self.logout_btn.config(state="normal")
            messagebox.showinfo("Admin Registered", f"{ntid} is now the first admin.")
            return

        if ntid in self.admin_list:
            self.current_user = ntid
            self._enable_admin_permissions()
            self.user_label.config(text=f"Logged in as: {self.current_user} (Admin)")
            self.logout_btn.config(state="normal")
            messagebox.showinfo("Login Success", "Admin access granted.")
        else:
            messagebox.showerror("Access Denied", "You are not an admin.")
            self.log("Login denied: not in admin list", "error")

    # --------------------------------------------------------
    # Logout
    # --------------------------------------------------------
    def _logout(self):
        if not self.current_user:
            messagebox.showinfo("Logout", "No user is currently logged in.")
            return

        user = self.current_user
        self.current_user = None
        self.logout_btn.config(state="disabled")
        self._set_default_permissions()
        self.log(f"User '{user}' logged out.", "info")
        messagebox.showinfo("Logout", "Logged out successfully.")
        self.user_label.config(text="Not logged in")

    # --------------------------------------------------------
    # Capture
    # --------------------------------------------------------
    def start_capture(self):
        ntid = self.name_entry.get().strip()
        if not ntid:
            messagebox.showerror("Error", "Please enter NTID.")
            return

        self.log(f"Validating NTID '{ntid}' with AD...", "info")

        if not self._validate_ntid_in_ad(ntid):
            messagebox.showerror("Invalid NTID", f"'{ntid}' is not a valid NTID.")
            self.log(f"NTID '{ntid}' validation failed.", "error")
            return

        self.log(f"NTID '{ntid}' validated successfully.", "success")

        name = ntid
        save_base = self.pc_save_path if self.pc_save_path else DATASET_DIR
        if self._capture_running or self._recognition_running:
            self.log("Another process is already running.", "warn")
            return

        self._capture_running = True
        self._capture_event.clear()
        self._stop_capture_event.clear()

        # ── Toggle capture_btn → "Take Photo" ──
        self.capture_btn.config(
            text="Take Photo",
            bg=SUCCESS,
            activebackground=SUCCESS,
            command=self.trigger_capture
        )
        self.delete_btn.config(state="disabled")
        self.train_btn.config(state="disabled")
        self.recog_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        mode_label = "auto-detect" if self.auto_capture_enabled else "manual"
        self.log(f"Starting capture for '{name}' [{mode_label} mode]...", "info")

        def frame_cb(frame):
            self._show_frame(frame)
            if self._stop_capture_event.is_set():
                return "stop"
            if self._capture_event.is_set():
                self._capture_event.clear()
                return "capture"
            return None

        def thread():
            try:
                captured, total = capture_faces(
                    name, mode="add",
                    save_base=save_base,
                    frame_callback=frame_cb,
                    stop_flag=lambda: self._stop_capture_event.is_set(),
                    auto_capture=self.auto_capture_enabled
                )
                if self.auto_capture_enabled and captured >= 10:
                    self.log(f"Auto-capture complete. {captured} photos saved. Total: {total}", "success")
                    # Show finish popup on main thread, wait for user to close it
                    popup_closed = Event()

                    def show_finish_popup(n=name, c=captured, ev=popup_closed):
                        popup = tk.Toplevel(self.root)
                        popup.title("Capture Complete")
                        popup.geometry("340x160")
                        popup.resizable(False, False)
                        popup.configure(bg=CARD)
                        popup.grab_set()

                        tk.Label(
                            popup,
                            text="✅  Auto-Capture Finished",
                            font=self.font_label,
                            bg=CARD, fg=TEXT_PRI
                        ).pack(pady=(24, 8))

                        tk.Label(
                            popup,
                            text=f"Captured {n.upper()} for {c} photos.",
                            font=self.font_body,
                            bg=CARD, fg=TEXT_SEC
                        ).pack(pady=(0, 20))

                        def on_close():
                            popup.destroy()
                            ev.set()   # unblock background thread

                        self._btn(popup, "Close", ACCENT, on_close).pack()

                        popup.protocol("WM_DELETE_WINDOW", on_close)

                    self.root.after(0, show_finish_popup)
                    popup_closed.wait()   # block thread until user clicks Close

                    # Trigger stop to clear canvas after popup closed
                    self.root.after(0, self.stop_all)

                else:
                    self.log(f"Capture done. {captured} new photos. Total: {total}", "success")
                self.refresh_dataset()
            except Exception as e:
                self.log(f"Capture error: {e}", "error")
            finally:
                self._capture_running = False
                self._capture_event.clear()
                self._stop_capture_event.clear()
                # ── Revert capture_btn → "Capture Face" ──
                self.capture_btn.config(
                    text="Capture Face",
                    bg=ACCENT,
                    activebackground=ACCENT_HOV,
                    command=self.start_capture
                )
                # Restore buttons based on login state
                if self.current_user:
                    self.delete_btn.config(state="normal")
                    self.train_btn.config(state="normal")
                    self.capture_btn.config(state="normal")
                else:
                    self.delete_btn.config(state="disabled")
                    self.train_btn.config(state="disabled")
                    self.capture_btn.config(state="disabled")
                self.recog_btn.config(state="normal")
                self.stop_btn.config(state="disabled")

        Thread(target=thread, daemon=True).start()

    def trigger_capture(self):
        self._capture_event.set()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    def start_train(self):
        if self._capture_running or self._recognition_running:
            self.log("Another process is already running.", "warn")
            return

        self.train_btn.config(state="disabled")
        self.log("Training model (new images only)...", "info")

        def thread():
            try:
                count = train_model(progress_callback=lambda m: self.log(m, "info"))
                if count == 0:
                    self.log("No new images found to train.", "warn")
                else:
                    self.log(f"Training complete. {count} new face(s) added.", "success")
                self.refresh_dataset()
            except Exception as e:
                self.log(f"Training error: {e}", "error")
            finally:
                if self.current_user:
                    self.capture_btn.config(state="normal")
                    self.train_btn.config(state="normal")
                    self.delete_btn.config(state="normal")
                else:
                    self.capture_btn.config(state="disabled")
                    self.train_btn.config(state="disabled")
                    self.delete_btn.config(state="disabled")

        Thread(target=thread, daemon=True).start()

    # --------------------------------------------------------
    # Recognition
    # --------------------------------------------------------
    def start_recognition_thread(self):
        if self._recognition_running or self._capture_running:
            self.log("Another process is already running.", "warn")
            return

        self._recognition_running = True
        self.stop_flag = False
        self.recog_btn.config(state="disabled")
        self.capture_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.log("Starting face recognition...", "info")

        def frame_cb(frame, names):
            self._show_frame(frame)
            if names:
                name, conf = names[0]
                self.log(f"Recognized: {name} ({conf*100:.1f}%)", "success")

        def thread():
            try:
                start_recognition(
                    frame_callback=frame_cb,
                    stop_flag=lambda: self.stop_flag,
                )
            except Exception as e:
                self.log(f"Recognition error: {e}", "error")
            finally:
                self._recognition_running = False
                self.stop_flag = False
                self.recog_btn.config(state="normal")
                if self.current_user:
                    self.capture_btn.config(state="normal")
                else:
                    self.capture_btn.config(state="disabled")
                self.stop_btn.config(state="disabled")
                self.log("Recognition stopped.", "info")

        Thread(target=thread, daemon=True).start()

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------
    def stop_all(self):
        self.stop_flag = True
        self._stop_capture_event.set()
        self.log("Stopping...", "warn")
        # If capture was running, revert capture_btn immediately on stop
        if self._capture_running:
            self.capture_btn.config(
                text="Capture Face",
                bg=ACCENT,
                activebackground=ACCENT_HOV,
                command=self.start_capture
            )
        self.root.after(500, self._clear_canvas)

    # --------------------------------------------------------
    # Delete user
    # --------------------------------------------------------
    def delete_user(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name to delete.")
            return

        user_path = os.path.join(DATASET_DIR, name)
        if not os.path.exists(user_path):
            messagebox.showerror("Error", f"No dataset found for '{name}'.")
            return

        if not messagebox.askyesno("Confirm Delete",
                                   f"Delete all data for '{name}'? This cannot be undone."):
            return

        try:
            shutil.rmtree(user_path)
            self.log(f"Deleted dataset folder for '{name}'.", "warn")
        except Exception as e:
            self.log(f"Failed to delete folder: {e}", "error")
            return

        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)

                filtered = {"encodings": [], "names": [], "trained_files": []}
                for enc, n in zip(data["encodings"], data["names"]):
                    if n != name:
                        filtered["encodings"].append(enc)
                        filtered["names"].append(n)

                filtered["trained_files"] = [
                    tf for tf in data.get("trained_files", [])
                    if not tf.startswith(f"{name}/")
                ]

                with open(ENCODINGS_FILE, "wb") as f:
                    pickle.dump(filtered, f)

                self.log(f"Removed '{name}' from encodings database.", "warn")
            except Exception as e:
                self.log(f"Failed to update encodings: {e}", "error")

        self.name_entry.delete(0, tk.END)
        self.refresh_dataset()

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _on_canvas_resize(self, event):
        self.canvas.coords("placeholder", event.width // 2, event.height // 2)

    def _show_frame(self, frame):
        try:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w < 10 or h < 10:
                w, h = 640, 480
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((w, h), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.delete("placeholder")
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.imgtk = imgtk
        except:
            pass

    def _clear_canvas(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            w, h = 640, 480
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, w, h, fill="#B0B0B0", outline="")
        self.canvas.create_text(w // 2, h // 2, text="Camera stopped",
                                fill="#555555", font=self.font_sub)
        self.imgtk = None

    def _card(self, parent):
        return tk.Frame(parent, bg=CARD, bd=0,
                        highlightthickness=1,
                        highlightbackground=BORDER)

    def _section_label(self, parent, text):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=16, pady=(12, 6))
        tk.Label(row, text=text, font=self.font_label,
                 bg=CARD, fg=TEXT_PRI).pack(side="left")
        tk.Frame(row, bg=BORDER, height=1).pack(side="left", fill="x",
                                                 expand=True, padx=(8, 0))

    def _btn(self, parent, text, color, command, state="normal"):
        btn = tk.Button(
            parent, text=text,
            font=self.font_btn,
            bg=color, fg="white",
            activebackground=ACCENT_HOV,
            activeforeground="white",
            relief="flat", bd=0,
            padx=12, pady=8,
            cursor="hand2",
            state=state,
            command=command
        )
        return btn


# --------------------------------------------------------
# Entry point
# --------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
