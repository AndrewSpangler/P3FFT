import tkinter as tk
from tkinter import ttk, filedialog
import tkinter.font as tkfont
import json
import platform
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from pygments import lex
from pygments.lexers import PythonLexer, GLShaderLexer
import theme

def dark_title_bar(window):
    if platform.system() != "Windows": return
    import ctypes as ct
    window.update()
    hwnd = ct.windll.user32.GetParent(window.winfo_id())
    ct.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ct.byref(ct.c_int(2)), 4)

class LexedText(tk.Text):
    def __init__(self, master, lexer, *args, **kwargs):
        self.frame = tk.Frame(master, bg=theme.COLORS["bg_dark"])
        tk.Text.__init__(self, self.frame, *args, undo=True, font=theme.FONTS["code"], 
                         bg=theme.COLORS["bg_dark"], fg=theme.COLORS["fg_main"], 
                         insertbackground=theme.COLORS["caret"], selectbackground=theme.COLORS["selection"], 
                         wrap=tk.NONE, **kwargs)
        self.lexer = lexer
        for token, color in theme.SYNTAX_MAP.items():
            self.tag_configure(str(token), foreground=color)
        
        self.enumerator = tk.Canvas(self.frame, width=45, bg=theme.COLORS["bg_header"], highlightthickness=0)
        self.vscroll = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.yview)
        self.hscroll = ttk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.xview)
        self.configure(yscrollcommand=self.vscroll.set, xscrollcommand=self.hscroll.set)
        
        self.enumerator.grid(row=0, column=0, sticky="ns")
        tk.Text.grid(self, row=0, column=1, sticky="nsew")
        self.vscroll.grid(row=0, column=2, sticky="ns")
        self.hscroll.grid(row=1, column=1, sticky="ew")
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

        self.bind("<KeyRelease>", self.schedule_relex)
        self.bind("<Configure>", self.refresh_numbers)
        self.after(50, self.relex)

    def pack(self, *args, **kwargs): self.frame.pack(*args, **kwargs)
    
    def set_content(self, val):
        self.delete("1.0", tk.END)
        self.insert("1.0", val)
        self.relex()

    def schedule_relex(self, event=None):
        if hasattr(self, "_relex_id"): self.after_cancel(self._relex_id)
        self._relex_id = self.after(150, self.relex)

    def relex(self):
        text = self.get("1.0", "end-1c")
        for tag in self.tag_names():
            if tag not in ("sel", "insert"): self.tag_remove(tag, "1.0", "end")
        pos = "1.0"
        for token, value in lex(text, self.lexer):
            end_pos = self.index(f"{pos}+{len(value)}c")
            self.tag_add(str(token), pos, end_pos)
            pos = end_pos
        self.refresh_numbers()

    def refresh_numbers(self, event=None):
        self.enumerator.delete("all")
        i = self.index("@0,0")
        while True:
            dline = self.dlineinfo(i)
            if dline is None: break
            linenum = str(i).split(".")[0]
            self.enumerator.create_text(5, dline[1], anchor="nw", text=linenum,
                                       font=theme.FONTS["code"], fill=theme.COLORS["line_numbers"])
            i = self.index(f"{i}+1line")

class PreviewWindow(ShowBase):
    def __init__(self, root, vert, frag, setup, logic):
        ShowBase.__init__(self)
        self.root = root
        wp = WindowProperties()
        wp.setTitle("Shader Preview (Press ESC to close)")
        self.win.requestProperties(wp)
        self.accept('escape', self.close_window)
        
        self.disableMouse()
        cm = CardMaker('card')
        cm.setFrame(-1, 1, -1, 1)
        self.card = self.render2d.attachNewNode(cm.generate())
        
        try:
            self.shader = Shader.make(Shader.SL_GLSL, vert, frag)
            self.card.setShader(self.shader)
        except Exception as e: print(f"Shader Error: {e}")

        self.start_time = globalClock.getFrameTime()
        if setup.strip():
            try:
                local_scope = {'self': self}
                exec(setup, globals(), local_scope)
            except Exception as e: print(f"Setup Error: {e}")
        
        self.custom_logic_func = None
        if logic.strip():
            try:
                local_scope = {}
                indented = "\n".join(["    " + l for l in logic.splitlines()])
                exec(f"def custom_update(self, task):\n{indented}", globals(), local_scope)
                self.custom_logic_func = local_scope['custom_update']
            except Exception as e: print(f"Logic Compilation Error: {e}")
            
        self.taskMgr.add(self.update, "update")

    def close_window(self):
        if self.win: self.graphicsEngine.removeWindow(self.win)

    def userExit(self):
        self.root.end_preview()

    def update(self, task):
        if not self.win:
            return False
        self.card.setShaderInput("iTime", globalClock.getFrameTime() - self.start_time)
        w, h = self.win.getXSize(), self.win.getYSize()
        self.card.setShaderInput("iResolution", LVecBase3f(w, h, 1.0))
        
        if self.mouseWatcherNode.hasMouse():
            x, y = self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY()
            px, py = (x + 1) * 0.5 * w, (y + 1) * 0.5 * h
            self.card.setShaderInput("iMouse", LVecBase4f(px, py, 0, 0))
            
        if self.custom_logic_func:
            try: self.custom_logic_func(self, task)
            except Exception as e:
                print(f"Logic Runtime Error: {e}")
                self.custom_logic_func = None
        return task.cont

class EditorWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Panda3D Shader & Logic Lab")
        self.root.geometry("1500x900")
        theme.apply_ttk_theme()
        
        self.preview_app = None
        self.update_id = None

        self.main_container = tk.Frame(root, bg=theme.COLORS["bg_dark"])
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.main_container, bg=theme.COLORS["bg_header"], width=200, 
                                highlightbackground=theme.COLORS["border"], highlightthickness=1)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.sidebar.pack_propagate(False)

        self.edit_area = tk.Frame(self.main_container, bg=theme.COLORS["bg_dark"])
        self.edit_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.edit_area.grid_rowconfigure((0, 1), weight=1)
        self.edit_area.grid_columnconfigure((0, 1), weight=1)

        self.setup_text = self.add_section(0, 0, "Setup (Python - runs once)", PythonLexer())
        self.logic_text = self.add_section(1, 0, "Update Logic (Python - runs every frame)", PythonLexer())
        self.vert_text = self.add_section(0, 1, "Vertex Shader (GLSL)", GLShaderLexer())
        self.frag_text = self.add_section(1, 1, "Fragment Shader (GLSL)", GLShaderLexer())

        self.add_sidebar_buttons()
        self.load_demo()

    def add_section(self, r, c, title, lexer):
        f = ttk.LabelFrame(self.edit_area, text=title)
        f.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
        t = LexedText(f, lexer)
        t.pack(fill=tk.BOTH, expand=True)
        return t

    def add_sidebar_buttons(self):
        opts = {"fill": tk.X, "padx": 15, "pady": 8}
        tk.Label(self.sidebar, text="CONTROLS", bg=theme.COLORS["bg_header"], 
                 fg=theme.COLORS["fg_main"], font=theme.FONTS["ui_bold"]).pack(pady=(20, 20))
        tk.Button(self.sidebar, text="Save Project", command=self.save_project, bg=theme.COLORS["bg_hover"], fg="white", relief=tk.FLAT).pack(**opts)
        tk.Button(self.sidebar, text="Load Project", command=self.load_project, bg=theme.COLORS["bg_hover"], fg="white", relief=tk.FLAT).pack(**opts)
        tk.Frame(self.sidebar, height=2, bg=theme.COLORS["border"]).pack(fill=tk.X, pady=30)
        tk.Button(self.sidebar, text="LAUNCH PREVIEW", command=self.launch_preview, bg=theme.COLORS["accent"], fg="white", font=theme.FONTS["ui_button"], relief=tk.FLAT, height=2).pack(**opts)

    def load_demo(self):
        self.setup_text.set_content(self.demo_setup)
        self.logic_text.set_content(self.demo_logic)
        self.vert_text.set_content(self.demo_vert)
        self.frag_text.set_content(self.demo_frag)

    def save_project(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            data = {"setup": self.setup_text.get("1.0", tk.END), "logic": self.logic_text.get("1.0", tk.END),
                    "vert": self.vert_text.get("1.0", tk.END), "frag": self.frag_text.get("1.0", tk.END)}
            with open(path, 'w') as f: json.dump(data, f, indent=4)

    def load_project(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            with open(path, 'r') as f:
                data = json.load(f)
                self.setup_text.set_content(data.get("setup", ""))
                self.logic_text.set_content(data.get("logic", ""))
                self.vert_text.set_content(data.get("vert", ""))
                self.frag_text.set_content(data.get("frag", ""))

    def launch_preview(self):
        if self.preview_app: return
        self.preview_app = PreviewWindow(self, self.vert_text.get("1.0", tk.END), self.frag_text.get("1.0", tk.END),
                                         self.setup_text.get("1.0", tk.END), self.logic_text.get("1.0", tk.END))
        self.step_preview()

    def step_preview(self):
        if self.preview_app and self.preview_app.win:
            self.preview_app.taskMgr.step()
            self.update_id = self.root.after(16, self.step_preview)
        else:
            self.end_preview()

    def end_preview(self):
        if self.update_id:
            self.root.after_cancel(self.update_id)
            self.update_id = None
        if self.preview_app:
            self.preview_app.close_window()
            self.preview_app.destroy()
            self.preview_app = None

    @property
    def demo_setup(self):
        return """# Setup code - runs once before the loop starts
# Initialize variables, load resources, etc.

# Example: Create custom shader variables
self.card.setShaderInput("myColor", LVecBase3f(1.0, 0.5, 0.0))
self.rotation_speed = 1.0
print("Setup complete!")"""


    @property
    def demo_logic(self):
        return """
# Use 'self.card.setShaderInput(name, val)'
import math

current_time = globalClock.getFrameTime()
elapsed = current_time - self.start_time
        
# Update shader uniforms
self.card.setShaderInput("iTime", elapsed)
self.card.setShaderInput("iTimeDelta", globalClock.getDt())
self.card.setShaderInput("iFrame", task.frame)
self.card.setShaderInput("iFrameRate", globalClock.getAverageFrameRate())

sin_val = (math.sin(current_time * 2.0) + 1.0) / 2.0
self.card.setShaderInput('myVariable', sin_val)

""".strip()

    @property
    def demo_vert(self):
        return """#version 430
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
        uv = p3d_MultiTexCoord0;
}"""

    @property
    def demo_frag(self):
        return """#version 430
uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iTime;                 // shader playback time (in seconds)
uniform float     iTimeDelta;            // render time (in seconds)
uniform float     iFrameRate;            // shader frame rate
uniform int       iFrame;                // shader playback frame
uniform float     iChannelTime[4];       // channel playback time (in seconds)
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4      iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
// uniform samplerXX iChannel0..3;          // input channel. XX = 2D/Cube
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float myVariable;

// Input from vertex shader
in vec2 texcoord;

out vec4 fragColor;
void main() {
    vec2 p = gl_FragCoord.xy / iResolution.xy;
        fragColor = vec4(p.x, p.y, myVariable, 1.0);
}"""

def main():
    root = tk.Tk()
    EditorWindow(root)
    dark_title_bar(root)
    root.mainloop()

if __name__ == "__main__":
    main()