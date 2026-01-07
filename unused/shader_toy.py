import tkinter as tk
from tkinter import messagebox
import traceback
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *

class PreviewWindow(ShowBase):
    def __init__(self, vert_content, frag_content, logic_code):
        ShowBase.__init__(self)
        
        # Hide default close button and handle window close
        wp = WindowProperties()
        wp.setTitle("Shader Preview (Press ESC to close)")
        self.win.requestProperties(wp)
        self.accept('escape', self.close_window)
        
        # Write temp shader files
        with open("temp.vert", "w") as f: f.write(vert_content)
        with open("temp.frag", "w") as f: f.write(frag_content)

        self.disableMouse()
        cm = CardMaker('card')
        cm.setFrame(-1, 1, -1, 1)
        self.card = self.render2d.attachNewNode(cm.generate())
        
        try:
            self.shader = Shader.load(Shader.SL_GLSL, vertex="temp.vert", fragment="temp.frag")
            self.card.setShader(self.shader)
        except Exception as e:
            print(f"Shader Error: {e}")
        
        self.start_time = globalClock.getFrameTime()
        
        # Dynamic compilation of the Logic Pane
        self.custom_logic_func = None
        if logic_code.strip():
            try:
                local_scope = {}
                # Indent user code to sit inside a function definition
                indented_code = "\n".join(["    " + line for line in logic_code.splitlines()])
                exec_code = f"def custom_update(self, task):\n{indented_code}"
                exec(exec_code, globals(), local_scope)
                self.custom_logic_func = local_scope['custom_update']
            except Exception as e:
                print(f"Logic Compilation Error: {e}")

        self.taskMgr.add(self.update, "update")
        self.updateResolution()
    
    def close_window(self):
        """Close the preview window without exiting the process"""
        if self.win:
            self.graphicsEngine.removeWindow(self.win)

    def updateResolution(self):
        if self.win:
            w, h = self.win.getXSize(), self.win.getYSize()
            self.card.setShaderInput("iResolution", LVecBase3f(w, h, 1.0))

    def update(self, task):
        self.card.setShaderInput("iTime", globalClock.getFrameTime() - self.start_time)
        
        if self.mouseWatcherNode.hasMouse():
            x, y = self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY()
            w, h = self.win.getXSize(), self.win.getYSize()
            px, py = (x + 1) * 0.5 * w, (y + 1) * 0.5 * h
            self.card.setShaderInput("iMouse", LVecBase4f(px, py, 0, 0))

        if self.custom_logic_func:
            try:
                self.custom_logic_func(self, task)
            except Exception as e:
                print(f"Logic Runtime Error: {e}")
                self.custom_logic_func = None 

        return task.cont

class ShaderEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Panda3D Shader & Logic Lab")
        self.root.geometry("800x950")
        
        self.preview_app = None
        self.update_id = None

        # Main Vertical Splitter
        self.pane = tk.PanedWindow(root, orient=tk.VERTICAL, sashpad=4, sashrelief=tk.RAISED)
        self.pane.pack(fill=tk.BOTH, expand=1)

        # Reusable Text Widget settings
        text_settings = {
            "wrap": tk.NONE,
            "undo": True,
            "font": ("Courier New", 12),
            "bg": "#1e1e1e",
            "fg": "#dcdcdc",
            "insertbackground": "white" # Caret color
        }

        # 1. Vertex Shader Pane
        self.vert_frame = tk.LabelFrame(self.pane, text="Vertex Shader (GLSL)", padx=5, pady=5)
        self.vert_text = tk.Text(self.vert_frame, height=8, **text_settings)
        self.vert_text.pack(fill=tk.BOTH, expand=1)
        self.pane.add(self.vert_frame)

        # 2. Fragment Shader Pane
        self.frag_frame = tk.LabelFrame(self.pane, text="Fragment Shader (GLSL)", padx=5, pady=5)
        self.frag_text = tk.Text(self.frag_frame, height=12, **text_settings)
        self.frag_text.pack(fill=tk.BOTH, expand=1)
        self.pane.add(self.frag_frame)

        # 3. Logic Pane (Now identical in style)
        self.logic_frame = tk.LabelFrame(self.pane, text="Update Logic (Python)", padx=5, pady=5)
        self.logic_text = tk.Text(self.logic_frame, height=8, **text_settings)
        self.logic_text.pack(fill=tk.BOTH, expand=1)
        self.pane.add(self.logic_frame)

        # Set defaults
        self.vert_text.insert(tk.END, self.default_vert)
        self.frag_text.insert(tk.END, self.default_frag)
        self.logic_text.insert(tk.END, self.default_logic)

        self.btn_launch = tk.Button(root, text="Launch Shader Preview", command=self.launch_preview, 
                                   bg="#2e7d32", fg="white", font=('Arial', 11, 'bold'), pady=12)
        self.btn_launch.pack(fill=tk.X)

    def step_preview(self):
        if self.preview_app and self.preview_app.win:
            self.preview_app.taskMgr.step()
            self.update_id = self.root.after(16, self.step_preview)
        else:
            if self.preview_app:
                self.preview_app.destroy()
                self.preview_app = None
            self.update_id = None

    def launch_preview(self):
        if self.preview_app:
            return
        
        self.preview_app = PreviewWindow(
            self.vert_text.get("1.0", tk.END),
            self.frag_text.get("1.0", tk.END),
            self.logic_text.get("1.0", tk.END)
        )
        self.step_preview()

    @property
    def default_vert(self):
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
    def default_frag(self):
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

    @property
    def default_logic(self):
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

if __name__ == "__main__":
    root = tk.Tk()
    ShaderEditorApp(root)
    root.mainloop()