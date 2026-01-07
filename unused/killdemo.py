import tkinter as tk
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *

class PreviewWindow(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.accept('escape', self.close_window)
        
    def close_window(self):
        """Close the preview window without exiting the process"""
        if self.win:
            self.graphicsEngine.removeWindow(self.win)

class ShaderEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shader Preview")

        self.preview_app = None
        self.update_id = None

        self.btn_launch = tk.Button(root, text="Launch Panda", command=self.launch_preview, pady=12)
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
        
        self.preview_app = PreviewWindow()
        self.step_preview()

if __name__ == "__main__":
    root = tk.Tk()
    ShaderEditorApp(root)
    root.mainloop()