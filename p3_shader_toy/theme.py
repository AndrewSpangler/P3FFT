import tkinter as tk
from tkinter import ttk

COLORS = {
    "bg_dark": "#1e1e1e",
    "bg_header": "#252526",
    "bg_hover": "#2d2d30",
    "fg_main": "#cccccc",
    "accent": "#0e639c",
    "accent_hover": "#1177bb",
    "border": "#3e3e42",
    "line_numbers": "#858585",
    "selection": "#264f78",
    "caret": "#ffffff"
}

SYNTAX_MAP = {
    "Token.Keyword": "#569cd6",
    "Token.Keyword.Type": "#569cd6",
    "Token.Keyword.Namespace": "#569cd6",
    "Token.Keyword.Constant": "#569cd6",
    "Token.Keyword.Declaration": "#569cd6",
    "Token.Name": "#dcdcdc",
    "Token.Name.Builtin": "#4ec9b0",
    "Token.Name.Builtin.Pseudo": "#569cd6",
    "Token.Name.Function": "#dcdcaa",
    "Token.Name.Function.Magic": "#dcdcaa",
    "Token.Name.Class": "#4ec9b0",
    "Token.Name.Decorator": "#dcdcaa",
    "Token.Name.Variable": "#9cdcfe",
    "Token.Literal": "#b5cea8",
    "Token.Literal.String": "#ce9178",
    "Token.Literal.String.Double": "#ce9178",
    "Token.Literal.String.Single": "#ce9178",
    "Token.Literal.String.Doc": "#6a9955",
    "Token.Literal.Number": "#b5cea8",
    "Token.Literal.Number.Integer": "#b5cea8",
    "Token.Literal.Number.Float": "#b5cea8",
    "Token.Operator": "#d4d4d4",
    "Token.Operator.Word": "#569cd6",
    "Token.Punctuation": "#d4d4d4",
    "Token.Comment.Single": "#6a9955",
    "Token.Comment.Double": "#6a9955",
    "Token.Error": "#f44747",
    "Token.Text": "#dcdcdc",
}


FONTS = {
    "code": ("Courier New", 11),
    "ui_bold": ("Arial", 10, "bold"),
    "ui_button": ("Arial", 11, "bold")
}

def apply_ttk_theme():
    style = ttk.Style()
    style.theme_use('clam')
    style.configure(".", background=COLORS["bg_dark"], foreground=COLORS["fg_main"], borderwidth=0)

    for orient in ["Vertical", "Horizontal"]:
        style.configure(f"{orient}.TScrollbar", 
            gripcount=0, background=COLORS["border"], 
            darkcolor=COLORS["bg_dark"], lightcolor=COLORS["bg_dark"],
            troughcolor=COLORS["bg_dark"], bordercolor=COLORS["bg_dark"], 
            arrowcolor=COLORS["fg_main"])
        style.map(f"{orient}.TScrollbar", background=[('active', COLORS["bg_hover"])])

    style.configure("TLabelframe", background=COLORS["bg_dark"], bordercolor=COLORS["border"])
    style.configure("TLabelframe.Label", background=COLORS["bg_dark"], foreground=COLORS["fg_main"])