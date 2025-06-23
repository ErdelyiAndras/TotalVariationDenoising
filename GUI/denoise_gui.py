import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import subprocess
import os

class DenoiseGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Denoising GUI")
        self.state('zoomed')

        self.grid_columnconfigure(0, weight = 3)
        self.grid_columnconfigure(1, weight = 1)
        self.grid_rowconfigure(0, weight = 1)

        self.img_frame = tk.Frame(self, bd = 0, relief = "solid")
        self.img_frame.grid(row = 0, column = 0, sticky = "nsew", padx = 30, pady = 30)
        self.img_frame.grid_rowconfigure(0, weight = 1)
        self.img_frame.grid_columnconfigure(0, weight = 1)

        self.img_label = tk.Label(self.img_frame)
        self.img_label.grid(row = 0, column = 0, sticky = "nsew")
        
        self.exe_path_var = tk.StringVar()
        
        self.exe_label = tk.Label(self.img_frame, text = "Denoising executable path:")
        self.exe_label.grid(row = 1, column = 0, sticky = "w", padx = 10, pady = (10, 0))
        self.browse_btn = tk.Button(self.img_frame, text = "Browse executable", command = self.browse_exe)
        self.browse_btn.grid(row = 1, column = 1, sticky = "ew", padx = 5, pady = (10, 0))

        self.ctrl_frame = tk.Frame(self)
        self.ctrl_frame.grid(row = 0, column = 1, sticky = "nsew", padx = 30, pady = 30)
        for i in range(8):
            self.ctrl_frame.grid_rowconfigure(i, weight = 1)
        self.ctrl_frame.grid_columnconfigure(0, weight = 1)

        self.strength_var = tk.StringVar(value = "0.1")
        self.step_var = tk.StringVar(value = "0.01")
        self.tol_var = tk.StringVar(value = "0.0032")

        tk.Label(self.ctrl_frame, text = "Strength:").grid(row = 0, column = 0, sticky = "w", padx = 5, pady = (10, 2))
        self.strength_entry = tk.Entry(self.ctrl_frame, textvariable = self.strength_var)
        self.strength_entry.grid(row = 0, column = 1, sticky = "ew", padx = 5, pady = (10, 2))

        tk.Label(self.ctrl_frame, text = "Step Size:").grid(row = 1, column = 0, sticky = "w", padx = 5, pady = 2)
        self.step_entry = tk.Entry(self.ctrl_frame, textvariable = self.step_var)
        self.step_entry.grid(row = 1, column = 1, sticky = "ew", padx = 5, pady = 2)

        tk.Label(self.ctrl_frame, text = "Tolerance:").grid(row = 2, column = 0, sticky = "w", padx = 5, pady = 2)
        self.tol_entry = tk.Entry(self.ctrl_frame, textvariable = self.tol_var)
        self.tol_entry.grid(row = 2, column = 1, sticky = "ew", padx = 5, pady = 2)

        self.ctrl_frame.grid_columnconfigure(1, weight = 1)

        self.load_btn = tk.Button(self.ctrl_frame, text = "Load Image", command = self.load_image)
        self.load_btn.grid(row = 4, column = 0, columnspan = 2, pady = (30, 10), sticky = "ew")
        
        self.output_img_path_var = tk.StringVar()
        
        self.output_img_entry = tk.Entry(self.ctrl_frame, textvariable = self.output_img_path_var)
        self.output_img_entry.grid(row = 5, column = 0, columnspan = 2, sticky = "ew", padx = 1, pady = (10, 0))
        placeholder = "Output image path"
        self.output_img_entry.insert(0, placeholder)
        self.output_img_entry.config(fg = 'grey')
        
        def on_entry_click(event):
            if self.output_img_entry.get() == placeholder:
                self.output_img_entry.delete(0, "end")
                self.output_img_entry.config(fg = 'black')

        def on_focusout(event):
            if not self.output_img_entry.get():
                self.output_img_entry.insert(0, placeholder)
                self.output_img_entry.config(fg = 'grey')
                self.output_img_path_var.set("")
            else:
                self.output_img_path_var.set(self.output_img_entry.get())
        
        self.output_img_path_var.set("")
        self.output_img_entry.bind('<FocusIn>', on_entry_click)
        self.output_img_entry.bind('<FocusOut>', on_focusout)
        

        self.denoise_btn = tk.Button(
            self.ctrl_frame, text = "Denoise", command = self.on_denoise, font = ("Arial", 12, "bold")
        )
        self.denoise_btn.grid(row = 6, column = 0, columnspan = 2, pady = (10, 30), sticky = "ew")

        self.image = None
        self.image_path = None

    def browse_exe(self):
        path = filedialog.askopenfilename(filetypes = [("Executable files", "*.exe")])
        if path:
            self.exe_path_var.set(path)
            self.exe_label.config(text = f"Denoising executable path: {path}")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes = [("Image files", "*.png;*.jpg;*.jpeg")])
        if path:
            orig_img = Image.open(path)
            img_to_display = orig_img.copy()
            self.image_path = path

            self.img_frame.update_idletasks()
            frame_width = self.img_frame.winfo_width()
            frame_height = self.img_frame.winfo_height()

            if orig_img.width > frame_width or orig_img.height > frame_height:
                orig_img.thumbnail((frame_width, frame_height), Image.LANCZOS)

            self.image = ImageTk.PhotoImage(orig_img)
            self.img_label.config(image = self.image)

    def on_denoise(self):
        exe_path = self.exe_path_var.get()
        output_img = self.output_img_path_var.get()
        strength = self.strength_var.get()
        step = self.step_var.get()
        tol = self.tol_var.get()
        
        if not exe_path or not os.path.isfile(exe_path):
            messagebox.showerror("Error", "Please provide a valid path to the denoising executable.")
            return
        
        if not self.image:
            messagebox.showerror("Error", "Please load an image before denoising.")
            return
        
        if not output_img:
            messagebox.showerror("Error", "Please provide a valid output image path.")
            return
        
        if not strength or not step or not tol:
            messagebox.showerror("Error", "Please fill in all parameters (strength, step size, tolerance).")
            return
        
        try:
            # Pass the parameters as command-line arguments
            result = subprocess.run(
                [exe_path, self.image_path, output_img, strength, step, tol, "true"],
                capture_output = True, text = True, check = True
            )
            print("Denoising output:\n", result.stdout)
            out_img = Image.open(output_img)
            frame_width = self.img_frame.winfo_width()
            frame_height = self.img_frame.winfo_height()

            if out_img.width > frame_width or out_img.height > frame_height:
                out_img.thumbnail((frame_width, frame_height), Image.LANCZOS)

            self.image = ImageTk.PhotoImage(out_img)
            self.img_label.config(image = self.image)
            # messagebox.showinfo("Success", "Denoising completed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error:", e.stderr)
            messagebox.showerror("Error", f"Failed to run denoising executable:\n{e.stderr}")
