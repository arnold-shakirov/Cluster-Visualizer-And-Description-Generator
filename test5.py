import io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas, Label
import threading
import torch
from PIL import Image, ImageDraw, ImageTk, ImageSequence
import open_clip
import glob
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN
import numpy as np
import os

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)
model = model.to(device)

def animate_gif(label, start):
    """ Function to animate a gif on a label widget, controlled by start flag. """
    gif_path = "cool-fun.gif"
    gif = Image.open(gif_path)
    frames = [ImageTk.PhotoImage(image=img) for img in ImageSequence.Iterator(gif)]
    frame_count = len(frames)
    counter = [0] 

    def update_frame():
        if start:
            frame = frames[counter[0] % frame_count]
            label.config(image=frame)
            counter[0] += 1
            label.after(100, update_frame)
        else:
            label.config(image='')  

    if start:
        update_frame()
    else:
        label.after_cancel(update_frame)  

def select_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        folder_entry.delete(0, tk.END)
        folder_entry.insert(0, folder_selected)

def start_processing():
    folder_path = folder_entry.get()
    if not folder_path:
        messagebox.showerror("Error", "No folder selected. Please select a folder.")
        return
    gif_label.pack()  # Ensure GIF is visible
    animate_gif(gif_label, True)
    threading.Thread(target=lambda: process_images(folder_path), daemon=True).start()

def generate_description(image, preprocess, model):
    im = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(im, generation_type="top_p")
    return open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

#it makes descriptions for each photo but it should not
def process_images(folder_path):
    image_files = glob.glob(os.path.join(folder_path, "*.png")) + \
                  glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                  glob.glob(os.path.join(folder_path, "*.JPEG")) + \
                  glob.glob(os.path.join(folder_path, "*.PNG")) + \
                  glob.glob(os.path.join(folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(folder_path, "*.JPG"))

    if not image_files:
        messagebox.showinfo("Information", "No JPG files found in the folder.")
        animate_gif(gif_label, False)
        gif_label.pack_forget()
        return

    num_files = len(image_files)
    embeddings, tsne_results, cluster_labels, descriptions = [], [], [], []

    for idx, f in enumerate(image_files):
        try:
            im = Image.open(f).convert("RGB")
            description = generate_description(im, preprocess, model)
            im = preprocess(im).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(im)
            embeddings.append(image_features.squeeze(0).cpu().numpy())
            descriptions.append(description)
            progress_var.set((idx + 1) / num_files * 100)
            app.update_idletasks()
        except Exception as e:
            print(f"Error processing file {f}: {e}")

    if embeddings:
        embeddings = np.vstack(embeddings)
        tsne_results = TSNE(n_components=2, perplexity=min(100, len(embeddings) - 1), random_state=42).fit_transform(embeddings)
        clusterer = HDBSCAN(min_samples=1, min_cluster_size=2)
        cluster_labels = clusterer.fit_predict(embeddings)
        display_images_with_clusters(image_files, tsne_results, cluster_labels, descriptions, display_canvas)

    animate_gif(gif_label, False)  # Stop the GIF
    gif_label.pack_forget()  # Hide the GIF label
    progress_var.set(0)  # Reset the progress bar


def save_results():
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if file_path:
        ps = display_canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save(file_path)
        messagebox.showinfo("Save", "File successfully saved!")


def display_images_with_clusters(image_files, tsne_results, cluster_labels, descriptions, display_canvas):
    canvas_size = 800
    pil_image = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
    thumbnail_size = 150

    min_x, max_x = np.min(tsne_results[:, 0]), np.max(tsne_results[:, 0])
    min_y, max_y = np.min(tsne_results[:, 1]), np.max(tsne_results[:, 1])

    if max_x == min_x or max_y == min_y:  # Check to prevent division by zero
        print("Error: TSNE results are degenerate.")
        return

    tsne_x = (tsne_results[:, 0] - min_x) / (max_x - min_x) * (canvas_size - thumbnail_size)
    tsne_y = (tsne_results[:, 1] - min_y) / (max_y - min_y) * (canvas_size - thumbnail_size)

    for label in np.unique(cluster_labels):
        indices = np.where(cluster_labels == label)[0]
        if indices.size > 0:
            representative_idx = indices[0]
            x, y = tsne_x[representative_idx], tsne_y[representative_idx]
            if np.isnan(x) or np.isnan(y):  # Skip if NaN
                continue
            img = Image.open(image_files[representative_idx])
            img.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
            x_pos, y_pos = int(x), int(y)
            pil_image.paste(img, (x_pos, y_pos))
            draw = ImageDraw.Draw(pil_image)
            draw.text((x_pos, y_pos + thumbnail_size), descriptions[representative_idx], fill='black')

    photo = ImageTk.PhotoImage(pil_image)
    display_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    display_canvas.image = photo  

app = tk.Tk()
app.title("Image Clustering")

folder_label = tk.Label(app, text="Choose a folder to analyze:")
folder_label.pack()

folder_entry = tk.Entry(app, width=50)
folder_entry.pack()

browse_button = tk.Button(app, text="Choose the folder", command=select_folder)
browse_button.pack()

process_button = tk.Button(app, text="Process the data", command=start_processing)
process_button.pack()

save_button = tk.Button(app, text="Save Results", command=save_results)
save_button.pack()

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(app, variable=progress_var, maximum=100)
progress_bar.pack()

gif_label = Label(app)
gif_label.pack()

display_canvas = Canvas(app, width=800, height=800)
display_canvas.pack()

app.mainloop()
