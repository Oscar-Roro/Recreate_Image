# analyze_clusters_vs_power.py
import numpy as np
import pandas as pd
import re
from pathlib import Path
import os
import sys
from skimage.measure import label, regionprops
from skimage.exposure import histogram
from skimage.morphology import remove_small_objects as rmv
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Slider
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from tqdm import tqdm
import seaborn as sns 
plt.close("all")

# --- Base directory and filename
base_dir = sys.path[0]
add_dir = "2025_10_27_00" #! write "User/Documents" not "/User/Documents" 
folder = Path(os.path.join(base_dir,add_dir))
print("Folder being used:",folder,"\n")
# --- Some PARAMETERS
CONNECTIVITY = 2
MIN_SIZE_CLUSTER = 9

# --- Load image
rows = []
files = sorted(folder.glob("*.npz"))
for i, npz_path in enumerate(files):
    if i < 8:
        load_data = np.load(npz_path)
        img_data = load_data["images"]  # shape: (N_frames, H, W)
        # --- Extract metadata
        n_frames = img_data.shape[0]
        height = img_data.shape[1]
        width = img_data.shape[2]
        # print(f"{i}: {npz_path}")
        # --- Store in a dict instead of directly appending the array
        rows.append({
            "file_path": npz_path,
            "n_frames": n_frames,
            "height": height,
            "width": width,
            "data": img_data
        })
    else:
        break
print("Loaded images shape:", img_data.shape)
# --- Create DataFrame
df = pd.DataFrame(rows)
# print("Headers:\n",df.head())
print("\n Dataframe shape: ",df.shape) # (nbr_files,nbr_keys)

# --- Useful functions
def count_clusters(img, thr, connectivity_= CONNECTIVITY, min_size_= MIN_SIZE_CLUSTER):
    mask = img > thr
    labeled = rmv(mask, min_size=min_size_)
    labeled = label(mask, connectivity=connectivity_)
    return labeled.max()
def parse_fname(name):
    fname = os.path.basename(name)
    print("fname",fname)
    # Pattern to match filenames like:
    # gn4095_n1_t0p001_gate_DDG_width1p00e-05_2025-10-27_17-38.npz
    pattern = (
        r"gn(\d+)_n(\d+)_t([\d\w\+\-]+)_gate_DDG_width([\d\w\+\-]+)_"  # gn, n, t, width
        r"([\d\-]+)_([\d\-]+)"  # date, time
        r"(?:_(.*))?\.npz$"         # optional extra args
    )
    m = re.match(pattern, fname)
    if m:
        gain = int(m.group(1))
        npics = int(m.group(2))
        texp_s = float(m.group(3).replace("p", "."))
        gate_width_s = float(m.group(4).replace("p", "."))
        date = m.group(5)
        time = m.group(6)
        args = m.group(7)  # optional extra info (can be None)
    else:
        raise ValueError(f"Nombre no reconocido: {name}")
    print("gain",gain," npics",npics," texp_s",texp_s," gate_width_s",gate_width_s)    
    return {
        "file": fname,
        "gain": gain,
        "npics": npics,
        "texp_s": texp_s,
        "gate_width_s": gate_width_s,
        "date": date+"_"+time,
        "args": args or "" # Empty string if None
        } 


# --- Some tests
imgs = df.loc[0, "data"]     # numpy array for that file
plt.figure(num="File 0 open with df.loc[0,'data']")
plt.imshow(imgs[0],origin="lower", cmap="gray")
plt.title(f"{df.loc[0, 'file_path']} (Frame 0)")

df_img = df.loc[0] 
print(df_img)
total_frames = df_img["n_frames"]
print(total_frames) 
print(df_img["data"])
print(df_img["data"].shape)
plt.figure(num="File 0 open with df.loc[0]['n_frames']")
plt.imshow(df_img["data"][0],origin="lower",cmap="gray")


# --- Plotting functions
# :: Get all images related to a single file
def plot_images_in_file(df_img,n=1):
    total_frames = df_img["n_frames"]
    n = min(n,total_frames)
    print(f"Plotting {n} frames\n")
    # --- Plotting
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
    if n == 1: # For non-iterable
        axes = [axes]
    for i in range(n):
        img = df_img["data"][i]
        img_display = axes[i].imshow(img, origin="lower", cmap="gray", vmin=100)
        axes[i].set_title(f"Frame {i}")
        
    fig.colorbar(img_display,ax=axes[0],label="Intensidad (12bits)")
    fig.tight_layout()
    plt.show()

# :: Get image with slider to change threshold value
def plot_image_in_file_histogram(df_img, threshold_val=105,n=1):
    # --- Params
    img = df_img[0]

    # --- Main figure setup
    fig, (ax_img,ax_hist) = plt.subplots(
        1,2,figsize=(12,7),layout="constrained")
    fig.suptitle("plot_image_in_file_histogram")
    # --- Initial image
    thresh_im = img > threshold_val
    im_display = ax_img.imshow(thresh_im,origin="lower",cmap="gray")
    # --- Counting clusters
    counts = rmv(counts, min_size=MIN_SIZE_CLUSTER)
    counts = label(thresh_im, connectivity=CONNECTIVITY)

    ax_img.set_title(f"Thresholded image  (thr={threshold_val}), counts={counts.max()}")
    # --- Initial histogram
    counts_hist =  np.bincount(img.ravel())
    hist_line = ax_hist.bar(range(img.max()+1), counts_hist, width=0.8, align='center',
                 log ="False",edgecolor="black",linewidth=0.1)
    ax_hist.set(xticks=range(img.max()+1), xlim=[95, img.max()+1])
    ax_hist.tick_params(axis='x',rotation=90)
    hist_thresh_line = ax_hist.axvline(
        threshold_val, color="red",linestyle="--",label="Threshold")
    ax_hist.set_title("Histogram of gray values")
    ax_hist.set_xlabel("Gray value (0–4095 for 12-bit image)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend()  
    fig.colorbar(im_display,ax=ax_img, location='right',
                 shrink=0.45,pad=0.01)
    # --- Slider
    ax_slider = plt.axes([0.1, 0.1, 0.35, 0.025])  #(left, bottom, width, height)
    slider = Slider(ax_slider, label="Threshold",valmin=95,valmax=130,valinit=threshold_val,valstep=1)
    # --- Selection of cluster by Patch
    patch_collect = PatchCollection([],alpha = 0.8)
    ax_img.add_collection(patch_collect)
    # --- Updating subplots <!>
    def update(threshold_val):
        # --- Cluster update
        new_thresh_im = img > threshold_val
        # --- Counting clusters update
        new_counts = rmv(new_counts,min_size=MIN_SIZE_CLUSTER)
        new_counts = label(new_thresh_im, connectivity=CONNECTIVITY)
        # --- New XY data for imshow
        im_display.set_data(new_thresh_im)
        ax_img.set_title(f"Threshold image(thr={threshold_val}), counts={new_counts.max()}")
        # --- Patches of clusters update
        new_patches = []
        if new_counts.max() < 100 :
            for region in regionprops(new_counts):
                minr,minc,maxr,maxc = region.bbox
                rect = mpatches.Rectangle(
                    (minc,minr),maxc-minc,maxr-minr,fill=False,linewidth=2)
                new_patches.append(rect)
        patch_collect.set_paths(new_patches)
        # --- Histogram update
        # counts_hist =  np.bincount(img[new_thresh_im].ravel())
        # for hist_l, h in zip(hist_line, counts_hist):
        #     hist_l.set_height(h)
        # ax_hist.set(xticks=range(img[new_thresh_im].max()+1), xlim=[img[new_thresh_im].min(), img[new_thresh_im].max()+1])
        #ax_hist.tick_params(axis='x',rotation=90)
        hist_thresh_line.set_xdata([threshold_val,threshold_val])
        # ax_hist.relim()
        # ax_hist.autoscale_view()
        fig.canvas.draw_idle()
    # --- End updating subplots <!>
    # --- Initialize and Connect the Slider
    update(threshold_val)
    slider.on_changed(update)
    plt.show()
    

# :: Get images for files with different parameters
def plot_images_across_files(df_files,n=1):
    total_frames = df_files["n_frames"]
    n = min(n,total_frames.min()) 
    n_row = n
    n_col = len(df_files) 
    print(f"\nPlotting {n} frames from {n_col} files\n") 
    fig, axes = plt.subplots(n_row,n_col, figsize=(8*n_row, 8*n_col)) 
    print("AXES",axes, "type", type(axes)," shape", axes.shape)
    if n == 1: 
        axes = axes[np.newaxis,:]
        print("AXES:",axes,"type",type(axes),"AXES[0]:",axes[0],"  AXES.shape:",axes.shape)
    else: 
        axes = axes[i,j]
        print(axes.shape) 
        print("AXES[0]:",axes[0]) 
        print("AXES[0][0]:",axes[0,0])
    # create a single norm to be shared across all images
    norm = colors.Normalize(vmin=100,vmax=150)
    for j in range(len(df_files)): 
        print("J:",j) 
        for i in range(n): 
            print("I:",i) 
            img = df_files["data"][j][i] 
            print("IMG",img) 
            img_display = axes[i,j].imshow(img, origin="lower", cmap="gray",norm=norm) 
            axes[i,j].set_title(f"Frame {j}, index file {j}") 
    # fig.subplots_adjust(bottom=0.8)
    cbar_ax = fig.add_axes([0.15,0.1,0.8,0.01]) #(left, bottom, width, height)
    fig.colorbar(img_display,cax=cbar_ax,orientation='horizontal', label="Intensidad (12bits)") 
    fig.tight_layout() 
    plt.show()

# :: Plot multiple histogramsCHATGPT
def plot_images_and_histograms(df_img, n=1, bins=256):
    total_frames = df_img["n_frames"]
    n = min(n, total_frames)
    fig, axes = plt.subplots(n, 2, figsize=(8*n, 4),
            layout="tight",gridspec_kw={'width_ratios': [2, 1]})
    if n == 1:
        axes = axes[np.newaxis,:]
    for i in range(n):
        img = df_img["data"][i]
        # --- Image ---
        im = axes[i, 0].imshow(img,cmap='gray', origin='lower')
        axes[i, 0].set_title(f"Frame {i}")
        fig.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)
        # --- Histogram ---
        counts_hist =  np.bincount(img.ravel())
        axes[i,1].bar(range(img.max()+1), counts_hist, 
                width=0.8, align='center', log ="True",edgecolor="black",linewidth=0.1)
        axes[i,1].set(xticks=range(img.max()+1), xlim=[100, img.max()+1])
        axes[i,1].tick_params(axis='x',rotation=90)
        # axes[i, 1].hist(img.ravel(),
                # log=True, histtype="barstacked",
                # bins=bins, color='gray')
        axes[i, 1].set_xlabel("Gray Value")
        axes[i, 1].set_ylabel("Pixel Count in log")
    
    
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=None)
    fig.suptitle("plot_images_and_histograms "+ f"for {total_frames} frames")
    plt.show()





if False:
    df_file = df.loc[0]
    df_img = df_file
    plot_images_in_file(df_img,n=2)
if False:
    df_couple_of_files = df.loc[0:2]
    print("df_couple_of_files = df.loc[0:2]:\n",df_couple_of_files)
    print("\n df_couple_of_files['data']:\n",df_couple_of_files['data'])
    print("\n df_couple_of_files['data'][0]:\n",df_couple_of_files['data'][0])
    plot_images_across_files(df_couple_of_files)

if True:
    df_file = df.loc[0]
    df_img = df_file["data"]
    # print(df_img)
    plot_image_in_file_histogram(df_img)

if False:
    
    plot_images_and_histograms(df.loc[0], n=1)
    # print("\n",df.loc[0])
    # print("\n",df.loc[0].head)
    # print("\n",parse_fname(df.loc[0]["file_path"]))

