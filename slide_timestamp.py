import vlc
import tkinter as tk
import csv
import sys
import os
from pathlib import Path

path = Path(sys.argv[1])

player = vlc.MediaPlayer(path / "Talk.mp4")
player.play()

slide_num = 1

root = tk.Tk()

def mark_slide(event=None):
    global slide_num

    ms = player.get_time()
    seconds = ms // 1000

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    timestamp = f"{h:02}:{m:02}:{s:02}"

    with open("timestamps.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, f"slide_{slide_num:03}.png"])

    print(timestamp, slide_num)

    slide_num += 1

root.bind("<space>", mark_slide)

label = tk.Label(
    root,
    text="SPACE = next slide",
    font=("Arial", 24)
)
label.pack(padx=50, pady=50)

root.mainloop()