# Facial Paralysis Simulator 🧠💻

A realistic facial paralysis simulation app built with **Python**, **OpenCV**, **MediaPipe**, and **Gradio**.  
Captures a webcam image, applies asymmetric facial droop to simulate paralysis, and generates a printable report layout.

---

## 🧩 Features
- Realistic facial deformation using Delaunay triangulation
- MediaPipe face mesh landmark tracking
- Aspect-safe portrait placement in a custom printable layout
- Built-in Gradio UI (camera → simulation → print-ready report)

---

## 🚀 How to Run Locally

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/facial-paralysis-sim.git
cd facial-paralysis-sim

# Create environment
conda env create -f environment.yml
conda activate facial-paralysis-sim

# Run the app
python app.py
