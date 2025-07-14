# 🔥 Flame and Smoke Detection using Enhanced YOLOv5s + CNN

This project is a real-time fire detection system using deep learning and a GUI built with Tkinter.

## 🔧 Features

- Real-time camera feed analysis
- Detection of flame and smoke
- Adjustable confidence thresholds
- Alarm alerts (audio)
- Upload video files for analysis
- Event logging and analytics visualization

## 🧠 Architecture

- CNN-based classifier (PyTorch)
- Tkinter-based GUI
- YOLO-inspired design (see docs for FLAME model)

## 📁 Folder Structure

- `App.py` - Main Python GUI application
- `alarm.mp3` - Alarm audio file
- `docs/` - Research papers and project report
- `model/` - Trained model weights (optional)
- `logs/` - Detection logs (auto-generated)

## ▶️ How to Run

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the app:

    ```bash
    python App.py
    ```

3. Optional: Put your model weights in the `model/` folder.

## 📄 Requirements

See `requirements.txt`.

## 📜 References

- [FLAME: Deep Neural Fire Detection with Motion Analysis](https://doi.org/10.1007/s00521-024-10963-z)
- [ODConvBS-YOLOv5s Detection Paper](./docs/YOLOv5_ODConvBS_Document.docx)
