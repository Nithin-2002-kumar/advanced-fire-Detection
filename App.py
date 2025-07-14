import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import pygame
import threading
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize pygame for alarm sound
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound('C:\\Users\\DELL\\PycharmProjects\\MaheshProject\\alarm.mp3')
except:
    print("Alarm sound file not found. Using default beep.")
    # Create a simple beep sound
    beep = pygame.mixer.Sound(buffer=bytearray([128] * 8000 * 2))  # 1 second of beep
    alarm_sound = beep


# Enhanced CNN Model with Dropout and BatchNorm
class EnhancedDetectionModel(nn.Module):
    def __init__(self):
        super(EnhancedDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)  # 2 outputs: smoke and flame probabilities

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x


# Load model function with error handling
def load_model():
    model = EnhancedDetectionModel()
    try:
        # In a real application, you would load your trained weights here
        # model.load_state_dict(torch.load('fire_detection_model.pth'))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return model


model = load_model()

# Enhanced transform with data augmentation for training (not used in inference)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform for input images during inference
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Fire Detection System")
        self.root.geometry("1000x800")

        # Configure styles
        self.configure_styles()

        # Variables
        self.camera_active = False
        self.alarm_playing = False
        self.current_frame = None
        self.detection_active = True
        self.recording = False
        self.detection_history = []
        self.threshold_history = []
        self.video_writer = None
        self.record_start_time = None
        self.video_capture = None  # For uploaded video playback
        self.video_playing = False  # Track if video is playing

        # Thresholds for detection with adjustable values
        self.smoke_threshold = tk.DoubleVar(value=0.7)
        self.flame_threshold = tk.DoubleVar(value=0.7)

        # Create GUI
        self.create_widgets()

        # Start with webcam disabled
        self.cap = None

        # Initialize detection log file
        self.init_log_file()

    def configure_styles(self):
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Alert.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Red.TButton', foreground='red')

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(title_frame, text="Advanced Fire Detection System", style='Title.TLabel').pack()

        # Video and controls frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - video and detection
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Frame for video display
        self.video_frame = ttk.LabelFrame(left_panel, text="Live Detection")
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()

        # Detection results
        self.result_frame = ttk.LabelFrame(left_panel, text="Detection Results")
        self.result_frame.pack(fill=tk.X, pady=(0, 10))

        self.smoke_label = ttk.Label(self.result_frame, text="Smoke: Not Detected", style='TLabel')
        self.smoke_label.pack(anchor=tk.W)

        self.flame_label = ttk.Label(self.result_frame, text="Flame: Not Detected", style='TLabel')
        self.flame_label.pack(anchor=tk.W)

        self.confidence_frame = ttk.LabelFrame(left_panel, text="Detection Confidence")
        self.confidence_frame.pack(fill=tk.X)

        self.smoke_confidence = ttk.Label(self.confidence_frame, text="Smoke Confidence: 0%")
        self.smoke_confidence.pack(anchor=tk.W)

        self.flame_confidence = ttk.Label(self.confidence_frame, text="Flame Confidence: 0%")
        self.flame_confidence.pack(anchor=tk.W)

        # Control buttons
        self.control_frame = ttk.Frame(left_panel)
        self.control_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_btn = ttk.Button(self.control_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(self.control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.upload_btn = ttk.Button(self.control_frame, text="Upload File", command=self.upload_file)
        self.upload_btn.pack(side=tk.LEFT, padx=2)

        self.close_file_btn = ttk.Button(self.control_frame, text="Close File", command=self.close_file,
                                         state=tk.DISABLED)
        self.close_file_btn.pack(side=tk.LEFT, padx=2)

        self.record_btn = ttk.Button(self.control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=2)

        self.alarm_btn = ttk.Button(self.control_frame, text="Test Alarm", command=self.test_alarm)
        self.alarm_btn.pack(side=tk.LEFT, padx=2)

        # Right panel - settings and analytics
        right_panel = ttk.Frame(content_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Settings frame
        settings_frame = ttk.LabelFrame(right_panel, text="Detection Settings")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="Smoke Threshold:").pack(anchor=tk.W)
        self.smoke_slider = ttk.Scale(settings_frame, from_=0.1, to=0.9, variable=self.smoke_threshold,
                                      command=lambda v: self.update_thresholds())
        self.smoke_slider.pack(fill=tk.X)
        self.smoke_threshold_label = ttk.Label(settings_frame, text=f"Current: {self.smoke_threshold.get():.2f}")
        self.smoke_threshold_label.pack(anchor=tk.W)

        ttk.Label(settings_frame, text="Flame Threshold:").pack(anchor=tk.W, pady=(5, 0))
        self.flame_slider = ttk.Scale(settings_frame, from_=0.1, to=0.9, variable=self.flame_threshold,
                                      command=lambda v: self.update_thresholds())
        self.flame_slider.pack(fill=tk.X)
        self.flame_threshold_label = ttk.Label(settings_frame, text=f"Current: {self.flame_threshold.get():.2f}")
        self.flame_threshold_label.pack(anchor=tk.W)

        # Analytics frame
        analytics_frame = ttk.LabelFrame(right_panel, text="Detection Analytics")
        analytics_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure for matplotlib plot
        self.fig, self.ax = plt.subplots(figsize=(4, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=analytics_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Log frame
        log_frame = ttk.LabelFrame(right_panel, text="Event Log")
        log_frame.pack(fill=tk.BOTH, pady=(10, 0))

        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def update_thresholds(self):
        self.smoke_threshold_label.config(text=f"Current: {self.smoke_threshold.get():.2f}")
        self.flame_threshold_label.config(text=f"Current: {self.flame_threshold.get():.2f}")

    def init_log_file(self):
        log_dir = "detection_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(self.log_file, 'w') as f:
            f.write("Timestamp,Smoke Detected,Flame Detected,Smoke Confidence,Flame Confidence\n")

    def log_event(self, smoke, flame, smoke_conf, flame_conf):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"{timestamp},{smoke},{flame},{smoke_conf:.2f},{flame_conf:.2f}\n"

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

        # Update GUI log
        self.log_text.insert(tk.END, f"{timestamp} - ")
        if smoke:
            self.log_text.insert(tk.END, "SMOKE detected ", 'alert')
        if flame:
            self.log_text.insert(tk.END, "FLAME detected ", 'alert')
        if not smoke and not flame:
            self.log_text.insert(tk.END, "No detection")
        self.log_text.insert(tk.END, f" (Smoke: {smoke_conf:.2f}, Flame: {flame_conf:.2f})\n")
        self.log_text.see(tk.END)

        # Keep only the last 100 entries
        if int(self.log_text.index('end-1c').split('.')[0]) > 100:
            self.log_text.delete(1.0, 2.0)

    def update_analytics(self, smoke_conf, flame_conf):
        # Add to history
        self.detection_history.append((smoke_conf, flame_conf))
        self.threshold_history.append((self.smoke_threshold.get(), self.flame_threshold.get()))

        # Keep only the last 50 points
        if len(self.detection_history) > 50:
            self.detection_history.pop(0)
            self.threshold_history.pop(0)

        # Update plot
        self.ax.clear()

        if len(self.detection_history) > 0:
            x = range(len(self.detection_history))
            smoke_confs = [d[0] for d in self.detection_history]
            flame_confs = [d[1] for d in self.detection_history]
            smoke_thresh = [t[0] for t in self.threshold_history]
            flame_thresh = [t[1] for t in self.threshold_history]

            self.ax.plot(x, smoke_confs, label='Smoke Confidence', color='blue')
            self.ax.plot(x, flame_confs, label='Flame Confidence', color='orange')
            self.ax.plot(x, smoke_thresh, '--', label='Smoke Threshold', color='lightblue')
            self.ax.plot(x, flame_thresh, '--', label='Flame Threshold', color='peachpuff')

            self.ax.set_ylim(0, 1)
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Confidence')
            self.ax.legend(loc='upper right', fontsize='small')
            self.ax.grid(True)

        self.canvas.draw()

    def start_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video device")
                return

            self.camera_active = True
            self.video_playing = False
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.NORMAL)
            self.close_file_btn.config(state=tk.DISABLED)
            self.status_bar.config(text="Camera active - Monitoring for fire/smoke")
            self.update_camera()

    def stop_camera(self):
        if self.camera_active:
            self.camera_active = False
            if self.cap is not None:
                self.cap.release()
            self.stop_recording()
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.upload_btn.config(state=tk.NORMAL)
            self.video_label.config(image='')
            self.status_bar.config(text="Camera stopped")

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.camera_active:
            messagebox.showwarning("Warning", "Camera must be active to start recording")
            return

        self.recording = True
        self.record_btn.config(text="Stop Recording")
        self.record_start_time = datetime.now()

        # Create video output directory if it doesn't exist
        video_dir = "recordings"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        # Define the codec and create VideoWriter object
        timestamp = self.record_start_time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(video_dir, f"recording_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

        self.status_bar.config(text=f"Recording started - Saving to {output_file}")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.record_btn.config(text="Start Recording")
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            duration = (datetime.now() - self.record_start_time).total_seconds()
            self.status_bar.config(text=f"Recording stopped - Duration: {duration:.1f} seconds")

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image/Video Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"),
                       ("All Files", "*.*")])

        if file_path:
            self.stop_camera()  # Stop camera if running

            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.process_image(file_path)
            else:
                self.process_video(file_path)

    def process_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display image
            img_pil = Image.fromarray(img)
            img_pil.thumbnail((640, 480))
            img_tk = ImageTk.PhotoImage(img_pil)
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk

            # Process detection
            smoke, flame, smoke_conf, flame_conf = self.detect(img)
            self.update_results(smoke, flame, smoke_conf, flame_conf)

            if smoke or flame:
                self.trigger_alarm()

            self.status_bar.config(text=f"Processed image: {os.path.basename(image_path)}")
            self.close_file_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def process_video(self, video_path):
        try:
            # Stop any currently playing video
            self.close_file()

            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                raise ValueError("Could not open video file")

            self.video_playing = True
            self.close_file_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.DISABLED)
            self.start_btn.config(state=tk.DISABLED)

            def video_loop():
                while self.video_playing and self.video_capture.isOpened():
                    ret, frame = self.video_capture.read()
                    if not ret:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame)
                    frame_pil.thumbnail((640, 480))
                    frame_tk = ImageTk.PhotoImage(frame_pil)

                    self.video_label.config(image=frame_tk)
                    self.video_label.image = frame_tk

                    # Process detection
                    smoke, flame, smoke_conf, flame_conf = self.detect(frame)
                    self.update_results(smoke, flame, smoke_conf, flame_conf)

                    if smoke or flame:
                        self.trigger_alarm()

                    time.sleep(0.03)  # Control playback speed

                # Video finished or stopped
                if self.video_capture is not None:
                    self.video_capture.release()
                self.video_playing = False
                self.status_bar.config(text=f"Finished processing video: {os.path.basename(video_path)}")
                self.close_file_btn.config(state=tk.DISABLED)
                self.upload_btn.config(state=tk.NORMAL)
                self.start_btn.config(state=tk.NORMAL)

            threading.Thread(target=video_loop, daemon=True).start()
            self.status_bar.config(text=f"Processing video: {os.path.basename(video_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process video: {str(e)}")

    def close_file(self):
        """Stop video playback and clear the display"""
        self.video_playing = False

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        self.video_label.config(image='')
        self.close_file_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.NORMAL)
        self.status_bar.config(text="File closed")

    def update_camera(self):
        if self.camera_active and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = frame.copy()

                # Display frame
                frame_pil = Image.fromarray(frame)
                frame_pil.thumbnail((640, 480))
                frame_tk = ImageTk.PhotoImage(frame_pil)
                self.video_label.config(image=frame_tk)
                self.video_label.image = frame_tk

                if self.recording and self.video_writer is not None:
                    # Write frame to video file (convert back to BGR)
                    recording_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(cv2.resize(recording_frame, (640, 480)))

                if self.detection_active:
                    # Process detection
                    smoke, flame, smoke_conf, flame_conf = self.detect(frame)
                    self.update_results(smoke, flame, smoke_conf, flame_conf)

                    if smoke or flame:
                        self.trigger_alarm()

            self.root.after(10, self.update_camera)

    def detect(self, frame):
        try:
            # Preprocess frame for model
            img_pil = Image.fromarray(frame)
            img_tensor = transform(img_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)

            smoke_prob = outputs[0][0].item()
            flame_prob = outputs[0][1].item()

            smoke_detected = smoke_prob > self.smoke_threshold.get()
            flame_detected = flame_prob > self.flame_threshold.get()

            return smoke_detected, flame_detected, smoke_prob, flame_prob

        except Exception as e:
            print(f"Detection error: {e}")
            return False, False, 0.0, 0.0

    def update_results(self, smoke, flame, smoke_conf, flame_conf):
        # Update detection labels
        if smoke:
            self.smoke_label.config(text=f"SMOKE DETECTED!", style='Alert.TLabel', foreground='red')
        else:
            self.smoke_label.config(text="Smoke: Not Detected", style='TLabel', foreground='black')

        if flame:
            self.flame_label.config(text=f"FLAME DETECTED!", style='Alert.TLabel', foreground='red')
        else:
            self.flame_label.config(text="Flame: Not Detected", style='TLabel', foreground='black')

        # Update confidence levels
        self.smoke_confidence.config(text=f"Smoke Confidence: {smoke_conf * 100:.1f}%")
        self.flame_confidence.config(text=f"Flame Confidence: {flame_conf * 100:.1f}%")

        # Log the event
        self.log_event(smoke, flame, smoke_conf, flame_conf)

        # Update analytics plot
        self.update_analytics(smoke_conf, flame_conf)

    def trigger_alarm(self):
        if not self.alarm_playing:
            self.alarm_playing = True

            # Play sound in a separate thread to avoid blocking
            def play_alarm():
                try:
                    alarm_sound.play()
                except:
                    # Fallback beep if alarm sound fails
                    pygame.mixer.Sound.play(pygame.mixer.Sound(buffer=bytearray([128] * 8000 * 1)))  # 1 second beep
                self.alarm_playing = False

            threading.Thread(target=play_alarm, daemon=True).start()

            # Visual alert
            self.flash_screen()

    def flash_screen(self):
        original_bg = self.root.cget('bg')

        def flash():
            for _ in range(3):
                self.root.config(bg='red')
                self.root.update()
                time.sleep(0.3)
                self.root.config(bg=original_bg)
                self.root.update()
                time.sleep(0.3)

        threading.Thread(target=flash, daemon=True).start()

    def test_alarm(self):
        self.trigger_alarm()

    def on_closing(self):
        self.stop_camera()
        self.close_file()
        self.root.destroy()


# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()

    # Configure tag for alert text in log
    text = tk.Text()
    text.tag_config('alert', foreground='red')

    app = FireDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Center the window
    window_width = 1000
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    root.mainloop()