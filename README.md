
# 🚗 SafeDrive – Driver Drowsiness Detection System

SafeDrive is a real-time driver safety application that uses computer vision and machine learning to detect signs of fatigue (like prolonged eye closure and yawning) and issue timely alerts to prevent accidents.

---

## ⚙️ Key Features

- 🎯 **Real-Time Face & Landmark Detection** using OpenCV and MediaPipe  
- 😴 **Drowsiness Detection** via Blink Rate (EAR) and Yawn (MAR) analysis  
- 🖥 **Desktop GUI** built with Tkinter to show live video feed plus visual/audio alerts  
- 📊 **Session Logging & Multithreading** for responsive GUI and performance tracking  

---

## 🔧 Tech Stack

- Python 3.x  
- OpenCV  
- MediaPipe  
- scikit‑learn or custom threshold-based model  
- Tkinter (GUI)  
- Multithreading & session-based logging  

---

## 🧩 Installation

```bash
git clone https://github.com/vivekvarma-01/Safe-Drive.git
cd Safe-Drive
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py
```

Then:

1. Watch the live camera feed with drowsiness alerts  
2. Enable audio alerts for blink/yawn thresholds  
3. Logs stored for each session  

---

## 🧠 How It Works

1. Capture live webcam frames using OpenCV  
2. Detect facial landmarks via MediaPipe  
3. Compute EAR (Eye Aspect Ratio) & MAR (Mouth Aspect Ratio)  
4. Track blinking and yawning to assess fatigue  
5. Trigger visual and audio alerts when thresholds crossed  

---

## 📌 Possible Enhancements

- [ ] Integrate deep learning models to replace rule-based detection  
- [ ] Add mobile support using Kivy or Flutter  
- [ ] Export session logs to CSV or database  
- [ ] Real-time analytics and threshold tuning UI  
