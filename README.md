# 👁️ Facial Recognition and Behavior Analysis System

This project is an **AI-powered monitoring system** designed to recognize individuals and analyze worker behavior in real time using computer vision. It leverages facial recognition and behavioral analytics to identify inattentive or improper actions, improving workplace efficiency and safety.

---

## 🚀 Features
- **Real-time Face Detection** using OpenCV (Haar Cascade classifier)
- **Facial Recognition** with confidence level indicators
- **Behavioral Analysis** to detect actions such as:
  - Standing
  - Sitting
  - Throwing or abnormal motion
- **Live Dashboard** built with Dash and Flask
  - Displays live video feed
  - Detection logs with timestamps
  - Confidence gauge and analytics charts
- **Alert System** using audio notifications for improper behavior

---

## 🧩 System Architecture

## ⚙️ Tech Stack
| Component | Technology |
|------------|-------------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Web Framework | Flask |
| Dashboard | Dash + Plotly |
| Styling | Dash Bootstrap Components (Cyborg Theme) |
| Data Storage | JSON / MongoDB |
| Alert System | winsound (audio alert) |

---

## 🧠 How It Works
1. The camera captures the video stream.
2. OpenCV processes frames for **face detection and recognition**.
3. Behavioral patterns are analyzed (e.g., sitting, standing, throwing motion).
4. Detected events and confidence values are logged.
5. Flask serves the data to the Dash dashboard in real time.
6. The dashboard displays live metrics, graphs, and alerts.

---

## 🧩 Installation and Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<Richard-Sylvester>/Facial_Recognition_Alert_System.git
   cd Facial_Recognition_Alert_System
