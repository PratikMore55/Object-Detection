import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from PIL import Image, ImageTk
import time

# Load Machine Learning Model & Encoder
model = joblib.load("yoga_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load Pose Classification Model
pose_model = load_model("model.h5")
pose_labels = np.load("labels.npy")

# MediaPipe Pose Detection
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Yoga Routine Recommendations
yoga_routines = {
    "Normal": ["Tadasana", "Vrikshasana", "Bhujangasana"],
    "Healthy": ["Trikonasana", "Surya Namaskar", "Dhanurasana"],
    "Elevated": ["Balasana", "Anulom Vilom", "Shavasana"]
}

# Initialize Tkinter Window
root = tk.Tk()
root.title("AI-Powered Yoga Analyzer")

# Notebook Tabs
notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)  # Health Analysis Tab
tab2 = ttk.Frame(notebook)  # Yoga Practice Tab
notebook.add(tab1, text='Health Input')
notebook.add(tab2, text='Yoga Practice')
notebook.pack(expand=True, fill='both')

# === Health Analysis Tab ===
age_label = ttk.Label(tab1, text="Age:")
age_label.pack()
age_var = tk.IntVar()
age_entry = ttk.Entry(tab1, textvariable=age_var)
age_entry.pack()

hr_label = ttk.Label(tab1, text="Heart Rate (bpm):")
hr_label.pack()
hr_var = tk.StringVar()
hr_entry = ttk.Entry(tab1, textvariable=hr_var)
hr_entry.pack()

spo2_label = ttk.Label(tab1, text="SpO2 (%):")
spo2_label.pack()
spo2_var = tk.StringVar()
spo2_entry = ttk.Entry(tab1, textvariable=spo2_var)
spo2_entry.pack()

result_label = ttk.Label(tab1, text="Health Status: -\nRecommended Yoga: -")
result_label.pack()

def analyze_health():
    """Predicts health status and suggests yoga poses."""
    age = age_var.get()
    heart_rate = int(hr_var.get())
    spo2 = int(spo2_var.get())
    age_category = "15-20" if age < 21 else "21-25"

    input_data = pd.DataFrame([[heart_rate, spo2, age_category]], 
                              columns=['Heart Rate (bpm)', 'SpO2 (%)', 'Age Range'])
    input_data = pd.get_dummies(input_data, columns=['Age Range'])

    for col in model.feature_names_in_:
        if col not in input_data:
            input_data[col] = 0
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]
    health_status = label_encoder.inverse_transform([prediction])[0]
    
    yoga_poses = yoga_routines[health_status]
    result_label.config(text=f"Health Status: {health_status}\nRecommended Yoga: {', '.join(yoga_poses)}")

    # Set first recommended pose in practice tab
    pose_name.set(yoga_poses[0])
    load_pose_image(yoga_poses[0])

analyze_button = ttk.Button(tab1, text="Analyze", command=analyze_health)
analyze_button.pack()

# === Yoga Practice Tab ===
pose_name = tk.StringVar(value="Tadasana")  # Default pose
pose_label = ttk.Label(tab2, textvariable=pose_name, font=("Arial", 16))
pose_label.pack()

# Load Reference Pose Image
pose_image_label = tk.Label(tab2)
pose_image_label.pack()

def load_pose_image(pose):
    """Load the reference image of the suggested yoga pose."""
    try:
        img_path = f"poses/{pose}.jpg"  # Update path as per your folder structure
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        pose_image_label.config(image=imgtk)
        pose_image_label.image = imgtk
    except:
        pose_image_label.config(text="Image Not Found")

# Live Camera Feed
camera_label = tk.Label(tab2)
camera_label.pack()

def inFrame(lst):
    """Check if key body parts are visible"""
    return all(lst[i].visibility > 0.6 for i in [28, 27, 15, 16])

def update_camera():
    """Capture frame, process pose, and update Tkinter window."""
    ret, frm = cap.read()
    if ret:
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            lst = [i.x - res.pose_landmarks.landmark[0].x for i in res.pose_landmarks.landmark] + \
                  [i.y - res.pose_landmarks.landmark[0].y for i in res.pose_landmarks.landmark]

            p = pose_model.predict(np.array(lst).reshape(1, -1))
            pred = pose_labels[np.argmax(p)]
            confidence = p[0][np.argmax(p)]

            if confidence > 0.75:
                cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frm, "Incorrect Pose!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frm, "Ensure full body is visible!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

        # Convert frame to Tkinter format
        img = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        img = img.resize((300, 300))
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    camera_label.after(10, update_camera)

# Countdown Timer
timer_label = ttk.Label(tab2, text="Timer: 30s", font=("Arial", 16))
timer_label.pack()
exercise_time = 30

def countdown(time_left):
    """Update the countdown timer label."""
    if time_left >= 0:
        timer_label.config(text=f"Timer: {time_left}s")
        root.after(1000, countdown, time_left - 1)
    else:
        timer_label.config(text="Exercise Complete!")

def start_timer():
    """Start the exercise countdown."""
    countdown(exercise_time)

start_button = ttk.Button(tab2, text="Start Exercise", command=start_timer)
start_button.pack()

# Start Camera Feed
update_camera()

# Run Tkinter Main Loop
root.mainloop()

# Release Resources on Exit
cap.release()
cv2.destroyAllWindows()
