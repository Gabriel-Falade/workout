# app.py

import os
import sys
import time
import tempfile
from typing import List, Optional, Dict, Any
import cv2 as cv
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- Make 'src' importable when running from project root ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Project imports
from analysis.pose_helpers import mp_results_to_dict
from analysis.frame_metrics import compute_frame_metrics, FrameMetricsState

# Import all exercise detectors
from exercises.pushup import PushUpDetector
from exercises.jumping_jacks import JumpingJacksDetector
# from exercises.squat import SquatDetector  # Uncomment when implemented
# from exercises.burpee import BurpeeDetector  # Uncomment when implemented

# ------------------------------
# Exercise Configuration
# ------------------------------
AVAILABLE_EXERCISES = {
    "push_up": {
        "name": "Push-ups",
        "icon": "💪",
        "detector_class": PushUpDetector,
        "description": "Tracks push-up reps with depth and lockout validation",
        "metrics": ["ROM", "Velocity", "Torso Tilt"],
        "rom_key": "rom_pushup_smooth",
        "vel_key": "vel_pushup",
    },
    "jumping_jacks": {
        "name": "Jumping Jacks",
        "icon": "🤸",
        "detector_class": JumpingJacksDetector,
        "description": "Tracks jumping jacks based on arm and leg position",
        "metrics": ["ROM", "Arms Position", "Feet Position"],
        "rom_key": "rom_pushup_smooth",  # JJ uses custom ROM calculation
        "vel_key": None,
    },
    # Add more exercises as you implement them
    # "squat": {
    #     "name": "Squats",
    #     "icon": "🦵",
    #     "detector_class": SquatDetector,
    #     "description": "Tracks squat depth and form",
    #     "metrics": ["ROM", "Velocity", "Knee Valgus"],
    #     "rom_key": "rom_squat_smooth",
    #     "vel_key": "vel_squat",
    # },
}

# ------------------------------
# WebRTC Configuration
# ------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ------------------------------
# UI helpers
# ------------------------------
def draw_info_box(img, lines: List[str], padding=10, line_height=22):
    """Bottom-right semi-opaque box with lines of text."""
    if img is None or not lines:
        return img
    h, w = img.shape[:2]
    font = cv.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    thickness = 1

    sizes = [cv.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
    text_w = max((sz[0] for sz in sizes), default=0)
    box_w = text_w + padding * 2
    box_h = line_height * len(lines) + padding * 2

    x1 = max(0, w - box_w - 10)
    y1 = max(0, h - box_h - 10)
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = img.copy()
    cv.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    img = cv.addWeighted(overlay, 0.65, img, 0.35, 0)

    y = y1 + padding + 15
    for s in lines:
        cv.putText(img, s, (x1 + padding, y), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
        y += line_height
    return img

def class_to_color(rep_class: str):
    """Map rep classification to a BGR color."""
    if not rep_class:
        return (60, 180, 75)  # green
    rc_lower = rep_class.lower()
    if "fail" in rc_lower:
        return (40, 40, 220)  # red
    if "warn" in rc_lower:
        return (0, 215, 255)  # yellow
    return (60, 180, 75)  # green

# ------------------------------
# Workout Processor Class
# ------------------------------
class WorkoutProcessor:
    def __init__(self, exercise_key: str):
        """Initialize processor with selected exercise."""
        self.exercise_config = AVAILABLE_EXERCISES[exercise_key]
        self.exercise_key = exercise_key
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize the appropriate detector
        detector_class = self.exercise_config["detector_class"]
        self.detector = detector_class()
        
        # Frame metrics state
        self.fm_state = FrameMetricsState()
        
        # FPS tracking
        self.fps_ema = None
        self.prev_time = time.time()
        
    def process_frame(self, frame, lm_dict=None):
        """Process a single frame."""
        # Convert BGR to RGB
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # MediaPipe pose detection
        results = self.pose.process(image_rgb)
        
        # Convert back to BGR
        image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
        image.flags.writeable = True
        
        # Get landmarks
        lm = mp_results_to_dict(results) if lm_dict is None else lm_dict
        
        # Calculate timing
        now = time.time()
        dt = max(1e-3, now - self.prev_time)
        self.prev_time = now
        fps_inst = 1.0 / dt
        self.fps_ema = fps_inst if self.fps_ema is None else (0.2 * fps_inst + 0.8 * self.fps_ema)
        
        # Compute metrics
        fm = compute_frame_metrics(lm, dt, self.fm_state)
        
        # Update detector - handle different detector signatures
        try:
            # Try push-up style (fm only)
            rep_event, live = self.detector.update(fm, now_s=now)
        except TypeError:
            # Try jumping jacks style (fm and lm)
            rep_event, live = self.detector.update(fm, lm, now_s=now)
        
        # Draw skeleton
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # Mirror the image
        image = cv.flip(image, 1)
        
        # Top-left: reps & stage
        cv.rectangle(image, (0, 0), (240, 74), (245, 117, 16), -1)
        cv.putText(image, 'REPS', (15, 14), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, str(live.get("rep_count", 0)), (10, 62), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, 'STAGE', (88, 14), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, str(live.get("stage", "") or "--"), (80, 62), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
        
        # Bottom-right: calculations
        rom_key = self.exercise_config["rom_key"]
        vel_key = self.exercise_config["vel_key"]
        
        rom = live.get("rom") or fm.get(rom_key)
        vel = live.get("vel") or (fm.get(vel_key) if vel_key else None)
        tilt = fm.get("torso_tilt_deg")
        
        info_lines = [
            f"ROM:  {rom:5.1f} %" if rom is not None else "ROM:  --",
            f"Vel:  {vel:5.1f} %/s" if vel is not None else "Vel:  --" if vel_key else "",
            f"Tilt: {tilt:4.1f} deg" if tilt is not None else "Tilt: --",
            f"FPS:  {self.fps_ema:4.1f}" if self.fps_ema is not None else "FPS:  --",
            f"Mode: {self.exercise_config['name']}",
        ]
        info_lines = [line for line in info_lines if line]  # Remove empty lines
        image = draw_info_box(image, info_lines)
        
        return image, live, rep_event

# ------------------------------
# Video Processing Function
# ------------------------------
def process_video(video_path, exercise_key, st_video_placeholder, st_info_placeholder, st_progress_bar):
    """Process video file frame by frame."""
    video_capture = cv.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        st.error("Failed to open video file")
        return
    
    total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv.CAP_PROP_FPS)
    
    processor = WorkoutProcessor(exercise_key)
    frame_count = 0
    rep_events = []

    while video_capture.isOpened():
        ok, frame = video_capture.read()
        if not ok:
            break

        frame_count += 1
        progress = frame_count / total_frames
        st_progress_bar.progress(progress)

        # Process frame
        processed_frame, live, rep_event = processor.process_frame(frame)
        
        if rep_event:
            rep_events.append(rep_event)
        
        # Convert to RGB for Streamlit
        display_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
        st_video_placeholder.image(display_frame, channels="RGB", use_column_width=True)

        # Display stats
        rom = live.get("rom", 0.0)
        info_text = f"""
        **Current Stage**: {live.get("stage", "--")}  
        **Total Reps**: {live.get("rep_count", 0)}  
        **ROM**: {rom:.1f}%
        """
        st_info_placeholder.markdown(info_text)
        
        time.sleep(1.0 / fps if fps > 0 else 0.033)

    video_capture.release()
    
    # Show summary
    st.success(f"✅ Video processing completed! Total reps: {live.get('rep_count', 0)}")
    
    # Display rep breakdown if available
    if rep_events:
        st.subheader("📊 Rep Breakdown")
        for i, event in enumerate(rep_events, 1):
            color = "🟢" if event.get("counted") else "🔴"
            st.write(f"{color} Rep {i}: {event.get('class', 'N/A')} - {', '.join(event.get('cues', []))}")

# ------------------------------
# Webcam callback
# ------------------------------
def video_frame_callback(frame):
    """Callback function for processing webcam frames."""
    img = frame.to_ndarray(format="bgr24")
    
    # Get selected exercise from session state
    exercise_key = st.session_state.get('selected_exercise', 'push_up')
    
    # Initialize or update processor if exercise changed
    if 'processor' not in st.session_state or st.session_state.get('last_exercise') != exercise_key:
        st.session_state.processor = WorkoutProcessor(exercise_key)
        st.session_state.last_exercise = exercise_key
    
    # Process the frame
    processed_img, live, rep_event = st.session_state.processor.process_frame(img)
    
    # Store live stats and events
    st.session_state.live_stats = live
    if rep_event:
        if 'rep_events' not in st.session_state:
            st.session_state.rep_events = []
        st.session_state.rep_events.append(rep_event)
    
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# ------------------------------
# Main Streamlit App
# ------------------------------
def main():
    st.set_page_config(layout="wide", page_title="AI Workout Analyzer")
    st.title("🏋️ AI-Powered Workout App")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Exercise Selection
    st.sidebar.subheader("🎯 Select Exercise")
    
    exercise_options = {
        key: f"{config['icon']} {config['name']}"
        for key, config in AVAILABLE_EXERCISES.items()
    }
    
    selected_exercise_display = st.sidebar.selectbox(
        "Choose your workout:",
        options=list(exercise_options.values()),
        index=0
    )
    
    # Get the exercise key from display name
    selected_exercise = [k for k, v in exercise_options.items() if v == selected_exercise_display][0]
    
    # Store in session state
    if 'selected_exercise' not in st.session_state:
        st.session_state.selected_exercise = selected_exercise
    else:
        # Update if changed
        if st.session_state.selected_exercise != selected_exercise:
            st.session_state.selected_exercise = selected_exercise
            # Clear previous workout data
            if 'rep_events' in st.session_state:
                del st.session_state.rep_events
            if 'live_stats' in st.session_state:
                del st.session_state.live_stats
    
    # Display exercise info
    exercise_config = AVAILABLE_EXERCISES[selected_exercise]
    st.sidebar.info(f"**{exercise_config['icon']} {exercise_config['name']}**\n\n{exercise_config['description']}")
    
    st.sidebar.markdown("---")
    
    # Source Selection
    source = st.sidebar.radio("Video Source", ("📹 Live Webcam", "📁 Upload Video File"))
    
    # ===== WEBCAM MODE =====
    if source == "📹 Live Webcam":
        st.header(f"📹 Live {exercise_config['name']} Analysis")
        
        st.info("👇 Click 'START' to begin webcam analysis. Allow camera permissions when prompted.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            webrtc_ctx = webrtc_streamer(
                key=f"workout-analyzer-{selected_exercise}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        
        with col2:
            st.subheader("📊 Live Statistics")
            stats_placeholder = st.empty()
            
            # Update stats display
            if 'live_stats' in st.session_state:
                live = st.session_state.live_stats
                stats_placeholder.markdown(f"""
                ### Current Workout Stats
                
                - **Exercise**: {exercise_config['icon']} {exercise_config['name']}
                - **Stage**: {live.get('stage', '--')}
                - **Rep Count**: {live.get('rep_count', 0)}
                - **ROM**: {live.get('rom', 0.0):.1f}%
                """)
                
                # Show recent rep events
                if 'rep_events' in st.session_state and st.session_state.rep_events:
                    st.markdown("### 📝 Recent Reps")
                    for event in st.session_state.rep_events[-5:]:  # Last 5 reps
                        color = "🟢" if event.get("counted") else "🔴"
                        st.write(f"{color} {event.get('class', 'N/A')}: {', '.join(event.get('cues', ['Good form!']))}")
            else:
                stats_placeholder.info("Waiting for video stream...")
        
        st.markdown("---")
        st.markdown(f"""
        ### 💡 Tips for {exercise_config['name']}:
        - Position yourself so your full body is visible
        - Ensure good lighting
        - Stand about 6-8 feet from the camera
        - Maintain proper form throughout each rep
        - Click STOP when finished to see your results
        """)
    
    # ===== VIDEO UPLOAD MODE =====
    else:
        st.header(f"📁 {exercise_config['name']} Video Analysis")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload a video file", 
            type=["mp4", "mov", "avi", "mkv"]
        )
        
        if uploaded_file is None:
            st.info("👆 Please upload a video file to begin analysis")
            
            # Show example of what metrics will be tracked
            st.markdown("### 📊 Metrics Tracked:")
            for metric in exercise_config['metrics']:
                st.write(f"- {metric}")
            
            return
        
        st.sidebar.success("✅ File uploaded successfully!")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Create placeholders
        st_video_placeholder = st.empty()
        st_progress_bar = st.progress(0)
        st_info_placeholder = st.empty()
        
        # Start button
        if st.sidebar.button("▶️ Start Analysis", type="primary"):
            try:
                process_video(
                    video_path, 
                    selected_exercise,
                    st_video_placeholder, 
                    st_info_placeholder, 
                    st_progress_bar
                )
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(video_path):
                    os.unlink(video_path)

if __name__ == "__main__":
    main()