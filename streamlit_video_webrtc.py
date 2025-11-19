import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid
import cv2
import mediapipe as mp
import numpy as np
import imageio
import time

# --- Setup folders ---
BASE_DIR = Path.cwd()
VIDEO_DIR = BASE_DIR / "video_grammar"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# --- MediaPipe setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(input_path: Path, output_path: Path, progress_callback=None):
    """
    Process a video with MediaPipe Pose:
    - Draw landmarks on black background
    - Write output using imageio-ffmpeg (H.264)
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", ffmpeg_log_level="error")

    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    landmark_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3)
    connection_spec = mp_drawing.DrawingSpec(thickness=2)

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            black = np.zeros_like(frame)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    black,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_spec,
                    connection_spec
                )

            black_rgb = cv2.cvtColor(black, cv2.COLOR_BGR2RGB)
            writer.append_data(black_rgb)

            frame_idx += 1
            if progress_callback and total_frames > 0 and (frame_idx % 5 == 0 or frame_idx == total_frames):
                progress_callback(min(1.0, frame_idx / total_frames))
    finally:
        cap.release()
        writer.close()
        pose.close()

# --- Streamlit UI ---
st.set_page_config(page_title="Video Grammar — Streamlit", layout="centered")
st.title("Video Grammar — MediaPipe Landmarks on Black Background")
st.markdown("""
Upload a video (MP4 preferred). The app will process the video to display
MediaPipe Pose landmarks on a black background, play it in the web app,
and allow you to download it.
""")

uploaded = st.file_uploader("Choose a video file (mp4 preferred)", type=["mp4", "mov", "webm", "mkv"])

if uploaded is not None:
    video_bytes = uploaded.read()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uniq = uuid.uuid4().hex[:6]

    raw_name = f"raw_{ts}_{uniq}{Path(uploaded.name).suffix}"
    raw_path = VIDEO_DIR / raw_name
    with open(raw_path, "wb") as f:
        f.write(video_bytes)
    st.success(f"Saved raw video: `{raw_path.name}`")

    processed_name = f"processed_{ts}_{uniq}_landmarks_black.mp4"
    processed_path = VIDEO_DIR / processed_name

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.text("Processing video... this may take a while for long videos.")

    def progress_cb(frac):
        try:
            progress_bar.progress(min(1.0, float(frac)))
        except Exception:
            pass

    try:
        t0 = time.time()
        process_video(raw_path, processed_path, progress_callback=progress_cb)
        elapsed = time.time() - t0
        progress_bar.progress(1.0)
        size_bytes = processed_path.stat().st_size
        status_text.success(f"Processing complete in {elapsed:.1f}s. Saved to `{processed_path.name}` ({size_bytes:,} bytes).")

        # Display processed video
        with open(processed_path, "rb") as vf:
            st.video(vf.read())
        st.download_button("Download processed video", open(processed_path, "rb").read(), file_name=processed_path.name, mime="video/mp4")

    except Exception as e:
        st.error(f"Processing failed: {e}")

# Show recent processed videos
st.markdown("---")
st.subheader("Recent processed videos")
processed_files = sorted(VIDEO_DIR.glob("*landmarks_black.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
if not processed_files:
    st.info("No processed videos found yet.")
else:
    for p in processed_files[:6]:
        st.write(p.name)
        with open(p, "rb") as vf:
            st.video(vf.read())
        st.markdown("---")
