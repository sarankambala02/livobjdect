import streamlit as st 
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
import av
import cv2
import os
import time
import torch
from PIL import Image

# ====================== 🎨 LOGO + HEADING ======================
try:
    logo = Image.open("logo.png")
except Exception as e:
    st.warning(f"⚠ Logo not found: {e}")
    logo = None

col1, col2 = st.columns([1, 6])
with col1:
    if logo:
        st.image(logo, width=120)

with col2:
    st.markdown(
        """
        <h1 style="text-align:left; font-size:52px; font-weight:900; color:#0A84FF;">
            🔥 LIVE OBJECT DETECTION
        </h1>
        <p style="text-align:left; font-size:20px; font-weight:500; color:#4F4F4F;">
            <span style="font-weight:700; color:#FF9500;">Real‑Time AI Vision</span> powered by 
            <span style="font-weight:700; color:#00BFFF;">YOLOv8</span>
        </p>
        """,
        unsafe_allow_html=True,
    )

# ====================== ℹ ABOUT ======================
with st.expander("ℹ About this project"):
    st.markdown("""
    *👩‍💻 Developer:* Pavani Addala  

    *💡 Features:*  
    ✅ Real‑Time Object Detection (YOLOv8)  
    ✅ People Count & Object Labels  
    ✅ FPS & Frame Stats  
    ✅ Snapshot Save & Download  

    *📸 How to Use:*  
    ▶ Click *Start* to enable webcam.  
    📷 Watch live detections and counts.  
    💾 Use *Save Snapshot* to save an image.
    """)

# ====================== ⚙ SIDEBAR ======================
st.sidebar.header("⚙ Controls")
confidence_slider = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.3, 0.05)

# ====================== 🔍 YOLO MODEL LOAD (Safe) ======================
try:
    if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
        import ultralytics.nn.tasks as tasks
        torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': tasks.DetectionModel})

    model = YOLO("yolov8n.pt")
    st.sidebar.success("✅ YOLOv8 model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Model loading failed: {e}")
    st.stop()

# ====================== RTC CONFIG ======================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ====================== 🎥 VIDEO PROCESSOR ======================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.total_frames = 0
        self.start_time = time.time()

    def recv(self, frame):
        self.total_frames += 1
        elapsed = time.time() - self.start_time
        fps = self.total_frames / elapsed if elapsed > 0 else 0

        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, verbose=False)
        person_count = 0

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < confidence_slider:
                    continue
                cls = int(box.cls[0])
                class_name = model.names[cls]
                if class_name.lower() == "person":
                    person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(img, f"People Count: {person_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f"Frames: {self.total_frames}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imwrite("current_frame.jpg", img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ====================== 🧩 MAIN APP ======================
col_main, col_side = st.columns([3, 1])
with col_main:
    webrtc_streamer(
        key="detect",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_side:
    st.subheader("📸 Snapshot Feature")
    if st.button("📷 Save Snapshot"):
        if os.path.exists("current_frame.jpg"):
            os.makedirs("snapshots", exist_ok=True)
            filename = f"snapshots/snapshot_{int(time.time())}.jpg"
            frame = cv2.imread("current_frame.jpg")
            if frame is not None:
                cv2.imwrite(filename, frame)
                st.success(f"✅ Snapshot saved: {filename}")
                st.image(filename, caption="Saved Snapshot", use_container_width=True)
                with open(filename, "rb") as file:
                    st.download_button(
                        label="⬇ Download Snapshot",
                        data=file,
                        file_name=os.path.basename(filename),
                        mime="image/jpeg"
                    )
            else:
                st.warning("⚠ Could not read frame!")
        else:
            st.warning("⚠ No frame captured yet!")
