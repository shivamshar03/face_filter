import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading
from typing import Union

# Configure WebRTC for better connectivity
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.filter_type = "none"
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.lock = threading.Lock()

    def set_filter(self, filter_type: str):
        with self.lock:
            self.filter_type = filter_type

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with self.lock:
            current_filter = self.filter_type

        # Apply the selected filter
        if current_filter == "dog":
            img = self.apply_dog_filter(img)
        elif current_filter == "cat":
            img = self.apply_cat_filter(img)
        elif current_filter == "sunglasses":
            img = self.apply_sunglasses_filter(img)
        elif current_filter == "rainbow":
            img = self.apply_rainbow_filter(img)
        elif current_filter == "hearts":
            img = self.apply_heart_eyes_filter(img)
        elif current_filter == "mustache":
            img = self.apply_mustache_filter(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def detect_faces(self, image):
        """Detect faces using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        return results.detections if results.detections else []

    def get_face_landmarks(self, image):
        """Get detailed face landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        return results.multi_face_landmarks if results.multi_face_landmarks else []

    def apply_dog_filter(self, image):
        """Apply dog ears and nose filter"""
        detections = self.detect_faces(image)
        height, width = image.shape[:2]

        for detection in detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            # Dog ears
            ear_size = w // 3
            # Left ear
            pts1 = np.array([[x - ear_size // 3, y - ear_size // 3],
                             [x + ear_size // 3, y - ear_size],
                             [x + ear_size, y + ear_size // 3]], np.int32)
            cv2.fillPoly(image, [pts1], (101, 67, 33))  # Brown
            cv2.polylines(image, [pts1], True, (50, 25, 0), 2)

            # Right ear
            pts2 = np.array([[x + w - ear_size, y + ear_size // 3],
                             [x + w - ear_size // 3, y - ear_size],
                             [x + w + ear_size // 3, y - ear_size // 3]], np.int32)
            cv2.fillPoly(image, [pts2], (101, 67, 33))
            cv2.polylines(image, [pts2], True, (50, 25, 0), 2)

            # Dog nose
            nose_x = x + w // 2
            nose_y = y + int(h * 0.6)
            cv2.ellipse(image, (nose_x, nose_y), (w // 10, h // 15), 0, 0, 360, (0, 0, 0), -1)

            # Tongue
            tongue_pts = np.array([[nose_x - w // 15, nose_y + h // 15],
                                   [nose_x + w // 15, nose_y + h // 15],
                                   [nose_x + w // 20, nose_y + h // 8],
                                   [nose_x - w // 20, nose_y + h // 8]], np.int32)
            cv2.fillPoly(image, [tongue_pts], (203, 192, 255))

        return image

    def apply_cat_filter(self, image):
        """Apply cat ears and whiskers filter"""
        detections = self.detect_faces(image)
        height, width = image.shape[:2]

        for detection in detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            # Cat ears
            ear_size = w // 3
            # Left ear
            pts1 = np.array([[x, y + ear_size // 4],
                             [x + ear_size // 2, y - ear_size // 2],
                             [x + ear_size, y + ear_size // 2]], np.int32)
            cv2.fillPoly(image, [pts1], (128, 128, 128))

            # Inner ear (pink)
            inner_pts1 = np.array([[x + ear_size // 4, y + ear_size // 8],
                                   [x + ear_size // 2, y - ear_size // 4],
                                   [x + ear_size * 3 // 4, y + ear_size // 4]], np.int32)
            cv2.fillPoly(image, [inner_pts1], (203, 192, 255))

            # Right ear
            pts2 = np.array([[x + w - ear_size, y + ear_size // 2],
                             [x + w - ear_size // 2, y - ear_size // 2],
                             [x + w, y + ear_size // 4]], np.int32)
            cv2.fillPoly(image, [pts2], (128, 128, 128))

            # Inner ear (pink)
            inner_pts2 = np.array([[x + w - ear_size * 3 // 4, y + ear_size // 4],
                                   [x + w - ear_size // 2, y - ear_size // 4],
                                   [x + w - ear_size // 4, y + ear_size // 8]], np.int32)
            cv2.fillPoly(image, [inner_pts2], (203, 192, 255))

            # Whiskers
            whisker_y = y + h // 2
            whisker_length = w // 2

            # Left whiskers
            for i, offset in enumerate([-15, 0, 15]):
                cv2.line(image, (x - whisker_length, whisker_y + offset),
                         (x, whisker_y + offset), (0, 0, 0), 3)

            # Right whiskers
            for i, offset in enumerate([-15, 0, 15]):
                cv2.line(image, (x + w, whisker_y + offset),
                         (x + w + whisker_length, whisker_y + offset), (0, 0, 0), 3)

            # Cat nose (small triangle)
            nose_x = x + w // 2
            nose_y = y + int(h * 0.55)
            nose_pts = np.array([[nose_x, nose_y - h // 20],
                                 [nose_x - w // 25, nose_y + h // 30],
                                 [nose_x + w // 25, nose_y + h // 30]], np.int32)
            cv2.fillPoly(image, [nose_pts], (203, 192, 255))

        return image

    def apply_sunglasses_filter(self, image):
        """Apply sunglasses filter"""
        landmarks_list = self.get_face_landmarks(image)
        height, width = image.shape[:2]

        for landmarks in landmarks_list:
            # Get eye positions (approximate)
            left_eye = landmarks.landmark[33]  # Left eye corner
            right_eye = landmarks.landmark[263]  # Right eye corner

            # Convert to pixel coordinates
            left_eye_x = int(left_eye.x * width)
            left_eye_y = int(left_eye.y * height)
            right_eye_x = int(right_eye.x * width)
            right_eye_y = int(right_eye.y * height)

            # Calculate sunglasses dimensions
            eye_distance = right_eye_x - left_eye_x
            glasses_width = int(eye_distance * 1.5)
            glasses_height = int(glasses_width * 0.4)

            # Center position
            center_x = (left_eye_x + right_eye_x) // 2
            center_y = (left_eye_y + right_eye_y) // 2

            # Draw sunglasses frame
            top_left = (center_x - glasses_width // 2, center_y - glasses_height // 2)
            bottom_right = (center_x + glasses_width // 2, center_y + glasses_height // 2)

            # Frame outline
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 3)

            # Left lens
            lens_width = glasses_width // 2 - 15
            lens_height = glasses_height - 10
            left_lens_tl = (top_left[0] + 5, top_left[1] + 5)
            left_lens_br = (left_lens_tl[0] + lens_width, left_lens_tl[1] + lens_height)
            cv2.rectangle(image, left_lens_tl, left_lens_br, (40, 40, 40), -1)

            # Right lens
            right_lens_tl = (center_x + 10, top_left[1] + 5)
            right_lens_br = (right_lens_tl[0] + lens_width, right_lens_tl[1] + lens_height)
            cv2.rectangle(image, right_lens_tl, right_lens_br, (40, 40, 40), -1)

            # Bridge
            cv2.line(image, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 0), 3)

        return image

    def apply_rainbow_filter(self, image):
        """Apply rainbow aura around face"""
        detections = self.detect_faces(image)
        height, width = image.shape[:2]

        for detection in detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            center = (x + w // 2, y + h // 2)
            base_radius = max(w, h) // 2 + 30

            # Rainbow colors (ROYGBIV)
            colors = [(0, 0, 255), (0, 127, 255), (0, 255, 255), (0, 255, 0),
                      (255, 255, 0), (255, 0, 255), (255, 0, 0)]

            # Create overlay for transparency effect
            overlay = image.copy()

            # Draw rainbow circles
            for i, color in enumerate(colors):
                cv2.circle(overlay, center, base_radius + i * 8, color, 4)

            # Blend with original
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        return image

    def apply_heart_eyes_filter(self, image):
        """Apply heart eyes filter"""
        landmarks_list = self.get_face_landmarks(image)
        height, width = image.shape[:2]

        for landmarks in landmarks_list:
            # Get eye positions
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]

            left_eye_x = int(left_eye.x * width)
            left_eye_y = int(left_eye.y * height)
            right_eye_x = int(right_eye.x * width)
            right_eye_y = int(right_eye.y * height)

            # Draw hearts over eyes
            heart_size = 25

            # Left heart
            self.draw_heart(image, (left_eye_x, left_eye_y), heart_size, (0, 0, 255))
            # Right heart
            self.draw_heart(image, (right_eye_x, right_eye_y), heart_size, (0, 0, 255))

        return image

    def draw_heart(self, image, center, size, color):
        """Draw a heart shape"""
        x, y = center

        # Heart shape using circles and triangle
        # Top circles
        cv2.circle(image, (x - size // 3, y - size // 4), size // 3, color, -1)
        cv2.circle(image, (x + size // 3, y - size // 4), size // 3, color, -1)

        # Bottom triangle
        pts = np.array([[x - size // 2, y],
                        [x + size // 2, y],
                        [x, y + size // 2]], np.int32)
        cv2.fillPoly(image, [pts], color)

    def apply_mustache_filter(self, image):
        """Apply mustache filter"""
        landmarks_list = self.get_face_landmarks(image)
        height, width = image.shape[:2]

        for landmarks in landmarks_list:
            # Get nose tip position
            nose_tip = landmarks.landmark[1]  # Nose tip landmark

            nose_x = int(nose_tip.x * width)
            nose_y = int(nose_tip.y * height)

            # Mustache dimensions
            mustache_width = width // 8
            mustache_height = height // 25

            # Draw mustache (ellipse)
            cv2.ellipse(image, (nose_x, nose_y + mustache_height),
                        (mustache_width, mustache_height), 0, 0, 180, (0, 0, 0), -1)

            # Add some curves for style
            cv2.ellipse(image, (nose_x - mustache_width // 2, nose_y + mustache_height // 2),
                        (mustache_width // 4, mustache_height // 2), 45, 0, 180, (0, 0, 0), -1)
            cv2.ellipse(image, (nose_x + mustache_width // 2, nose_y + mustache_height // 2),
                        (mustache_width // 4, mustache_height // 2), 135, 0, 180, (0, 0, 0), -1)

        return image


def main():
    st.set_page_config(
        page_title="Real-time Snapchat Filters",
        page_icon="📸",
        layout="wide"
    )

    st.title(" Real-time Snapchat Filters")
    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col2:
        st.header(" Filter Controls")

        # Filter selection
        filter_options = {
            "None": "none",
            " Dog Filter": "dog",
            " Cat Filter": "cat",
            " Sunglasses": "sunglasses",
            " Rainbow Aura": "rainbow",
            " Heart Eyes": "hearts",
            " Mustache": "mustache"
        }

        selected_filter = st.selectbox(
            "Choose your filter:",
            list(filter_options.keys()),
            index=0
        )

        # Instructions
        st.markdown("""
        ###  Instructions:
        1. **Allow camera access** when prompted
        2. **Select a filter** from the dropdown above
        3. **Position your face** in the camera view
        4. **Have fun** with real-time filters!

        ###  Tips:
        - Good lighting improves detection
        - Keep your face centered
        - Move slowly for best tracking
        - Try different angles and expressions
        """)

        # Performance info
        st.info("""
        🔧 **Performance Notes:**
        - Filters apply in real-time
        - Multiple faces supported
        - Optimized for smooth playback
        """)

    with col1:
        st.header(" Live Camera Feed")

        # Create the video transformer
        ctx = webrtc_streamer(
            key="filters",
            video_transformer_factory=VideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1920},
                    "height": {"min": 480, "ideal": 720, "max": 1080},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True,
        )

        # Update filter in real-time
        if ctx.video_transformer:
            ctx.video_transformer.set_filter(filter_options[selected_filter])


if __name__ == "__main__":

    main()

