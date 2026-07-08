import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt, welch

class RPPGEstimator:
    def __init__(self, fps=30, window_size=300):
        """
        Initializes the rPPG Pipeline.
        :param fps: Sampling rate of your camera frame rate (Default: 30)
        :param window_size: Number of frames to look back for calculation (e.g., 300 frames = 10 seconds)
        """
        self.fps = fps
        self.window_size = window_size
        
        # Buffer arrays for raw RGB signals
        self.r_buffer = []
        self.g_buffer = []
        self.b_buffer = []
        
        # Initialize MediaPipe Face Landmarker
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Forehead Landmark Indices from MediaPipe Face Mesh Map
        self.forehead_indices = [68, 107, 9, 336, 298, 337, 8, 108, 69, 105, 10, 334]

    def extract_roi_mean(self, frame):
        """
        Detects face and calculates spatial average of RGB values in the forehead ROI.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Gather pixel coordinates of the forehead polygon
        polygon_points = []
        for idx in self.forehead_indices:
            pt = landmarks[idx]
            cx, cy = int(pt.x * w), int(pt.y * h)
            polygon_points.append([cx, cy])
            
        polygon_points = np.array(polygon_points, dtype=np.int32)
        
        # Create mask for the forehead ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 255)
        
        # Extract mean color channels inside the masked region
        # Note: opencv reads BGR, we convert carefully
        mean_bgr = cv2.mean(frame, mask=mask)[:3]
        
        return mean_bgr, polygon_points

    def pos_algorithm(self, r, g, b):
        """
        Implements Plane-Orthogonal-to-Skin (POS) rPPG Algorithm.
        """
        # Temporal normalization (Window-based)
        H = len(r)
        R = np.array(r)
        G = np.array(g)
        B = np.array(b)
        
        # Mean scaling matrix
        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)
        
        Rn = R / (mean_R + 1e-6)
        Gn = G / (mean_G + 1e-6)
        Bn = B / (mean_B + 1e-6)
        
        # Define the projection matrix axes
        # S = Mat * [Rn, Gn, Bn]^T
        X = 3 * Rn - 2 * Gn
        Y = 1.5 * Rn + 1 * Gn - 1.5 * Bn
        
        # Tuning alpha (standard deviation quotient)
        alpha = np.std(X) / (np.std(Y) + 1e-6)
        
        # The raw pulse signal
        pulse_signal = X - (alpha * Y)
        return pulse_signal

    def butter_bandpass_filter(self, data, lowcut=0.75, highcut=2.5, order=2):
        """
        Applies a Butterworth bandpass filter. 
        Default limits correspond roughly to 45 BPM - 150 BPM.
        """
        nyq = 0.5 * self.fps
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def calculate_bpm(self, filtered_signal):
        """
        Applies FFT to estimate the dominant pulse frequency.
        """
        N = len(filtered_signal)
        # Compute Fast Fourier Transform
        fft_data = np.abs(np.fft.rfft(filtered_signal))
        fft_freqs = np.fft.rfftfreq(N, d=1/self.fps)
        
        # Restrict frequencies strictly inside biological limits (45 to 150 bpm)
        valid_idx = np.where((fft_freqs >= 0.75) & (fft_freqs <= 2.5))[0]
        
        if len(valid_idx) == 0:
            return 0.0
            
        frequencies = fft_freqs[valid_idx]
        power_spectrum = fft_data[valid_idx]
        
        # Peak frequency dictates heart rate
        peak_idx = np.argmax(power_spectrum)
        bpm = frequencies[peak_idx] * 60.0
        return bpm

    def process_frame(self, frame):
        """
        Main entry point logic per frame. Updates buffers and calculates BPM.
        """
        mean_bgr, roi_poly = self.extract_roi_mean(frame)
        
        if mean_bgr is None:
            return "No Face Found", None
            
        # Push to buffers (BGR sequence -> index 2 is Red, 1 is Green, 0 is Blue)
        self.b_buffer.append(mean_bgr[0])
        self.g_buffer.append(mean_bgr[1])
        self.r_buffer.append(mean_bgr[2])
        
        # Keep buffer sizing bounded
        if len(self.r_buffer) > self.window_size:
            self.r_buffer.pop(0)
            self.g_buffer.pop(0)
            self.b_buffer.pop(0)
            
        # We need a fully populated window before computing steady HR output
        if len(self.r_buffer) < self.window_size:
            return f"Buffering... ({len(self.r_buffer)}/{self.window_size})", roi_poly
            
        # Run rPPG pipeline execution
        raw_pulse = self.pos_algorithm(self.r_buffer, self.g_buffer, self.b_buffer)
        filtered_pulse = self.butter_bandpass_filter(raw_pulse)
        estimated_bpm = self.calculate_bpm(filtered_pulse)
        
        return round(estimated_bpm, 1), roi_poly

# --- Main Application Execution Block ---
if __name__ == "__main__":
    # Use 0 for live built-in Webcam feed, or replace with a path string to a video file
    cap = cv2.VideoCapture(0)
    
    # Instantiate pipeline
    rppg = RPPGEstimator(fps=30, window_size=250) 
    
    print("Starting pipeline. Press 'q' to safely terminate standard app loop.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optional: Flip image for natural mirror orientation layout
        frame = cv2.flip(frame, 1)
        
        # Process frame updates through pipeline
        bpm, roi_polygon = rppg.process_frame(frame)
        
        # Visual Annotations UI Rendering
        if roi_polygon is not None:
            cv2.polylines(frame, [roi_polygon], True, (0, 255, 0), 1)
            
        cv2.putText(
            frame, 
            f"Heart Rate: {bpm} BPM", 
            (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255) if isinstance(bpm, float) else (255, 255, 0), 
            2
        )
        
        cv2.imshow("rPPG Engine Testing Environment", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()