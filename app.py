import cv2
import math
import mediapipe as mp
import numpy as np

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(x1, y1, x2, y2):
    theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    return int(180 / math.pi * theta)

def send_warning():
    pass

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'blue': (255, 127, 0),
            'red': (50, 50, 255),
            'green': (127, 255, 0),
            'light_green': (127, 233, 100),
            'yellow': (0, 255, 255),
            'pink': (255, 0, 255)
        }
        self.good_frames = 0
        self.bad_frames = 0
        self.aligned_frames = 0
        self.total_frames = 0
        self.neck_angles = []
        self.torso_angles = []

    def process_frame(self, frame):
        self.total_frames += 1
        h, w = frame.shape[:2]
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and get the pose landmarks
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract key points
            l_shldr = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                       int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            r_shldr = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            l_ear = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y * h))
            l_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h))
            
            # Calculate metrics
            offset = calculate_distance(*l_shldr, *r_shldr)
            neck_inclination = calculate_angle(*l_shldr, *l_ear)
            torso_inclination = calculate_angle(*l_hip, *l_shldr)
            
            # Store angles for averaging
            self.neck_angles.append(neck_inclination)
            self.torso_angles.append(torso_inclination)
            
            # Check alignment
            if offset < 100:
                cv2.putText(frame, f'{int(offset)} Good Posture', (w - 240, 30), self.font, 0.7, self.colors['green'], 2)
                self.aligned_frames += 1
            else:
                cv2.putText(frame, f'{int(offset)} Bad Posture', (w - 240, 30), self.font, 0.7, self.colors['red'], 2)
            
            # Check posture
            if neck_inclination < 40 and torso_inclination < 10:
                self.good_frames += 1
                color = self.colors['light_green']
            else:
                self.bad_frames += 1
                color = self.colors['red']
            
            # Draw posture lines and angles
            self.draw_posture_lines(frame, l_shldr, r_shldr, l_ear, l_hip, color)
            self.draw_posture_angles(frame, l_shldr, l_hip, neck_inclination, torso_inclination, color)
            
            # Display posture time
            self.display_posture_time(frame, h)
        
        # Display analytics in overlay box
        self.display_analytics(frame)
        
        return frame

    def draw_posture_lines(self, frame, l_shldr, r_shldr, l_ear, l_hip, color):
        cv2.line(frame, l_shldr, l_ear, color, 4)
        cv2.line(frame, l_shldr, (l_shldr[0], l_shldr[1] - 100), color, 4)
        cv2.line(frame, l_hip, l_shldr, color, 4)
        cv2.line(frame, l_hip, (l_hip[0], l_hip[1] - 100), color, 4)
        
        # Draw key points
        for point in [l_shldr, r_shldr, l_ear, l_hip]:
            cv2.circle(frame, point, 7, self.colors['yellow'], -1)

    def draw_posture_angles(self, frame, l_shldr, l_hip, neck_inclination, torso_inclination, color):
        cv2.putText(frame, f'Neck: {neck_inclination}  Torso: {torso_inclination}', (10, 30), self.font, 0.9, color, 2)
        cv2.putText(frame, str(int(neck_inclination)), (l_shldr[0] + 10, l_shldr[1]), self.font, 0.9, color, 2)
        cv2.putText(frame, str(int(torso_inclination)), (l_hip[0] + 10, l_hip[1]), self.font, 0.9, color, 2)

    def display_posture_time(self, frame, h):
        fps = 30  # Assuming 30 fps, adjust if different
        good_time = self.good_frames / fps
        bad_time = self.bad_frames / fps
        
        if good_time > bad_time:
            time_string = f'Time: {good_time:.1f}s'
            color = self.colors['green']
        else:
            time_string = f'Time: {bad_time:.1f}s'
            color = self.colors['red']
        
        cv2.putText(frame, time_string, (10, h - 20), self.font, 0.7, color, 2)
        
        if bad_time > 180:
            send_warning()

    def display_analytics(self, frame):
        analytics = np.zeros((150, 300, 3), dtype=np.uint8)
        cv2.putText(analytics, f"Neck Angle: {self.get_avg_neck_angle():.1f}", (10, 30), self.font, 0.7, self.colors['yellow'], 2)
        cv2.putText(analytics, f"Torso Angle: {self.get_avg_torso_angle():.1f}", (10, 60), self.font, 0.7, self.colors['yellow'], 2)
        cv2.putText(analytics, f"Alignment perc: {self.get_alignment_percentage():.1f}%", (10, 90), self.font, 0.7, self.colors['yellow'], 2)
        cv2.putText(analytics, f"Posture perc: {self.get_good_posture_percentage():.1f}%", (10, 120), self.font, 0.7, self.colors['yellow'], 2)
        
        frame[10:160, 10:310] = analytics

    def get_avg_neck_angle(self):
        return np.mean(self.neck_angles) if self.neck_angles else 0

    def get_avg_torso_angle(self):
        return np.mean(self.torso_angles) if self.torso_angles else 0

    def get_alignment_percentage(self):
        return (self.aligned_frames / self.total_frames) * 100 if self.total_frames > 0 else 0

    def get_good_posture_percentage(self):
        return (self.good_frames / self.total_frames) * 100 if self.total_frames > 0 else 0

def main():
    cap = cv2.VideoCapture("video.mp4")  
    analyzer = PostureAnalyzer()

    # Meta
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Video writer
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame = analyzer.process_frame(frame)
        video_output.write(processed_frame)
        cv2.namedWindow('Posture Analysis', cv2.WINDOW_NORMAL)
        cv2.imshow('Posture Analysis', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
