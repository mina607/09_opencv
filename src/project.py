import cv2
import dlib
from scipy.spatial import distance as dist

cap = cv2.VideoCapture(0)

# EAR 계산 함수
def calculate_EAR(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# 눈 좌표 추출 함수
def extract_eye_points(landmarks, eye_indices):
    return [(landmarks.part(n).x, landmarks.part(n).y) for n in eye_indices]

def main():
    # 상수 및 변수
    EAR_THRESHOLD = 0.20
    CLOSED_EYES_FRAME = 48
    frame_counter = 0

    LEFT_EYE_IDX = list(range(42, 48))
    RIGHT_EYE_IDX = list(range(36, 42))

    # dlib 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # 랜드마크 그리기
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # EAR 계산
            left_eye = extract_eye_points(landmarks, LEFT_EYE_IDX)
            right_eye = extract_eye_points(landmarks, RIGHT_EYE_IDX)
            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # EAR 출력
            cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0,
