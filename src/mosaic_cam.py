import cv2

# 얼굴 검출을 위한 Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠 시작
cap = cv2.VideoCapture(0)
rate = 15               # 모자이크 축소 비율
win_title = 'Face Mosaic'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환 (얼굴 인식 정확도 향상)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face = frame[y:y+h, x:x+w]

        # 축소 → 확대 (모자이크 효과)
        small = cv2.resize(face, (w // rate, h // rate))
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)

        # 모자이크 얼굴을 원본에 덮어쓰기
        frame[y:y+h, x:x+w] = mosaic

    # 화면에 출력
    cv2.imshow(win_title, frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
