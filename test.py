import cv2
from utils.predict import predict_activity

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    activity, conf = predict_activity(frame)

    cv2.putText(frame, f"{activity} ({conf:.2f})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("HAR", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()