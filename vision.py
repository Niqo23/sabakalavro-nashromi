import cv2                      # OpenCV – ვიდეო ნაკადისა და გამოსახულების დასამუშავებლად
import mediapipe as mp          # Mediapipe – ადამიანის პოზის (pose) ტრეკინგისთვის
import numpy as np              # NumPy – კუთხის გამოსათვლელი ვექტორული მათემატიკისთვის

#  კუთხის გამოთვლის ფუნქცია სამი წერტილის საფუძველზე
def calculate_angle(a, b, c):
    # წერტილები ვექტორებად გარდაიქმნება (NumPy array)
    a, b, c = np.array(a), np.array(b), np.array(c)

    # arctangent ფუნქციით გამოითვლება ორი ვექტორის შორის კუთხე
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # თუ კუთხე > 180°, გადავაბრუნოთ – მივიღოთ სწორი კუთხე (0–180° შუალედი)
    return 360 - angle if angle > 180 else angle

# ️ ძირითადი ანალიზის ფუნქცია – shoulder press შეფასებისთვის
def analyze_shoulder_press_strict(landmarks, image, counter, stage):
    # ამოვიღოთ პოზიციური წერტილები მარჯვენა და მარცხენა მხარისთვის
    right_shoulder = [landmarks[12].x, landmarks[12].y]
    right_elbow    = [landmarks[14].x, landmarks[14].y]
    right_wrist    = [landmarks[16].x, landmarks[16].y]

    left_shoulder = [landmarks[11].x, landmarks[11].y]
    left_elbow    = [landmarks[13].x, landmarks[13].y]
    left_wrist    = [landmarks[15].x, landmarks[15].y]

    # გამოვთვალოთ იდაყვის კუთხეები მარცხენა და მარჯვენა ხელისთვის
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    #  თუ ორივე იდაყვი გაშლილია (≥160°) და ფაზა იყო ქვედა → ჩაითვალოს 1 რეპი
    if right_angle >= 160 and left_angle >= 160 and stage == 'down':
        stage = 'up'
        counter += 1

    #  თუ რომელიმე იდაყვი <70°, ნიშნავს რომ მოძრაობა ქვედა ფაზაშია
    if right_angle < 70 or left_angle < 70:
        stage = 'down'

    feedback_r = ""
    feedback_l = ""

    #  ვიზუალური უკუკავშირი – ხაზებისა და ტექსტების გამოსატანი ფუნქცია
    def draw_lines_and_feedback(img, a, b, c, angle, label):
        pts = [a, b, c]
        for i in range(2):
            p1 = tuple(np.multiply(pts[i], [img.shape[1], img.shape[0]]).astype(int))
            p2 = tuple(np.multiply(pts[i+1], [img.shape[1], img.shape[0]]).astype(int))

            # თუ კუთხე არ არის ნორმალურ დიაპაზონში (70° < X < 180°), ხაზია წითელი
            if angle <= 70 or angle >= 180:
                color = (0, 0, 255)  # წითელი
            else:
                color = (0, 255, 0)  # მწვანე

            # ხაზები წერტილებს შორის
            cv2.line(img, p1, p2, color, 4)
            # კუთხის მნიშვნელობა (ტექსტურად)
            cv2.putText(img, f"{label}: {int(angle)}°", (p1[0], p1[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return color

    # ორი ხელისთვის გამოვიძახოთ ხაზების დახატვა და შეფასება
    color_r = draw_lines_and_feedback(image, right_shoulder, right_elbow, right_wrist, right_angle, "Right")
    color_l = draw_lines_and_feedback(image, left_shoulder, left_elbow, left_wrist, left_angle, "Left")

    # ტექსტური შეფასება (feedback)
    feedback_r = "Right arm good" if 70 < right_angle < 180 else "Right arm out of range"
    feedback_l = "Left arm good" if 70 < left_angle < 180 else "Left arm out of range"

    return [feedback_r, feedback_l], counter, stage

#  Mediapipe-ის ინსტანციები პოზის ამოსაცნობად
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# რეპების მთვლელი და ფაზის კონტროლი
counter = 0
stage = None

# კამერის ჩართვა (0 = default webcam)
cap = cv2.VideoCapture(0)

# Mediapipe-ის პოზის დადგენის მოდული
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    # ციკლი კამერიდან კადრების წასაკითხად
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # კადრის კონვერტაცია RGB -> BGR და reverse
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # თუ მოიძებნა pose landmarks (ანუ პოზის წერტილები)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            feedback, counter, stage = analyze_shoulder_press_strict(lm, image, counter, stage)

            # ხაზავს ადამიანის სკელეტს
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ეკრანზე აჩვენებს ტექსტურ უკუკავშირს
            y0 = 30
            for line in feedback:
                color = (0, 255, 0) if "good" in line.lower() else (0, 0, 255)
                cv2.putText(image, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y0 += 30

            # რეპების რაოდენობის გამოტანა
            cv2.putText(image, f"Reps: {counter}", (10, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # შედეგის ჩვენება OpenCV ფანჯარაში
        cv2.imshow('Shoulder Press Strict Tracker', image)

        # თუ დააჭერს 'q' – პროგრამა დასრულდება
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# კამერის გათავისუფლება და ფანჯრების დახურვა
cap.release()
cv2.destroyAllWindows()
