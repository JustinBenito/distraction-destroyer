import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh

disturbed = False

cap = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh() as face_mesh:

    closed_eye_counter = 0
    nose_3d = np.array([0, 0, 0], dtype=np.float64)
    nose_2d = np.array([0, 0])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        face_results = face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -8:
                    text = "looking left"
                    disturbed = True
                    cv2.putText(frame, "Get Back to Work", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                elif y > 8:
                    text = "looking right"
                    disturbed = True
                    cv2.putText(frame, "Get Back to Work", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                elif x < -8:
                    text = "looking down"
                    cv2.putText(frame, "Get Back to Work", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                elif x > 8:
                    text = "Looking up"
                    disturbed = True
                    cv2.putText(frame, "Get Back to Work", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                else:
                    text = "Forward"
                    disturbed = False


        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:

                left_eye_landmarks = [landmarks.landmark[i] for i in range(159, 145, -1)]
                right_eye_landmarks = [landmarks.landmark[i] for i in range(386, 374, -1)]


                left_eye_ear = (cv2.norm(left_eye_landmarks[1].x - left_eye_landmarks[5].x, left_eye_landmarks[1].y - left_eye_landmarks[5].y) +
                                cv2.norm(left_eye_landmarks[2].x - left_eye_landmarks[4].x,left_eye_landmarks[2].y - left_eye_landmarks[4].y)) / (2 * cv2.norm(left_eye_landmarks[0].x - left_eye_landmarks[3].x, left_eye_landmarks[0].y - left_eye_landmarks[3].y))

                right_eye_ear = (cv2.norm(right_eye_landmarks[1].x - right_eye_landmarks[5].x, right_eye_landmarks[1].y - right_eye_landmarks[5].y) + cv2.norm(right_eye_landmarks[2].x - right_eye_landmarks[4].x, right_eye_landmarks[2].y - right_eye_landmarks[4].y)) / (2 * cv2.norm(right_eye_landmarks[0].x - right_eye_landmarks[3].x, right_eye_landmarks[0].y - right_eye_landmarks[3].y))

                if left_eye_ear < 0.2 or right_eye_ear < 0.2:
                    closed_eye_counter += 1
                else:
                    closed_eye_counter = 0

                if closed_eye_counter > 10: 
                    cv2.putText(frame, "Get Back to Work", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                    disturbed = True
                
        cv2.imshow("Frame", frame)
        print(disturbed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
