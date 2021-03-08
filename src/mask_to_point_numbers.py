import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
    image = cv2.imread("girl.jpg")
    image_mesh = image.copy()
    h, w, c = image.shape
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if results.multi_face_landmarks:
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            #print('face_landmarks:', face_landmarks)
            for idx, coordinate in enumerate(face_landmarks.landmark):
                x = coordinate.x
                y = coordinate.y
                coordinate.x = int(x * w)
                coordinate.y = int(y * h)
                print(int(coordinate.x), int(coordinate.y))
                image_mesh = cv2.circle(image_mesh, (int(coordinate.x) ,int(coordinate.y)), radius=0, color=(0, 0, 255), thickness=3)
                cv2.putText(image_mesh, str(idx), (int(coordinate.x)+5 ,int(coordinate.y)+5), cv2.FONT_HERSHEY_PLAIN, 1.0, cv2.LINE_AA, 1)

            print(face_landmarks.landmark)
            # mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACE_CONNECTIONS,
            #     landmark_drawing_spec=drawing_spec,
            #     connection_drawing_spec=drawing_spec)
        cv2.imwrite('mesh.png', image_mesh)