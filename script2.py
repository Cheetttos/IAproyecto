import cv2
import face_recognition
import os

# Lista de nombres de las personas en la carpeta "Images"
person_names = []
known_face_encodings = []

# Ruta de la carpeta que contiene las imágenes
images_folder = "Images"

# Iterar a través de las imágenes en la carpeta
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg"):
        person_name = os.path.splitext(filename)[0]
        person_image = face_recognition.load_image_file(os.path.join(images_folder, filename))
        person_face_encoding = face_recognition.face_encodings(person_image)[0]
        person_names.append(person_name)
        known_face_encodings.append(person_face_encoding)



######################################################################################
# Video Streaming

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# ...

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.flip(frame, 1)

    face_locations = face_recognition.face_locations(frame, model="hog")
    if face_locations != []:
        for face_location in face_locations:
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
            results = face_recognition.compare_faces(known_face_encodings, face_frame_encodings)
            
            for i, result in enumerate(results):
                if result:
                    text = person_names[i]
                    color = (125, 220, 0)
                    break
                else:
                    text = "Desconocido"
                    color = (50, 50, 255)

            cv2.rectangle(frame, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), color, -1)
            cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
            cv2.putText(frame, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == 27 & 0xFF:
        break
    

# ...


cap.release()
cv2.destroyAllWindows()