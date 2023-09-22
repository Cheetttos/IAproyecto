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

# Video Streaming
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
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

    # Esperar por eventos del teclado
    key = cv2.waitKey(1)

    # Tecla 's' para guardar la imagen actual en la carpeta "Images" con un nombre personalizado
    if key == ord('s'):
        user_input = input("Ingrese el nombre de la persona: ")
        if user_input:
            img_name = os.path.join(images_folder, f"{user_input}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Imagen guardada como {img_name}")
            person_image = face_recognition.load_image_file(img_name)
            person_face_encoding = face_recognition.face_encodings(person_image)[0]
            person_names.append(user_input)
            known_face_encodings.append(person_face_encoding)

    # Tecla 'q' para salir de la aplicación
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
