import cv2
import os
import transform_images

contents = os.listdir("images")
input = input("entrer le prénom de la personne: ")
input = input.lower()

if input in contents:
    path = input
else:
    os.mkdir('images/'+input)
    path = input

# on charge le model
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# allumer la camera
# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture('video.mp4')

# photo size
c = 50
# photo id
id = 0

while True:
    # on charge l'image de la caméra
    ret, frame = capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # on detecte l'image
    face = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(c, c))
    # on boucle pour prendre des photos et on stocke
    for (x, y, w, h) in face:
        face_img_path ="images/"+path+"/p-{:d}.png".format(id)
        cv2.imwrite(face_img_path, frame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        transformed_img_path = "images/"+path+"/transformed-p-{:d}.png".format(id)
        transform_images.transform_images(face_img_path, transformed_img_path)

        id += 1
    # on affiche l'image
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("q"):
    #     break
    # if key == ord("a"):
    #     for cpt in range(100):
    #         ret, frame = capture.read()

capture.release()
cv2.destroyAllWindows()
