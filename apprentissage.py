import cv2 # image package
import os # nous executer des commande basique
import numpy as np # mathématique poussé
import pickle  # pickle package

# py -3 -m pip install numpy pickle
while True:
    image_dir = "./clean/" # repo
    current_id = 0 # id a nos images
    label_ids = {} # labels { allan_0: image_1}
    x_train = [] # { entrainenment }
    y_labels = [] # label tester

    for root, dirs, files in os.walk(image_dir):
        if len(files):
            label = root.split("/")[-1]
            for file in files:
                if file.endswith("png"):
                    path = os.path.join(root, file)
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    id_ = label_ids[label]
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # print(image)
                    image = cv2.resize(image, (200, 200))
                    x_train.append(image)
                    y_labels.append(id_)

    with open("labels.pickle", "wb") as f:
        a = pickle.dump(label_ids, f)

    x_train = [cv2.resize(img, (200, 200)) for img in x_train]

    x_train = np.array(x_train)
    y_labels = np.array(y_labels)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(x_train, y_labels)
    recognizer.save("trainner.yml")