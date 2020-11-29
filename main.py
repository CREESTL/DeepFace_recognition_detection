from deepface import DeepFace
from deepface.basemodels import Facenet
import cv2
import numpy as np
import dlib

# TO-DO

# 1) На фотке с компа рисовать лендмарки
# 2) Разбить все на функции
# 3) Юзать dlib?


# def video_recognition():


print("Would you like to stream video from webcam or take an image from you PC?")
print("1) Web-cam")
print("2) Single image")
choice = int(input())

# стрим с камеры
if choice == 1:
    print("Would you like to do face recognition or verification?")
    print("1) Recognition")
    print("2) Verification")
    choice = int(input())
    # только распознавание с камеры и рисование всех признаков
    if choice == 1:
        dlib_choice = input("Activate dlib? (y/n) - ")
        if dlib_choice == 'y':
            print("Loading FaceNet...")
            # создается модель
            face_model = Facenet.loadModel()
            print("Loading dlib...")
            face_detector = dlib.get_frontal_face_detector()
            shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            #print(face_model.summary())
            cap = cv2.VideoCapture(0)
            grab, frame = cap.read()
            while True:
                grab, original_img = cap.read()

                # --- работа SSD (детекция)---
                ssd_img = original_img.copy()
                detected_face = DeepFace.detectFace(ssd_img, detector_backend='ssd')

                # --- работа dlib (распознавание)---
                dlib_img = original_img.copy()
                dets = face_detector(dlib_img, 1)

                # рисование лендмарок
                for det in dets:
                    shape = shape_predictor(dlib_img, det)
                    points = []
                    for i in range(68):
                        point = (shape.part(i).x, shape.part(i).y)
                        # рисуется на фото с боксом!
                        cv2.circle(detected_face, point, 1, (255, 255, 0), 1)

                cv2.imshow("Output", detected_face)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] process finished by user")
                    break

        elif dlib_choice == 'n':
            print("Loading FaceNet...")
            # создается модель
            face_model = Facenet.loadModel()
            cap = cv2.VideoCapture(0)
            grab, frame = cap.read()
            while True:
                grab, original_img = cap.read()

                # --- работа SSD (детекция)---
                ssd_img = original_img.copy()
                detected_face = DeepFace.detectFace(ssd_img, detector_backend='ssd')
                cv2.imshow("Output", detected_face)

                # для прекращения работы необходимо нажать клавишу "q"
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] process finished by user")
                    break

        else:
            raise ValueError("Unknown command!")

        cap.release()
        cv2.destroyAllWindows()

    # сравнивание видео с камеры с иходным фото
    elif choice == 2:
        # faces/me.jpg
        photo_path = input("Please provide a full path to the photo for comparison:\n")
        print("Loading...")
        original_img = cv2.imread(photo_path)
        # уменьшение размера фото, чтобы сеть могла его обработать
        original_img_copy = cv2.resize(original_img.copy(), (500, 500))
        cv2.imshow('Original image', original_img_copy)
        original_img = cv2.resize(original_img.copy(), (160, 160))
        # увеличение размерности фото
        original_img = np.expand_dims(original_img, axis=0)
        cap = cv2.VideoCapture(0)
        grab, frame = cap.read()
        grab, compared_img = cap.read()
        while True:
            grab, compared_img = cap.read()
            # уменьшение размера фото, чтобы сеть могла его обработать
            cv2.imshow('Compared image', compared_img)
            compared_img = cv2.resize(compared_img.copy(), (160, 160))
            # увеличение размерности фото
            compared_img = np.expand_dims(compared_img, axis=0)
            # здесь появляется ошибка в файле functions/detect_face (строка 204)
            # потому что если размерность изображения - 4, то resize() не работает!
            print("Verifying two images...")
            ver_result = DeepFace.verify(original_img, compared_img, distance_metric='euclidean', model_name='Facenet', detector_backend='ssd')
            verified = bool(ver_result['verified'])
            if not verified:
                print("[RESULT] - These are different persons!")
            else:
                print("[RESULT]- Same person")

            key = cv2.waitKey(1) & 0xFF
            # для прекращения работы необходимо нажать клавишу "q"
            if key == ord("q"):
                print("[INFO] process finished by user")
                break
        cap.release()
        cv2.destroyAllWindows()

# распознавание фото с компа и рисование всех признаков
elif choice == 2:
    # faces/big_linus.jpg'
    photo_path = input("Please provide a full path to the photo:\n")
    dlib_choice = input("Activate dlib? (y/n) - ")
    if dlib_choice == 'y':
        print("Loading dlib...")
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        img = cv2.imread(photo_path)
        img = cv2.resize(img, (500, 500))
        cv2.imshow('Image to process', img)
        detected_face = DeepFace.detectFace(img, detector_backend='ssd')

        # --- работа dlib (распознавание)---
        dlib_img = img.copy()
        dets = face_detector(dlib_img, 1)
        # рисование лендмарок
        for det in dets:
            shape = shape_predictor(dlib_img, det)
            points = []
            for i in range(68):
                point = (shape.part(i).x, shape.part(i).y)
                # рисуется на фото с боксом!
                cv2.circle(detected_face, point, 1, (255, 255, 0), 1)

        cv2.imshow('Detected face', detected_face)
        cv2.waitKey(0)
    elif dlib_choice == 'n':
        img = cv2.imread(photo_path)
        img = cv2.resize(img, (500, 500))
        cv2.imshow('Image to process', img)
        detected_face = DeepFace.detectFace(img, detector_backend='ssd')
        cv2.imshow('Detected face', detected_face)
        cv2.waitKey(0)
    else:
        raise ValueError("Unknown command!")
else:
    raise ValueError("Unknown command!")
