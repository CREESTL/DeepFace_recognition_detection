from deepface import DeepFace
from deepface.basemodels import Facenet
import cv2
import numpy as np
import dlib


'''
Функция распознавания лица на указанном фото
'''
def photo_recognition(use_dlib=False):
    if use_dlib:
        # распознавание с использованием dlib
        print("Loading dlib...")
        # создание детекторов
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        img = cv2.imread(photo_path)
        img = cv2.resize(img, (500, 500))
        cv2.imshow('Image to process', img)
        # распознавание лица
        # ВНИМАНИЕ! Функция detectFace была изменена мной вручную. Сейчас она возвращает фото с наложенным на него
        # боксом распознанного лица. По умолчанию detectFace не рисует бокс!
        detected_face = DeepFace.detectFace(img, detector_backend='ssd')

        # --- работа dlib (распознавание)---
        dlib_img = img.copy()
        dets = face_detector(dlib_img, 1)
        # рисование лендмарок
        for det in dets:
            shape = shape_predictor(dlib_img, det)
            for i in range(68):
                point = (shape.part(i).x, shape.part(i).y)
                # рисуется на фото с боксом!
                cv2.circle(detected_face, point, 1, (255, 255, 0), 1)
        # вывод результата
        cv2.imshow('Detected face', detected_face)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("[INFO] process finished by user")
    else:
        # детектированиетолько с ипользованием DeepFace
        img = cv2.imread(photo_path)
        img = cv2.resize(img, (500, 500))
        cv2.imshow('Image to process', img)
        # распознавание лица
        # ВНИМАНИЕ! Функция detectFace была изменена мной вручную. Сейчас она возвращает фото с наложенным на него
        # боксом распознанного лица. По умолчанию detectFace не рисует бокс!
        detected_face = DeepFace.detectFace(img, detector_backend='ssd')
        cv2.imshow('Detected face', detected_face)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("[INFO] process finished by user")


'''
Функция распознавания лица на видеопотоке с веб-камеры
'''
def video_recognition(use_dlib=False):
    if use_dlib:
        # распознавание с использованием dlib
        print("Loading dlib...")
        # создание детекторов
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        cap = cv2.VideoCapture(0)
        while True:
            grab, original_img = cap.read()

            # --- работа SSD (детектирование)---
            ssd_img = original_img.copy()
            # распознавание лица
            # ВНИМАНИЕ! Функция detectFace была изменена мной вручную. Сейчас она возвращает фото с наложенным на него
            # боксом распознанного лица. По умолчанию detectFace не рисует бокс!
            detected_face = DeepFace.detectFace(ssd_img, detector_backend='ssd')

            # --- работа dlib (распознавание)---
            dlib_img = original_img.copy()
            dets = face_detector(dlib_img, 1)

            # рисование лендмарок
            for det in dets:
                shape = shape_predictor(dlib_img, det)
                for i in range(68):
                    point = (shape.part(i).x, shape.part(i).y)
                    # рисуется на фото с боксом!
                    cv2.circle(detected_face, point, 1, (255, 255, 0), 1)
            # вывод результата
            cv2.imshow("Output", detected_face)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] process finished by user")
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        # детектирование только с ипользованием DeepFace
        cap = cv2.VideoCapture(0)
        while True:
            grab, original_img = cap.read()

            # --- работа SSD (детекция)---
            ssd_img = original_img.copy()
            # распознавание лица
            # ВНИМАНИЕ! Функция detectFace была изменена мной вручную. Сейчас она возвращает фото с наложенным на него
            # боксом распознанного лица. По умолчанию detectFace не рисует бокс!
            detected_face = DeepFace.detectFace(ssd_img, detector_backend='ssd')
            cv2.imshow("Output", detected_face)

            # для прекращения работы необходимо нажать клавишу "q"
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] process finished by user")
                break
        cap.release()
        cv2.destroyAllWindows()

'''
Функция сравнения указанной фотографии с видео с веб-камеры
'''
def video_verification():
    # faces/me.jpg
    photo_path = input("Please provide a full path to the photo for comparison:\n")
    print("Loading...")
    original_img = cv2.imread(photo_path)
    original_img_copy = cv2.resize(original_img.copy(), (500, 500))
    cv2.imshow('Original image', original_img_copy)
    # уменьшение размера фото, чтобы сеть могла его обработать
    original_img = cv2.resize(original_img.copy(), (160, 160))
    # увеличение размерности фото
    original_img = np.expand_dims(original_img, axis=0)
    cap = cv2.VideoCapture(0)
    while True:
        grab, compared_img = cap.read()
        cv2.imshow('Compared image', compared_img)
        # уменьшение размера фото, чтобы сеть могла его обработать
        compared_img = cv2.resize(compared_img.copy(), (160, 160))
        # увеличение размерности фото
        compared_img = np.expand_dims(compared_img, axis=0)
        print("Verifying two images...")
        # сравнение исходного фото с кадром с камеры
        ver_result = DeepFace.verify(original_img, compared_img, distance_metric='euclidean', model_name='Facenet',
                                     detector_backend='ssd')
        # вывод результата
        verified = bool(ver_result['verified'])
        if not verified:
            print("[RESULT] - These are different persons!")
        else:
            print("[RESULT]- Same person")
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] process finished by user")
            break
    cap.release()
    cv2.destroyAllWindows()






########################################################################################################################
print("Would you like to stream video from webcam or take an image from you PC?")
print("1) Web-cam")
print("2) Single image")
choice = int(input())
# обработка видео с камеры
if choice == 1:
    print("Would you like to do face recognition or verification?")
    print("1) Recognition")
    print("2) Verification")
    choice = int(input())
    if choice == 1:
        dlib_choice = input("Activate dlib? (y/n) - ")
        if dlib_choice == 'y':
            video_recognition(use_dlib=True)
        elif dlib_choice == 'n':
            video_recognition(use_dlib=False)
        else:
            raise ValueError("Unknown command!")
    elif choice == 2:
        video_verification()
elif choice == 2:
    # faces/big_linus.jpg'
    photo_path = input("Please provide a full path to the photo:\n")
    dlib_choice = input("Activate dlib? (y/n) - ")
    if dlib_choice == 'y':
       photo_recognition(use_dlib=True)
    elif dlib_choice == 'n':
        photo_recognition(use_dlib=False)
    else:
        raise ValueError("Unknown command!")
else:
    raise ValueError("Unknown command!")
