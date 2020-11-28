from deepface import DeepFace
from deepface.basemodels import Facenet
import cv2
import numpy as np
import dlib

# TO-DO

# 1) На фотке с компа рисовать лендмарки
# 2) Разбить все на функции
# 3) Юзать dlib?


def video_recognition():


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
        print("Loading...")
        # создается модель
        face_model = Facenet.loadModel()
        #print(face_model.summary())
        cap = cv2.VideoCapture(0)
        grab, frame = cap.read()
        win1 = dlib.image_window()
        while True:
            grab, original_img = cap.read()

            # --- работа SSD (детекция)---
            ssd_img = original_img.copy()
            ssd_img = cv2.resize(ssd_img, (500, 500))
            detected_face = DeepFace.detectFace(ssd_img, detector_backend='ssd')
            cv2.imshow('Output', detected_face)

            # --- КОСТЫЛЬ работа dlib (распознавание)---
            dlib_img = original_img.copy()
            detector = dlib.get_frontal_face_detector()
            sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

            win1.clear_overlay()
            win1.set_image(dlib_img)
            dets = detector(dlib_img, 1)

            for k, d in enumerate(dets):
                shape = sp(dlib_img, d)
                win1.clear_overlay()
                win1.add_overlay(d)
                win1.add_overlay(shape)
                #win1.wait_until_closed()
            #--- работа FaceNet (распознавание)---

            # # фото меняется на подходящий для сети размер
            # facenet_img = original_img.copy()
            # facenet_img = cv2.resize(facenet_img, (160, 160))
            # # создается копия фото с добавлением одной оси
            # facenet_img = np.expand_dims(facenet_img, axis=0)
            # # копия фото помещается в сеть
            # # работает
            # # выводит массив (128,0)
            # predictions = face_model.predict(facenet_img)[0]
            # # не работает
            # #predictions = face_model(facenet_img)
            # print(f'predictions shape is {predictions.shape}')

            key = cv2.waitKey(1) & 0xFF
            # для прекращения работы необходимо нажать клавишу "q"
            if key == ord("q"):
                print("[INFO] process finished by user")
                break
        cap.release()
        cv2.destroyAllWindows()
    # сравнивание видео с камеры с иходным фото
    elif choice == 2:
        print("Loading...")
        photo_path = "faces/me.jpg"
        original_img = cv2.imread(photo_path)
        # уменьшение размера фото, чтобы сеть могла его обработать
        original_img = cv2.resize(original_img, (160, 160))
        cv2.imshow('original_img', original_img)
        # увеличение размерности фото
        original_img = np.expand_dims(original_img, axis=0)
        cap = cv2.VideoCapture(0)
        grab, frame = cap.read()
        grab, compared_img = cap.read()
        while True:
            grab, compared_img = cap.read()
            # уменьшение размера фото, чтобы сеть могла его обработать
            compared_img = cv2.resize(compared_img, (160, 160))
            cv2.imshow('compared_img', compared_img)
            # увеличение размерности фото
            compared_img = np.expand_dims(compared_img, axis=0)
            # здесь появляется ошибка в файле functions/detect_face (строка 204)
            # потому что если размерность изображения - 4, то resize() не работает!
            ver_result = DeepFace.verify(original_img, compared_img, distance_metric='euclidean', model_name='Facenet', detector_backend='ssd')
            for k, v in ver_result.items():
                print(f'{k} : {v}')
            key = cv2.waitKey(1) & 0xFF
            # для прекращения работы необходимо нажать клавишу "q"
            if key == ord("q"):
                print("[INFO] process finished by user")
                break
        cap.release()
        cv2.destroyAllWindows()

# распознавание фото с компа и рисование всех признаков
elif choice == 2:
    img = cv2.imread('faces/big_linus.jpg')
    img = cv2.resize(img, (500, 500))
    cv2.imshow('Image to process', img)
    cv2.waitKey(0)
    detected_face = DeepFace.detectFace(img, detector_backend='ssd')
    cv2.imshow('Detected face', detected_face)
    cv2.waitKey(0)

else:
    raise ValueError("Unknown command!")