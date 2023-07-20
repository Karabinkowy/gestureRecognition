# Przetwarzanie obrazu z kamery poprzez oznaczenie punktów
# charakterystycznych dłoni oraz podanie nazwy wykonywanego gestu wraz z pewnością.
# Przewidywanie na podstawie modelu trenowanego za pomocą wykonania pliku keypoint_classification.py

import csv
import copy
import itertools
import numpy as np

import cv2 as cv
import mediapipe as mp

# Pobranie klasy KeyPointClassifier do wskazania nazwy gestu
from keypoint_classifier import KeyPointClassifier


# Liczba rozpoznawanych gestów dla jednej/dwóch rąk, liczone od 0
# Jeżeli dodano nowy gest, należy zwiększyć te wartości o 1 za każdy nowy gest
# Jeżeli nowy gest do modelu z jedną ręką, to zmienną GESTURES_ONE_HAND
# Jeżeli do modelu z dwiema rękami, to zmienną GESTURES_TWO_HANDS
GESTURES_ONE_HAND = 13
GESTURES_TWO_HANDS = 4

# WAŻNE: Może być konieczna zmiana lokalizacji na globalną

# Lokalizacja plików z zapisanymi współrzędnymi punktów do nauki modelu dla jednej/dwóch rąk
# Plik utworzony poprzez naciskanie #k przy wykonywaniu żądanego gestu przy odpowiednim indeksie ustawionym na suwaku
# Kolejność zapisywania gestów i ich indeksy zgodnie z plikiem z nazwami keypoint_classifier_label.csv
DATASET_ONE_HAND = 'model/keypoint_classifier/keypoint.csv'
DATASET_TWO_HANDS = 'model2/keypoint_classifier2/keypoint2.csv'

# Lokalizacja plików z zapisanymi współrzędnymi punktów do nauki modelu dla jednej/dwóch rąk
# Plik utworzony poprzez naciskanie przycisku na klawiaturze #k przy wykonywaniu żądanego gestu przy odpowiednim indeksie ustawionym na suwaku
# Kolejność zapisywania gestów i ich indeksy zgodnie z plikiem z nazwami keypoint_classifier_label.csv
CLASSIFIER_LABEL_ONE_HAND = 'model/keypoint_classifier/keypoint_classifier_label.csv'
CLASSIFIER_LABEL_TWO_HANDS = 'model2/keypoint_classifier2/keypoint_classifier_label2.csv'

# Lokalizacja plików z modelami dla jednej/dwóch rąk utworzona po wykonaniu pliku keypoint_classification.py
MODEL_PATH_ONE_HAND = 'model/keypoint_classifier/keypoint_classifier.tflite'
MODEL_PATH_TWO_HANDS = 'model2/keypoint_classifier2/keypoint_classifier2.tflite'

class Recognizer:
    def __init__(
        self,
        # klasa przyjmuje zmienną z liczbą rąk do rozpoznawania
        numberOfHands,
    ):
        self.numberOfHands = numberOfHands
        if self.numberOfHands == 1:
            self.model_path = MODEL_PATH_ONE_HAND
            self.dataset = DATASET_ONE_HAND
            self.classifier_label = CLASSIFIER_LABEL_ONE_HAND
            # ile jest zaplanowanych gestów do rozpoznawania
            self.gestureNumber = GESTURES_ONE_HAND
        elif self.numberOfHands == 2:
            self.model_path = MODEL_PATH_TWO_HANDS
            self.dataset = DATASET_TWO_HANDS
            self.classifier_label = CLASSIFIER_LABEL_TWO_HANDS
            self.gestureNumber = GESTURES_TWO_HANDS

    # zmiany modu z 0 na 1 przez naciśnięcie klawisza #k sprawia, że zapisywane są współrzędne punktów charakterystycznych w danym
    # momencie
    def select_mode(
        self,
        key,
        mode,
    ):
        if key == 107:  # k
            mode = 1
        return mode

    # przetwarzanie współrzędnych punktów charakterystycznych wykrytych przez moduł Mediapipe; prztwarzanie na piksele
    # a dalej normalizacja
    def calc_landmark_list(
        self,
        image,
        landmarks,
    ):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(
        self,
        landmark_list,
    ):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    # funkcja zapisująca dane położenie keypointów do pliku keypoint.csv
    # użytkownik wykonał gest, nacisnął przycisk, a poniższa funkcja zapisuje do pliku keypoint.csv
    def logging_csv(
        self,
        number,
        mode,
        landmark_list,
        dataset,
    ):
        if mode == 0:
            pass
        if mode == 1:
            with open(dataset, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        return

    # oznaczanie punktów charakterystycznych okręgami i łączenie ich odcinkami
    def draw_landmarks(
        self,
        image,
        landmark_point,
    ):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # 手首1
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # 手首2
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # 親指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # 親指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # 親指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # 人差指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # 人差指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # 人差指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # 人差指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # 中指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # 中指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # 中指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # 中指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # 薬指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # 薬指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # 薬指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # 薬指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # 小指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # 小指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # 小指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # 小指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    # umieszczenie w lewym górnym rogu etykiety z nazwą wykrytego gestu oraz dokładnością przewidywania
    def draw_info_text(
        self,
        image,
        gesture_name,
        gesture_accuracy,
    ):
        cv.putText(image, gesture_name, (10, 50), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(image, gesture_accuracy, (450, 50), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)

        return image

    # przetworzenie zdjęcia:
    def editImage(
        self,
    ):
        def nothing(x):
            pass

        # Przygotowanie okna wraz z suwakiem do wybierania indeksu zapisywanego gestu do pliku keypoint.csv (wykorzystywanego dalej do trenowania)
        # oraz uruchomienie kamery laptopa
        cv.namedWindow('Gesture recognition')
        cv.createTrackbar('index', 'Gesture recognition', 0, self.gestureNumber, nothing)
        img = np.zeros((480, 640, 3), np.uint8)
        cv.imshow('Gesture recognition', img)
        cap = cv.VideoCapture(0)

        # Wczytanie modelu do wskazywania punktów charakterystycznych dłoni
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=int(self.numberOfHands), min_detection_confidence=0.7)

        # Obiekt do rozpoznawania gestu
        keypoint_classifier = KeyPointClassifier(self.model_path)

        # Odczytywanie nazwy gestu
        with open(self.classifier_label,
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]

        # Zmienna 'mode' odpowiada za tryb działania, mode = 0 to tryb rozpozawania gestu
        # po wciśnięciu klawisza 'k' mode = 1
        mode = 0
        while True:
            # Odczytaj numer indeksu gestu, który ma być dodany do pliku z danymi do trenowania modelu
            # po naciśnięciu przycisku #k
            gestureIndex = cv.getTrackbarPos('index', 'Gesture recognition')
            # Naciśnięcia ESC powoduje opuszczenie programu
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            mode = self.select_mode(key, mode)

            # Camera capture #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            #  przetawrzanie znalezionych punktów, jeżeli moduł MediaPipe coś wykrył
            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Landmark calculation
                    # Przetworzenie współrzędnych na piksele
                    landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = self.pre_process_landmark(
                        landmark_list)
                    # Write to the dataset file
                    # funkcja możliwiająca po naciśnięciu przycisku #k na klawiaturze zapisanie współrzędnych punktów
                    # charakterystycznych w pliku keypoint.csv, który będzie służyć dalej do trenowania modelu

                    self.logging_csv(gestureIndex, mode, pre_processed_landmark_list, self.dataset)

                    # Hand sign classification
                    # funkcja służąca do znalezienia identyfikatora gestu, który ma największe prawdopodobieństwo na wystąpienie
                    hand_sign_id, hand_sign_accuracy = keypoint_classifier(pre_processed_landmark_list)

                    # Drawing part
                    # Rysowanie punktów charakterystycznych
                    debug_image = self.draw_landmarks(debug_image, landmark_list)

                # Umieszczenie w rogu ekranu nazwy rozpoznanego gestu wraz z pewnością
                debug_image = self.draw_info_text(debug_image, keypoint_classifier_labels[hand_sign_id], hand_sign_accuracy)
            # Screen reflection #############################################################
            cv.imshow('Gesture recognition', debug_image)

        cap.release()
        cv.destroyAllWindows()


# Wpisujemy tu '1' jeżeli chcemy analizowanie jednej ręki, '2' jeżeli dwóch
while True:
    print("Dla ilu rąk należy wykorzystać model? Wpisz '1' albo '2'")
    numberOfHands = input()
    numberOfHands = int(numberOfHands)
    if numberOfHands == 1:
        gestureRecognizer = Recognizer(1)
        break
    elif numberOfHands == 2:
        gestureRecognizer = Recognizer(1)
        break
# Wykonywane jest działanie programu
gestureRecognizer.editImage()