from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2


class DetectMask:

    def __init__(self):

        self.model_detector = load_model('model/mask_detector.model')
        self.proto_txt_path = 'deploy.prototxt.txt'
        self.model_path = 'model/res10_300x300_ssd_iter_140000.caffemodel'
        self.face_detector = cv2.dnn.readNetFromCaffe(self.proto_txt_path, self.model_path)
        self.mask_detector = load_model('model/mask_detector.model')
        self.label = "No Mask"

    def DetectFaceMaskFromVideo(self):

        print("Detecting Mask from WebCam")
        cap = cv2.VideoCapture('video/mask.mp4')

        while True:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=756)
            (h, w) = frame.shape[:2]
            # print(frame.shape)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))

            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()

            faces = []
            bbox = []
            results = []

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    faces.append(face)
                    bbox.append((startX, startY, endX, endY))

            if len(faces) > 0:
                results = self.mask_detector.predict(faces)

            for (face_box, result) in zip(bbox, results):
                (startX, startY, endX, endY) = face_box
                (mask, withoutMask) = result

                if mask > withoutMask:
                    self.label = "Mask"
                    color = (0, 255, 0)
                else:
                    self.label = "No Mask"
                    color = (0, 0, 255)

                cv2.putText(frame, self.label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def DetectFaceMaskFromImage(self, filePath):

        import keras
        import cv2
        import matplotlib.pyplot as plt
        model = keras.models.load_model("model/final_model")
        face_classifier = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

        image = cv2.imread(filePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        labels_dict = {0: 'NO MASK', 1: 'MASK'}
        color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

        faces = face_classifier.detectMultiScale(image, 1.3, 5)

        for x, y, w, h in faces:
            face_img = gray[y:y + w, x:x + w]
            resized = cv2.resize(face_img, (100, 100))
            normalized = resized / 255.
            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            result = model.predict(reshaped)

            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(image, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(image, (x, y - 10), (x + w, y), color_dict[label], -1)
            cv2.putText(image, labels_dict[label], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        plt.imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():

    choice=int(input("Detect Face from Image or WebCam \n1. Image,\n2.WebCam\nYour Choice is:"))

    detect = DetectMask()
    if choice == 1:
        image_dir = 'images/image1.jpg'
        detect.DetectFaceMaskFromImage(image_dir)
    else:
        detect.DetectFaceMaskFromVideo()


if __name__ == '__main__':
    main()