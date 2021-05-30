# -*- coding: utf-8 -*-
from detector import CNN
import torch
import cv2
import numpy as np

face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels_dict={0: 'NO MASK', 1: 'MASK'}
color_dict={0: (0, 0, 0), 1: (255, 255, 255)}

model = CNN(in_channels=3, num_classes=2)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

size = 120

source = cv2.VideoCapture(0)
with torch.no_grad():
    while True:
        ret, frame = source.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_clsfr.detectMultiScale(rgb,1.3,5)

        for (x, y, w, h) in faces:
            face_img_pred = rgb[y:y + w, x:x + w]
            face_img_reg = frame[y:y + w, x:x + w]

            img_array = cv2.cvtColor(face_img_pred, cv2.COLOR_BGR2RGB)

            new_array = cv2.resize(img_array, (size, size))
            new_array = np.transpose(new_array, (2, 0, 1)).astype(np.float32)
            new_array = torch.tensor(new_array).unsqueeze(0)

            prediction = model(new_array)
            predicted_class = np.argmax(prediction)
            label = int(predicted_class)
            print(predicted_class)
            cv2.rectangle(face_img_reg, (0, 0), (0 + w, 0 + h), color_dict[label], 3)
            cv2.putText(face_img_reg, labels_dict[label], (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_dict[label], 2)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    source.release()
    cv2.destroyAllWindows()

