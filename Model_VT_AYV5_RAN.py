import cv2
import torch
import numpy as np
from Evaluation import net_evaluation
from Model_RAN import Model_RAN
from yolov5_master.Fitting import fit
from sklearn.model_selection import train_test_split


# https://github.com/ultralytics/yolov5
def Model_YOLOV5(Images, BS=None, sol=None):
    if sol is None:
        sol = [5, 5, 5]
    if BS is None:
        BS = 4
    # Load 3DAMDY_AM model
    model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s', pretrained=True)
    IMG = []
    for i in range(len(Images)):
        print(i, len(Images))
        img = Images[i].astype('uint8')
        img = fit(img, epoch=int(sol[1]), batch_size=BS, steps_per_epoch=5)
        # Perform inference
        results = model(img)
        # Extract face bounding boxes
        faces = results.xyxy[0][results.xyxy[0][:, -1] == 0][:, :-1]
        # Draw bounding boxes on the image
        for face in faces:
            xy = (int(face[0]), int(face[1]))
            xy2 = (int(face[2]), int(face[3]))
            cv2.rectangle(img, xy, xy2, (0, 255, 0), 2)
        IMG.append(img)
    return np.asarray(IMG)


def Model_VT_AYV5_RAN(Train_Data, Train_Target, Test_Data, Test_Target, BS=None, sol=None):
    if sol is None:
        sol = [5, 5, 5, 5]
    if BS is None:
        BS = 4
    Images = np.concatenate((Train_Data, Test_Data))
    Target = np.concatenate((Train_Target, Test_Target), axis=0)
    Roi = Model_YOLOV5(Images, BS=BS, sol=sol[:2])
    EVAl = []
    for img in range(len(Roi)):
        act = cv2.cvtColor(Images[img], cv2.COLOR_BGR2GRAY)
        pred = cv2.cvtColor(Roi[img], cv2.COLOR_BGR2GRAY)
        Eval = net_evaluation(act, pred)
        EVAl.append(Eval)
    Eval_Images = np.mean(EVAl, axis=0, keepdims=True)[0]
    # X_train, X_test, y_train, y_test = train_test_split(Roi, Target, test_size=0.25, random_state=0)
    X_train, X_test, y_train, y_test = Roi[:Train_Data.shape[0]], Target[:Train_Target.shape[0]], Roi[:Test_Data.shape[0]], Target[:Test_Target.shape[0]]
    Eval, Pred = Model_RAN(X_train, y_train, X_test, y_test, BS=BS, sol=sol[2:])
    return Eval_Images, Eval, Pred

