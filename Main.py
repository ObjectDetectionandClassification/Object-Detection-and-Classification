import numpy as np
import os
import cv2 as cv
from numpy import matlib
from pycocotools.coco import COCO
from keras.utils import to_categorical
from CO import CO
from DOA import DOA
from Global_Vars import Global_Vars
from Model_CNN import Model_CNN
from Model_Resnet import Model_RESNET
from Model_VGG16 import Model_VGG16
from Model_VT_AYV5_RAN import Model_VT_AYV5_RAN
from Obj_fun import objfun_cls
from PLot_Results import *
from PROPOSED import PROPOSED
from SSABA import SSABA
from TSO import TSO

no_of_dataset = 2


def load_coco_data(coco, dataDir, dataType, num_images=10):
    global category_id
    imgIds = coco.getImgIds()
    imgIds = imgIds[:num_images]  # Limiting to a subset for example purposes
    images = []
    annotated_images = []
    targets = []
    Classes = []
    i = 0
    for imgId in imgIds:
        print(i, len(imgIds))
        img = coco.loadImgs(imgId)[0]

        if dataDir == './Datasets/Dataset_2/coco2017':
            img_path = dataDir + '/' + dataType + '/' + img['file_name']
            image = cv.imread(img_path)
        elif dataDir == './Datasets/Dataset_1':
            img_path = dataDir + '/' + dataType + '/' + dataType + '/' + img['file_name']
            image = cv.imread(img_path)

        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        annotated_image = image.copy()

        # Draw bounding boxes on the original image
        for annotation in anns:
            bbox = annotation['bbox']
            category_id = annotation['category_id']
            category_name = coco.loadCats(category_id)[0]['name']

            bbox = np.asarray(bbox).astype('int')
            xmin, ymin, width, height = bbox
            xmax = int(xmin + width)
            ymax = int(ymin + height)
            cv.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        image = cv.resize(image, (256, 256))
        annotated_image = cv.resize(annotated_image, (256, 256))
        images.append(np.uint8(image))
        annotated_images.append(np.uint8(annotated_image))
        targets.append(category_id)
        Classes.append(category_name)
        i = i + 1
    Target = to_categorical(targets, dtype="uint8")
    return images, annotated_images, Target


# Read the Dataset
an = 0
if an == 1:
    dataDir_1 = './Datasets/Dataset_1'
    dataType_1 = 'train2014'  # 'val2014'
    annFile_1 = f'{dataDir_1}/annotations_trainval2014/annotations/instances_{dataType_1}.json'
    coco_1 = COCO(annFile_1)
    Images_1, Annotated_images_1, Targets_1 = load_coco_data(coco_1, dataDir_1, dataType_1, num_images=5000)

    dataDir_2 = './Datasets/Dataset_2/coco2017'
    dataType_2 = 'val2017'
    annFile_2 = f'{dataDir_2}/annotations/instances_{dataType_2}.json'
    coco_2 = COCO(annFile_2)
    Images_2, Annotated_images_2, Targets_2 = load_coco_data(coco_2, dataDir_2, dataType_2, num_images=5000)

    np.save('Images_1.npy', Images_1)
    np.save('Annotated_1.npy', Annotated_images_1)
    np.save('Targets_1.npy', Targets_1)
    np.save('Images_2.npy', Images_2)
    np.save('Annotated_2.npy', Annotated_images_2)
    np.save('Targets_2.npy', Targets_2)


# Optimization for Object Detection and Classification
an = 0
if an == 1:
    BEST_Solution = []
    FITNESS = []
    for n in range(no_of_dataset):
        Feat = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Images
        Targets = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Targets
        Global_Vars.Feat = Feat
        Npop = 10
        Chlen = 4  # Hidden Neuron Count, Epoch in Yolov5 , Step per epoch in AR2Unet++
        xmin = matlib.repmat(np.asarray([5, 5, 5, 5]), Npop, 1)
        xmax = matlib.repmat(np.asarray([50, 255, 50, 255]), Npop, 1)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("TSO...")
        [bestfit1, fitness1, bestsol1, time1] = TSO(initsol, fname, xmin, xmax, Max_iter)  # TSO

        print("CO...")
        [bestfit2, fitness2, bestsol2, time2] = CO(initsol, fname, xmin, xmax, Max_iter)  # OOA

        print("DOA...")
        [bestfit3, fitness3, bestsol3, time3] = DOA(initsol, fname, xmin, xmax, Max_iter)  # DOA

        print("LOA...")
        [bestfit4, fitness4, bestsol4, time4] = SSABA(initsol, fname, xmin, xmax, Max_iter)  # LOA

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Improved LOA

        BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

        BEST_Solution.append(BestSol)
        FITNESS.append(fitness)

    np.save('Fitness.npy', np.asarray(FITNESS))
    np.save('BestSol.npy', np.asarray(BEST_Solution))  # Bestsol classification


# Classification by Varying Batch Size
an = 0
if an == 1:
    Evall_All = []
    for n in range(no_of_dataset):
        EVAL = []
        BestSol = np.load('BestSol.npy', allow_pickle=True)[n]
        Feat = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)
        Batchsize = [4, 8, 16, 32, 48]
        for Bs in range(len(Batchsize)):
            learnperc = round(Feat.shape[0] * 0.75)
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval_Ima, Eval[j, :], pred_0 = Model_VT_AYV5_RAN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs], sol=sol)
            Eval[5, :], pred1 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs])
            Eval[6, :], pred2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs])
            Eval[7, :], pred3 = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs])
            Eval_Ima, Eval[8, :], pred4 = Model_VT_AYV5_RAN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs])
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Evall_All.append(EVAL)
    np.save('Eval_All_BS.npy', np.asarray(Evall_All))  # Save the Eval all


plot_conv()
ROC_curve()
Plot_Batch_size()
Plot_Epoch()
plot_object_detection_results()
Plot_YOLoV5_Images()
