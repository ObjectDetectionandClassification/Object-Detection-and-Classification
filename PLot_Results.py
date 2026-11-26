import numpy as np
import warnings
from matplotlib import pylab
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv

warnings.filterwarnings("ignore")
no_of_dataset = 2


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    matplotlib.use('Qt5Agg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'TSO-VT-AYV5-RAN', 'CO-VT-AYV5-RAN', 'DO-VT-AYV5-RAN', 'SSABA-VT-AYV5-RAN', 'RSSAA-VT-AYV5-RAN']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((Fitness.shape[1], 5))
    for n in range(Fitness.shape[0]):
        for j in range(len(Algorithm) - 1):
            Conv_Graph[j, :] = stats(Fitness[n, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report Dataset ' + str(n+1),
              ' --------------------------------------------------')
        print(Table)
    for n in range(Fitness.shape[0]):
        length = np.arange(Fitness.shape[-1])
        Conv_Graph = Fitness[n]

        plt.plot(length, Conv_Graph[0, :], color='#f97306', linewidth=3, label='TSO-VT-AYV5-RAN')
        plt.plot(length, Conv_Graph[1, :], color='#3d7afd', linewidth=3, label='CO-VT-AYV5-RAN')
        plt.plot(length, Conv_Graph[2, :], color='#b9ff66', linewidth=3, label='DO-VT-AYV5-RAN')
        plt.plot(length, Conv_Graph[3, :], color='#bb3f3f', linewidth=3, label='SSABA-VT-AYV5-RAN')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label='RSSAA-VT-AYV5-RAN')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)

        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence')
        plt.savefig("./Results/Dataset_%s_Convergence.png" % (n + 1))
        plt.show()


def ROC_curve():
    lw = 2
    cls = ['Viola Jones', 'YOLO DisNet', 'OWA CNN', 'VT-AYV5-RAN', 'RSSAA-VT-AYV5-RAN']
    for n in range(no_of_dataset):
        Actual = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True).astype('int')
        colors = cycle(["#fe2f4a", "#0165fc", "#be03fd", "lime", "black"])
        for i, color in zip(range(len(cls)), colors):
            Predicted = np.load('Y_Score_' + str(n + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        path = "./Results/Dataset_%s_ROC.png" % str(n + 1)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('ROC Curve of Dataset ' + str(n+1))
        plt.savefig(path)
        plt.show()


def Plot_Batch_size():
    eval = np.load('Eval_ALL_BS.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1 score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [0, 2, 9, 10]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j] + 4]

            length = np.arange(Graph.shape[0])
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='D', markerfacecolor='red',  # 98F5FF
                    markersize=6, label='TSO-VT-AYV5-RAN')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='s', markerfacecolor='green',  # 7FFF00
                    markersize=6, label='CO-VT-AYV5-RAN')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='H', markerfacecolor='cyan',  # C1FFC1
                    markersize=8, label='DO-VT-AYV5-RAN')
            ax.plot(length, Graph[:, 3], color='#00ffff', linewidth=3, marker='p', markerfacecolor='#fdff38',
                    markersize=8, label='SSABA-VT-AYV5-RAN')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='w', markersize=12,
                    label='RSSAA-VT-AYV5-RAN')
            plt.xticks(length, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch Size', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('K Fold vs ' + Terms[Graph_Term[j]])
            path = "./Results/Dataset_%s_Batch Size_%s_lrean.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(Graph.shape[0])
            ax.bar(X + 0.00, Graph[:, 5], color='#ff5b00', edgecolor='w', width=0.15, label="Viola Jones")
            ax.bar(X + 0.15, Graph[:, 6], color='#0ffef9', edgecolor='w', width=0.15, label="YOLO DisNet")
            ax.bar(X + 0.30, Graph[:, 7], color='#fe46a5', edgecolor='w', width=0.15, label="OWA CNN")
            ax.bar(X + 0.45, Graph[:, 8], color='#61e160', edgecolor='w', width=0.15, label="VT-AYV5-RAN")
            ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="RSSAA-VT-AYV5-RAN")
            plt.xticks(X + 0.15, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch Size', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('K Fold vs ' + Terms[Graph_Term[j]])
            path = "./Results/Dataset_%s_Batch Size_%s_bar.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


def Plot_Epoch():
    eval = np.load('Eval_all_Epoch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1 score',
             'MCC', 'pt', 'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Table_Term = [0, 1, 14, 16, 17]
    Algorithm = ['TERMS', 'TSO-VT-AYV5-RAN', 'CO-VT-AYV5-RAN', 'DO-VT-AYV5-RAN', 'SSABA-VT-AYV5-RAN', 'RSSAA-VT-AYV5-RAN']
    Classifier = ['TERMS', 'Viola Jones', 'YOLO DisNet', 'OWA CNN', 'VT-AYV5-RAN', 'RSSAA-VT-AYV5-RAN']
    Epoch = [50, 100, 150, 200, 250]
    for n in range(eval.shape[0]):
        for i in range(eval.shape[1]):
            value = eval[n, i, :, 4:]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Term)])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[j, Table_Term])
            print('--------------------------------------------------' + str(Epoch[i]) + ' Epoch',
                  'Algorithm Comparison of Dataset ' + str(n + 1),
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Term])
            print('-------------------------------------------------- ' + str(Epoch[i]) + ' Epoch',
                  'Classifier Comparison of Dataset ' + str(n + 1),
                  '--------------------------------------------------')
            print(Table)


def plot_object_detection_results():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'PSNR', 'MSE', 'IoU', 'Sensitivity', 'Specificity',
             'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            Graph = stats[i, :5, 2]
            labels = ['TSO-VT-AYV5-RAN', 'CO-VT-AYV5-RAN', 'DO-VT-AYV5-RAN', 'SSABA-VT-AYV5-RAN', 'RSSAA-VT-AYV5-RAN']
            colors = ['#f97306', '#bf77f6', '#6e750e', '#ff474c', 'k']
            ax.bar(labels, Graph, color=colors)
            plt.xticks(labels, rotation=10)
            plt.ylabel(Terms[i - 4])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Segmented Mean vs ' + Terms[i - 4])
            path = "./Results/Mean_object_Dataset_%s_%s_Alg.png" % (n + 1, Terms[i - 4])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            Graph = stats[i, 5:10, 2]
            labels = ['Viola Jones', 'YOLO DisNet', 'OWA CNN', 'VT-AYV5-RAN', 'RSSAA-VT-AYV5-RAN']
            colors = ['#0165fc', '#dbb40c', '#ff474c', '#7af9ab', 'k']
            ax.bar(labels, Graph, color=colors)
            plt.xticks(labels, rotation=10)
            plt.ylabel(Terms[i - 4])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Segmented Mean vs ' + Terms[i - 4])
            path = "./Results/Mean_object_Dataset_%s_%s_mtd.png" % (n + 1, Terms[i - 4])
            plt.savefig(path)
            plt.show()


def Plot_YOLoV5_Images():
    for n in range(no_of_dataset):
        Orignal_Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Object_Detected = np.load('Object_Detected_' + str(n + 1) + '.npy', allow_pickle=True)

        for i in range(5):
            Orig_image = Orignal_Images[i]
            Detect_image = Object_Detected[i]
            cv.imshow('Dataset' + str(n+1) + 'Orignal Images ' + str(i + 1), cv.resize(Orig_image, (512, 512)))
            cv.imshow('Dataset' + str(n+1) + 'YOLO_V5 Images ' + str(i + 1),  cv.resize(Detect_image, (512, 512)))
            cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + '_Original_image_' + str(i + 1) + '.png',
                       Orig_image)
            cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + '_YOLO_V5_image_' + str(i + 1) + '.png',
                       Detect_image)
            cv.waitKey(0)


if __name__ == '__main__':
    plot_conv()
    ROC_curve()
    Plot_Batch_size()
    Plot_Epoch()
    plot_object_detection_results()
    Plot_YOLoV5_Images()
