from random import shuffle
import matplotlib.pyplot as plt
import os
import numpy as np
import csv


def split_data(labels, split):
    """
    split dataset into train/validation/test set

    :param labels: dictionary containing {file:class}
    :param split: tuple of 3 elements (train_fraction, validation_fraction, test_fraction) adding up to 1
    :return: tuple of 3 dictionaries train_set, validation_set, test_set
    """
    assert(len(split) == 3)
    assert(sum(split) == 1.0)

    keys = list(labels.keys())
    shuffle(keys)
    labels_train = {}
    labels_validation = {}
    labels_test = {}

    for k in range(len(keys)):
        key = keys[k]
        if k < split[0] * len(keys):
            labels_train[key] = labels[key]
        elif k < (split[0] + split[1]) * len(keys):
            labels_validation[key] = labels[key]
        else:
            labels_test[key] = labels[key]

    return labels_train, labels_validation, labels_test


def save_plot(history, metric, path):
    """
    save model train history as png image for the corresponding metric

    :param history: model train history
    :param metric: metric to plot
    :param path: where to save image
    """
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('model ' + metric)
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.legend(['train', 'validation'])
    plt.savefig(os.path.join(path, metric))
    plt.close()


def write_model_performance(y_true, y_pred, file_name, class_names=None):
    """
    save model performance [True Positive, False Positive, False Negative, Precision, Recall(TPR),
    F1 score, Accuracy] in csv file

    :param y_true: true labels(not in categorical/one hot encoding format)
    :param y_pred: predicted labels
    :param file_name: csv file name
    :param class_names: optional list of names for classes [0, 1, 2 ...] in order
    """
    assert(len(y_true) == len(y_pred))
    if class_names is None:
        # assume number of class is equal to maximum number of unique entries in y_true or y_pred
        class_names = [i for i in range(max(len(set(y_pred)), len(set(y_true))))]
    num_class = len(class_names)

    # calculate true_positive, false_positive, false_negative
    confusion_matrix = np.zeros((num_class, 3))
    for i in range(len(y_pred)):
        true_label = y_true[i]
        predicted_label = y_pred[i]
        if predicted_label == true_label:
            confusion_matrix[true_label][0] += 1
        else:
            confusion_matrix[true_label][2] += 1
            confusion_matrix[predicted_label][1] += 1

    # open csv file
    csv_file = open(file_name, "w")
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Class', 'Samples', 'True Positive', 'False Positive', 'False Negative', 'Precision',
                     'Recall(TPR)', 'F1 score', 'Accuracy'])

    # calculate precision, recall, f1-score, accuracy
    total_samples = len(y_pred)
    tp_total = 0
    fp_total = 0
    fn_total = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    for i in range(num_class):
        tp = confusion_matrix[i][0]
        fp = confusion_matrix[i][1]
        fn = confusion_matrix[i][2]
        tn = total_samples - (tp + fp + fn)
        samples = tp + fn

        tp_total += tp
        fp_total += fp
        fn_total += fn

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        accuracy = (tp + tn) / total_samples

        writer.writerow([class_names[i], samples, tp, fp, fn, round(precision, 4), round(recall, 4),
                         round(f1, 4), round(accuracy, 4)])

        avg_precision += precision * (confusion_matrix[i][0] + confusion_matrix[i][2])
        avg_recall += recall * (confusion_matrix[i][0] + confusion_matrix[i][2])
        avg_f1 += f1 * (confusion_matrix[i][0] + confusion_matrix[i][2])

    avg_precision /= total_samples
    avg_recall /= total_samples
    avg_f1 /= total_samples
    accuracy = tp_total / total_samples
    writer.writerow(['Total', total_samples, tp_total, fp_total, fn_total, round(avg_precision, 4),
                     round(avg_recall, 4), round(avg_f1, 4), round(accuracy, 4)])

    csv_file.close()
