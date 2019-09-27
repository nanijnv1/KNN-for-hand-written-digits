import math
import operator
#import matplotlib

#matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def euc_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def knn(training_data, test_data_instance, k):
    distances = []
    for x in range(len(training_data)):
        dist = euc_distance(test_data_instance[0], training_data[x][0], 64)
        distances.append((training_data[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    class_labels = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in class_labels:
            class_labels[response] += 1
        else:
            class_labels[response] = 1
    sorted_votes = sorted(class_labels.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


digits = datasets.load_digits()

k_values = [1, 3, 5, 100, 500]

digits_reshape = digits.images
digits_reshape = list(zip(digits_reshape.reshape((digits.images.shape[0], -1)),digits.target))


train_data, test_data = train_test_split(digits_reshape, test_size=0.5, random_state=42)

for k in k_values:
    true_values = []
    predictions = []
    for x in range(len(test_data)):
        nearest_neighbours = knn(train_data, test_data[x], k)
        result = get_response(nearest_neighbours)
        predictions.append(result)
        true_values.append(test_data[x][1])
    accuracy = get_accuracy(test_data, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    print('\n')
    print(classification_report(true_values, predictions))
    print('\n')
    confusion_mat=confusion_matrix(true_values, predictions)
    print(confusion_mat)
    print('\n')


