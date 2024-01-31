import matplotlib.pyplot as plt
import numpy as np
import csv

def calculate_Euclidean_distance(point1, point2):
    sum = 0
    for i in range(len(point1)):
        sum += ((point2[i] - point1[i]) ** 2)
        
    return np.round((sum ** (1/2)), 2)


def calculate_CityBlock_distance(point1, point2):
    sum = 0
    for i in range(len(point1)):
        sum += abs(point2[i] - point1[i])
    return sum


def calculate_Sorensen_distance(point1, point2):
    sum1 = 0
    sum2 = 0
    for i in range(len(point1)):
        sum1 += abs(point2[i] - point1[i])
        sum2 += (point2[i] + point1[i])
    
    return sum1 / sum2


def read_dataset(dataset_name):
    with open(dataset_name, 'r') as file:
        csvFile = csv.reader(file)

        # Separate header
        header = next(csvFile)

        features = []
        labels = []
        for row in csvFile:
            features.append(list(map(float, row[:-1])))
            labels.append(row[-1])
        
        # Organize data into clusters for plot
        unique_labels = list(set(labels))
        clusters = []
        for i in unique_labels:
            cluster = []
            for j in range(len(labels)):
                if i == labels[j]:
                    cluster.append(features[j])
            clusters.append(cluster)

    return features, labels, clusters, unique_labels


def plot2D(clusters, test_sample, nearest_cluster, unique_labels):        
    for i,cluster in enumerate(clusters):
        x, y = zip(*cluster)
        b = unique_labels[i]
        plt.scatter(x, y, label=f'Cluster {b}')

    plt.scatter(test_sample[0], test_sample[1], label=f'Test sample in {nearest_cluster}')

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot for Each cluster')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


def KNN(features, labels, k, test_sample, clusters, unique_labels, distance_standard):
    distances = []
    for i in range(len(features)):
        if distance_standard == 'Euclidean':
            distances.append((calculate_Euclidean_distance(features[i], test_sample), labels[i]))
        elif distance_standard == 'CityBlock':
            distances.append((calculate_CityBlock_distance(features[i], test_sample), labels[i]))
        elif distance_standard == 'Sorensen':
            distances.append((calculate_Sorensen_distance(features[i], test_sample), labels[i]))
    
    # Sort distances descending
    distances.sort()

    # Initial the number of nearest neighbor
    nearest_neighbor = {}
    for i in unique_labels:
        nearest_neighbor[i] = 0

    # Check which class is more close to test sample
    for d in distances[:k]:
        nearest_neighbor[d[1]] += 1
    
    nearest_cluster = max(nearest_neighbor, key=lambda k:nearest_neighbor[k])

    print(f"\nThe {k} nearest neighbor with {distance_standard} distance is: {distances[:k]}")
    print(f"\ntest sample label: {nearest_cluster}")

    if len(features[0]) == 2:
        plot2D(clusters, test_sample, nearest_cluster, unique_labels)


if __name__ == "__main__":
    features, labels, clusters, unique_labels = read_dataset('dataset.csv')
    
    k = int(input("Enter the number of K: "))

    test_sample = eval(input("\nEnter the sample (example: [10, 20]): "))

    # Check the dimension of test sample and data is the same
    while len(test_sample) != len(features[0]):
        test_sample = input("\nEnter the sample (example: [10, 20]): ")
        
    KNN(features, labels, k, test_sample, clusters, unique_labels, 'Euclidean') 

    print("--------------------------------------------------------")

    KNN(features, labels, k, test_sample, clusters, unique_labels, 'CityBlock')

    print("--------------------------------------------------------")
    
    KNN(features, labels, k, test_sample, clusters, unique_labels, 'Sorensen')

