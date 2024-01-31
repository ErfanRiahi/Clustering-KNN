import matplotlib.pyplot as plt
import numpy as np
import random
import csv

def create_data(n_samples, n_features, n_cluster):
    # Create random data
    allData = []
    for s in range(n_samples):
        row = []
        for f in range(n_features):
            row.append(np.random.randint(0, 1000)) # Create data number between -100 to 100
        allData.append(row)

    # Initial centers randomly from previous data
    centers = []
    for _ in range(n_cluster):
        centers.append(random.choice(allData))
    
    return allData, centers


def Euclidean_distance(point1, point2):
    sum = 0
    for i in range(len(point1)):
        sum += ((point2[i] - point1[i]) ** 2)
        
    return np.round((sum ** (1/2)), 2)


def CityBlock_distance(point1, point2):
    sum = 0
    for i in range(len(point1)):
        sum += abs(point2[i] - point1[i])
    return sum


def Sorensen_distance(point1, point2):
    sum1 = 0
    sum2 = 0
    for i in range(len(point1)):
        sum1 += abs(point2[i] - point1[i])
        sum2 += (point2[i] + point1[i])
    
    return sum1 / sum2


def update_center(points):
    new_center = []
    for i in range(len(points[0])):
        sum = 0
        for j in range(len(points)):
            sum += points[j][i]
        new_center.append(np.round(sum/len(points), 2))
    return new_center


def clustering(allData, centers, n_iteration, distance_standard):
    for _ in range(n_iteration):
        # Initial clusters to empty
        clusters = [[] for _ in range(len(centers))]

        for i in range(len(allData)):
            distance = []
            for j in range(len(centers)):
                if distance_standard == 'Euclidean':
                    distance.append(Euclidean_distance(allData[i], centers[j]))
                elif distance_standard == 'CityBlock':
                    distance.append(CityBlock_distance(allData[i], centers[j]))
                elif distance_standard == 'Sorensen':
                    distance.append(Sorensen_distance(allData[i], centers[j]))

            center_index = distance.index(min(distance))
            clusters[center_index].append(allData[i])

        for i in range(len(centers)):
            centers[i] = update_center(clusters[i])
    
    return clusters


def plot2D(clusters, centers):
    for i,cluster in enumerate(clusters):
        x ,y = zip(*cluster)
        plt.scatter(x, y, label=f'Cluster {i + 1}')

    for i,center in enumerate(centers):
        x ,y = center[0], center[1]
        plt.scatter(x, y)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot for Each cluster')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


def plot3D(clusters):
    ax = plt.axes(projection='3d')
    for i,cluster in enumerate(clusters):
        x ,y, z = zip(*cluster)
        ax.scatter(x, y, z, label=f'Cluster {i + 1}')

    # Set the view of plot
    # ax.view_init(elev=10, azim=10) # Y-Z
    # ax.view_init(elev=10, azim=80) # X-Z
    # ax.view_init(elev=80, azim=0) # X-Y

    # Add labels and title
    ax.set_title('Scatter Plot for Each cluster')

    # Set axes label
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display legend
    ax.legend()

    # Show the plot
    plt.show()


def save_data(clusters, n_features):
    # Specify the CSV file name
    csv_file_name = 'dataset.csv'

    # Write data to the CSV file
    with open(csv_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Generate header
        header = [f'F{i}' for i in range(1, n_features + 1)] + ['Label']
        csv_writer.writerow(header)
        
        # Write rows
        for class_index, class_data in enumerate(clusters, start=1):
            for coordinates in class_data:
                features = coordinates[:n_features]
                label = f'C{class_index}'
                csv_writer.writerow(features + [label])


if __name__ == "__main__":
    n_samples = 100
    n_features = 2
    n_cluster = 3
    n_iteration = 10

    # Create random data and initial random centers between that data
    allData, centers = create_data(n_samples, n_features, n_cluster)

    # Create clusters with Euclidean distance
    clusters = clustering(allData, centers, n_iteration, 'Sorensen')

    save_data(clusters, n_features)

    # Plot the clusters if number of features is 2 or 3
    if n_features == 2:
        plot2D(clusters, centers)
    elif n_features == 3:
        plot3D(clusters)

    