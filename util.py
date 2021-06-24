from iou import box3d_iou
import numpy as np

def get_onehot(num, total):
    hot = [0] * total
    hot[num] = 1
    return hot


def get_label(self, fname):
    constant = {
        "KITTI": [
            "_",  # Background class
            "Car",
            "Van",
            "Truck",
            "Pedestrian",
            "Person_sitting",
            "Cyclist",
            "Tram",
            "Misc",
        ],
        "KITTI_CAR": ["_", "Car"],
        "KITTI_CAR_PRED" : ["_","Car","Pedestrian"]
    }
    label_data = []
    with open(fname, "r") as f:
        for line in f.readlines():
            if len(line) > 3:
                value = line.split()
            if value[0] in constant[self.dataset]:
                value[0] = constant[self.dataset].index(value[0])
            else:
                value[0] = 0
            data = []
            if self.dataset in ["KITTI", "KITTI_CAR"]:
                data.extend(
                    get_onehot(int(value[0]), len(constant[self.dataset]))
                )
                data.extend([float(v) for v in value[8:]])
            label_data.append(data)
    f.close()
    return np.asarray(label_data)


def kmeans(boxes, k=7, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - box3d_iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters
