from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_point(arr):
	figure = plt.figure()
	ax = figure.add_subplot(111,projection='3d')
	x, y, z = arr[:,0],arr[:,1],arr[:2]
	ax.scatter(x,y,z,marker='o')
	plt.show()

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
