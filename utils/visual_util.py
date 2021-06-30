import numpy as np
import seaborn as sns
import mayavi.mlab as mlab

colors = sns.color_palette("Paired", 9 * 2)
classes = [
    "_",
    "Car",
    "Pedestrian",
    "Cyclist",
]


def visualize(arr, labels):
    def draw(p1, p2, front=1, is_str=0):
        mlab.plot3d(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=colors[classes.index(lab) * 2 + front]
            if is_str == 0
            else colors[int(lab.item()) * 2 + front],
            tube_radius=None,
            line_width=2,
            figure=fig,
        )

    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
    plot = mlab.points3d(arr[:, 0], arr[:, 1], arr[:, 2], mode="point", figure=fig)

    if isinstance(labels, str):
        with open(labels, "r") as f:
            labels = f.readlines()
        f.close()

        for line in labels:
            line = line.split()
            lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
            h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
            if lab in classes:
                x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
                z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
                corners_3d = np.vstack([x_corners, y_corners, z_corners])
                R = np.array(
                    [
                        [np.cos(rot), 0, np.sin(rot)],
                        [0, 1, 0],
                        [-np.sin(rot), 0, np.cos(rot)],
                    ]
                )
                corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])
                # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
                corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])
                print(corners_3d)
                draw(corners_3d[0], corners_3d[1], 0)  # front = 0 for the front lines
                draw(corners_3d[1], corners_3d[2])
                draw(corners_3d[2], corners_3d[3])
                draw(corners_3d[3], corners_3d[0])
                draw(corners_3d[4], corners_3d[5], 0)
                draw(corners_3d[5], corners_3d[6])
                draw(corners_3d[6], corners_3d[7])
                draw(corners_3d[7], corners_3d[4])
                draw(corners_3d[4], corners_3d[0], 0)
                draw(corners_3d[5], corners_3d[1], 0)
                draw(corners_3d[6], corners_3d[2])
                draw(corners_3d[7], corners_3d[3])
    else:
        for box in labels:
            lab, h, w, l, x, y, z, rot = box
            h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
            x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
            z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            corners_3d = np.vstack([x_corners, y_corners, z_corners])
            R = np.array(
                [
                    [np.cos(rot), 0, np.sin(rot)],
                    [0, 1, 0],
                    [-np.sin(rot), 0, np.cos(rot)],
                ]
            )
            corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])
            # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
            corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])
            print(corners_3d)
            draw(corners_3d[0], corners_3d[1], 0, 1)  # front = 0 for the front lines
            draw(corners_3d[1], corners_3d[2], 1, 1)
            draw(corners_3d[2], corners_3d[3], 1, 1)
            draw(corners_3d[3], corners_3d[0], 1, 1)
            draw(corners_3d[4], corners_3d[5], 0, 1)
            draw(corners_3d[5], corners_3d[6], 1, 1)
            draw(corners_3d[6], corners_3d[7], 1, 1)
            draw(corners_3d[7], corners_3d[4], 1, 1)
            draw(corners_3d[4], corners_3d[0], 0, 1)
            draw(corners_3d[5], corners_3d[1], 0, 1)
            draw(corners_3d[6], corners_3d[2], 1, 1)
            draw(corners_3d[7], corners_3d[3], 1, 1)
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=1)
    mlab.view(azimuth=230, distance=50)
    mlab.show()


if __name__ == "__main__":
    visualize(
        np.fromfile(
            "/home/patrick/Workspaces/LiDAR-Obj/dataset/training/velodyne_reduced/007112.bin",
            np.float32,
        ).reshape(-1, 4),
        "/home/patrick/Workspaces/LiDAR-Obj/dataset/training/label_2/007112.txt",
    )
