from re import sub
import subprocess
import glob
import streamlit as st

MURL = ""
st.title("3D object detection with point cloud")
st.subheader("Please upload your point cloud data as `.bin` file and we'll process")

fname = st.selectbox("", glob.glob("dataset/training/velodyne_reduced/*"))

if fname is None:
    st.text("Please upload valid file :(")
else:
    process = subprocess.Popen(
        [
            "python",
            ".//LiDAR-3D-Detector/tools/demo.py",
            "--ckpt",
            "./LiDAR-3D-Detector/output/kitti_models/3DSSD_openPCDet/3DSSD/ckpt/checkpoint_epoch_80.pth",
            '--cfg_file',
            './LiDAR-3D-Detector/output/kitti_models/3DSSD_openPCDet/3DSSD/3DSSD_openPCDet.yaml'
            '--data-path',
            fname
        ]
    )
    st.text(process.stdout)
