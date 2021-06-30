import glob
import PIL.Image as IMG
import streamlit as st
import numpy as np


MURL = ""
st.title("3D object detection with point cloud")
st.info(f"I use KITTI dataset for training this model based on OpenPCDet powered by MMLab. The model is constructed based on PointPillars achitecture by Facebook, but I've change something inside the backbone. Thanks for enjoying my experimental :-3")

fname = st.selectbox("", glob.glob("dataset/snapshot/*"))
st.text("Please select our point cloud data it'll show you our results")
st.image(IMG.open(fname))
