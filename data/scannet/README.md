### Prepare Dataset

1. Download ScanNet data [HERE](https://github.com/ScanNet/ScanNet). Please use this file to extract depth images, color images, and camera intrinsics using this [file](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

2. Extract point clouds using extract_pointcloud.py

python extract_pointcloud.py /path/to/extracted_data /path/to/extracted_pointcloud_visualization /path/to/extracted_pointclouds scannet_datalist.npy

The visualizations are optional.