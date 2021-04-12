### Prepare Dataset

1. Download ScanNet data [HERE](https://github.com/ScanNet/ScanNet).

2. Please use python2.7 to run the following command to extract depth images, color images, and camera intrinsics:

```
python2.7 reader.py --scans_path path/to/scannet_v2/scans/ --output_path path/to/extracted_data/ --export_depth_images --export_color_images --export_poses --export_intrinsics
```

2. Extract point clouds using extract_pointcloud.py

```
python extract_pointcloud.py path/to/extracted_data/depth/ path/to/extracted_pointcloud_visualization/ path/to/extracted_pointclouds/ scannet_datalist.npy
```

The visualizations are optional.