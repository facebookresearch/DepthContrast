### Prepare Dataset

1. Download Waymo data [HERE](https://waymo.com/open/). We suggest to use [this](https://github.com/RalphMao/Waymo-Dataset-Tool) tool for batch downloading.

2. Please install this [tool](https://github.com/waymo-research/waymo-open-dataset) for data preprocessing.

3. Create a data_folder for extracting point clouds. Use extract_pointcloud.py to extract point clouds.
python extract_pointcloud.py /path/to/downloaded_segments /path/to/data_folder waymo.npy
