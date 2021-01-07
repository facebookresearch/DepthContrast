### Prepare Dataset

1. Download Redwood data [HERE](https://github.com/intel-isl/redwood-3dscan).

2. Extract the pointclouds of the desired classes using extract_pointcloud.py

python extract_pointcloud.py /path/to/extracted_data /path/to/extracted_pointcloud_visualization /path/to/extracted_pointclouds redwood_datalist.npy

The visualizations are optional.

3. In our experiment, we use the following 10 classes:

car, chair, table, bench, bicycle, plant, playground, sculpture, sign, trash_container

4. You will need to concatenate the datalist files if you use data from multiple categories.
