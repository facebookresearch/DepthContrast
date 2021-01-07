### Prepare Dataset

We use scannet and redwood for Pointnet++ and MinkowskiEngine UNet pretraining.
We use waymo for PointnetMSG and Spconv-UNet model pretraining.

Please see the specific instructions under each folder for how to generate the training data.

Once you have generated the training data, the framework just takes a .npy file which consists of a list of paths to the extracted pointclouds:

[/path/to/pt1, /path/to/pt2, path/to/pt3, ..... ]

