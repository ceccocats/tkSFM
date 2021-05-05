# based on https://github.com/gaoxiang12/g2o_ba_example

# usage 
```
./ba_example ../data/1.png ../data/2.png
pcl_viewer pts.pcd  
```

# kitti data example
```
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
unzip 2011_09_26_drive_0001_sync.zip
./tkSFM 2011_09_26/2011_09_26_drive_0001_sync
```


# g2o_ba_example
An easy example of doing bundle adjustment within two images using g2o. 

Require: g2o, OpenCV 2.4.x

The program reads two images from the data/1.png and data/2.png, then finds and matches orb key-points between them. After these, it uses g2o to estimate the relative motion between frames and the 3d positions (under a unknown scale).

This is an example written for the beginners of g2o and SLAM.

For more details please see the corresponding blog: http://www.cnblogs.com/gaoxiang12/p/5304272.html (in Chinese only).
