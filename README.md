# PCR-ORB

PCR-ORB is a fork of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) that adds a CUDA-accelerated **P**oint **C**loud filte**R** to remove dynamic, ground, sky, and edge outliers from the tracked map before they corrupt pose estimation. It combines a YOLOv8 segmentation model (exported to TorchScript) with temporal motion masking and RANSAC-based ground plane estimation to keep only the static, reliable feature points used for tracking and mapping.

Supports Monocular, Stereo, RGB-D, and their Visual-Inertial counterparts, on standard datasets (EuRoC, TUM, KITTI, TUM-VI) and live Intel RealSense cameras, with optional ROS integration.

## What's different from upstream ORB-SLAM3

- **`PointCloudFilter`** ([include/PointCloudFilter.h](include/PointCloudFilter.h), [src/PointCloudFilter.cc](src/PointCloudFilter.cc)): runs a YOLOv8-seg TorchScript model per frame (GPU via LibTorch/CUDA) to build a dynamic-object segmentation mask, fuses it with a motion mask and an estimated ground plane, and scores/filters each candidate keypoint before it is handed to the tracker.
- Tracked filtering statistics (true/false positives/negatives) and confusion-matrix export for evaluating filter quality ([visualize_filter_results.py](visualize_filter_results.py) plots them).
- `Yolo.py` exports the `yolov8s-seg.pt` Ultralytics model to the TorchScript file the C++ filter loads at runtime.

Everything else — Atlas/Map management, tracking, local mapping, loop closing, IMU initialization, camera models — is inherited from ORB-SLAM3 v1.0. See [Changelog.md](Changelog.md) for upstream version history and [Dependencies.md](Dependencies.md) for third-party code/license attributions.

## Dependencies

- C++14 compiler
- [OpenCV](https://opencv.org/) >= 4.10, **built with CUDA modules** (`cudawarping`, `cudafilters`, `cudaarithm`, `cudaimgproc`, `cudaoptflow`)
- CUDA toolkit (tested with `sm_60`/`sm_75`/`sm_86` GPU architectures)
- [LibTorch](https://pytorch.org/get-started/locally/) 2.1.0 (CUDA build), expected at `Thirdparty/libtorch`
- [Eigen3](https://eigen.tuxfamily.org/) >= 3.1.0
- [Pangolin](https://github.com/stevenlovegrove/Pangolin) (viewer/UI)
- OpenMP
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) (optional, for live RealSense examples)
- ROS (optional, only needed to build `Examples/ROS`)

See [Dependencies.md](Dependencies.md) for the full list of bundled/modified third-party code and its licenses.

## Building

```bash
./build.sh
```

This builds the bundled `Thirdparty/DBoW2`, `Thirdparty/g2o`, and `Thirdparty/Sophus` libraries, decompresses the ORB vocabulary, and then builds PCR-ORB itself into `build/` and `lib/libORB_SLAM3.so`.

Before building, make sure LibTorch is present at `Thirdparty/libtorch` and that OpenCV was built with CUDA support — the CMake configuration will warn (but not fail) if CUDA modules are missing from OpenCV.

To also build the ROS examples:

```bash
./build_ros.sh
```

(requires `Examples/ROS/ORB_SLAM3` to be on your `ROS_PACKAGE_PATH`).

## Segmentation model setup

The point cloud filter expects a TorchScript-exported YOLOv8 segmentation model named `yolov8s-seg.torchscript` on the runtime path. Generate it with:

```bash
pip install ultralytics
python Yolo.py
```

This downloads `yolov8s-seg.pt` and exports it to TorchScript at 640x640 input resolution.

## Running

Executables are built into `Examples/<Monocular|Stereo|RGB-D|...>/`. Usage mirrors upstream ORB-SLAM3, e.g.:

```bash
./Examples/Monocular/mono_euroc \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/EuRoC.yaml \
    PATH_TO_EUROC_SEQUENCE \
    Examples/Monocular/EuRoC_TimeStamps/SEQUENCE.txt
```

```bash
./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM1.yaml \
    PATH_TO_TUM_SEQUENCE \
    Examples/RGB-D/associations/ASSOCIATION_FILE.txt
```

Calibration file format (including stereo rectification and image resizing options) is documented in [Calibration_Tutorial.pdf](Calibration_Tutorial.pdf).

## Evaluation

Trajectory accuracy against ground truth (EuRoC/TUM) can be computed with the scripts in [evaluation/](evaluation):

```bash
python evaluation/evaluate_ate_scale.py GROUND_TRUTH.txt ESTIMATED_TRAJECTORY.txt --plot output.png
```

Point-cloud filter quality (confusion matrix of true/false positive/negative dynamic-point classifications) can be visualized with:

```bash
python visualize_filter_results.py
```

## License

PCR-ORB is built on ORB-SLAM3, which is released under GPLv3. Bundled and modified third-party code (DBoW2, g2o, Sophus, and portions of OpenCV/OpenGV) retains its original BSD/MIT licenses — see [Dependencies.md](Dependencies.md) for details. If you plan to redistribute this repository, add an explicit `LICENSE` file consistent with these terms.
