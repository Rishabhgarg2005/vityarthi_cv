# Classical Object Detection Toolkit (Single-File Version)

A Computer Vision project that detects objects using **classical CV techniques** (no deep learning), built to cover topics from the CV syllabus: image processing, segmentation, feature extraction (SIFT), homography/RANSAC, HOG + SVM, and cascade classifiers.

Everything lives in one file: `object_detection_single.py`.

---

## Features

| Mode | What it detects | Technique |
|---|---|---|
| `shapes` | Circles, triangles, squares, rectangles, pentagons | Otsu threshold, morphology, contours, polygon approximation |
| `template` | A known object inside a cluttered scene (rotation/scale tolerant) | SIFT, ratio-test matching, RANSAC homography |
| `people` | Pedestrians / full-body people | HOG features + pre-trained SVM, sliding window, NMS |
| `faces` | Frontal human faces | Haar cascade (Viola-Jones) |
| `demo` | Runs `shapes` + `template` on built-in synthetic images | No input files needed |

---

## Project layout

```
cv_project1/
├── object_detection_single.py   <- the whole project
├── README.md                    <- this file
└── outputs/                     <- created automatically; results saved here
```

Inside the script, code is divided into five sections:

1. **Utilities**: `nms()`, `draw_box()`, `load_image()`, `save()`
2. **Demo data**: functions that generate synthetic test images
3. **Detection methods**: `detect_shapes()`, `find_object()`, `detect_people()`, `detect_faces()`
4. **Runners**: `run_shapes()`, `run_template()`, `run_people()`, `run_faces()`
5. **Main**: argument parsing and dispatch

---

## Requirements

- Python 3.9 to 3.12
- Libraries:

```bash
pip install numpy opencv-contrib-python
```

> SIFT needs OpenCV 4.4 or newer. If you also have `opencv-python` installed, uninstall it first to avoid conflicts: `pip uninstall opencv-python`.

---

## Usage

Run from the folder containing the script:

```bash
# Built-in demo (no images needed). Also the default if no mode is given.
python object_detection_single.py demo

# Detect and label shapes in your own image
python object_detection_single.py shapes --image objects.jpg

# Find a known object (template) inside a scene
python object_detection_single.py template --template logo.jpg --image scene.jpg

# Detect people
python object_detection_single.py people --image street.jpg

# Detect faces
python object_detection_single.py faces --image group.jpg
```

Results are saved to `outputs/`:

| File | Produced by |
|---|---|
| `shapes_result.png`, `shapes_mask.png` | `shapes` / `demo` |
| `template_result.png` | `template` / `demo` |
| `people_result.png` | `people` |
| `faces_result.png` | `faces` |

### Expected demo output

```
[shapes]
  pentagon   box=(273, 231, 114, 108) area=8583
  triangle   box=(45, 231, 111, 94) area=5549
  rectangle  box=(220, 60, 181, 81) area=14400
  square     box=(450, 50, 91, 91) area=8098
  circle     box=(46, 46, 109, 109) area=9318
[template]
  FOUND: 27 good matches, 26 RANSAC inliers
```

---

## How it works

### 1. Shape detection (`detect_shapes`)
1. Convert to grayscale and apply a Gaussian blur to reduce noise.
2. Sample the image border to decide whether the background is light or dark, then apply an **Otsu threshold** (automatic cut-off) to produce a binary mask.
3. Apply **morphological opening** (removes specks) and **closing** (fills holes).
4. `findContours` extracts each object's outer outline; blobs smaller than 500 px² are ignored.
5. `approxPolyDP` simplifies each outline to a polygon. The vertex count gives the label: 3 = triangle, 4 = square/rectangle (by aspect ratio), 5 = pentagon.
6. Otherwise **circularity** `4πA/P²` is checked. Above 0.8 means circle.

### 2. Template detection (`find_object`)
1. **SIFT** finds scale- and rotation-invariant keypoints and 128-dimensional descriptors in both images.
2. `knnMatch` finds the two nearest scene descriptors for each template descriptor. **Lowe's ratio test** (0.75) keeps only clearly unambiguous matches.
3. **RANSAC** (`findHomography`) repeatedly fits a 3x3 homography from 4 random matches and keeps the one most matches agree with, which rejects wrong matches.
4. The template's four corners are projected through the homography to outline the object in the scene.
5. If fewer than 10 good matches remain, the object is reported as not found.

### 3. People detection (`detect_people`)
- **HOG** describes each 64x128 window by histograms of gradient orientations. OpenCV's built-in linear **SVM** (pre-trained on people) scores each window.
- `detectMultiScale` scans an **image pyramid** (scale 1.05) to catch people at different sizes.
- `nms()` (**Non-Maximum Suppression**) keeps the highest-scoring box and removes boxes overlapping it by more than 40% IoU.

### 4. Face detection (`detect_faces`)
- **Histogram equalization** evens out lighting.
- A **Haar cascade** (chain of boosted classifiers on integral-image features) quickly rejects non-face windows.

---

## Syllabus mapping

| Syllabus topic | Where it appears |
|---|---|
| Image enhancement, histogram processing | `equalizeHist` in `detect_faces` |
| Filtering / convolution | Gaussian blur in `detect_shapes` |
| Image segmentation | Otsu threshold, morphology, contours |
| Feature extraction: SIFT | `find_object` |
| Homography, RANSAC | `find_object` |
| HOG, image pyramids | `detect_people` |
| Supervised classifiers (SVM) | `detect_people` |
| Object detection | All modes |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `error: the following arguments are required: mode` | Old version of the script. The current one defaults to `demo`. Or type the mode explicitly. |
| `Could not read image` | Check the path. Use quotes if it contains spaces. |
| `people mode needs --image` | `people` and `faces` need your own photo; only `shapes` and `template` have built-in demos. |
| `cv2.SIFT_create` missing | Upgrade: `pip install -U opencv-contrib-python` |
| `object NOT found` in template mode | The object needs texture and should be reasonably large in the scene. Plain single-color objects give too few SIFT keypoints. |
| No faces or people found | Use larger, clearer images. HOG works best for upright, fully visible people, and Haar for frontal faces. |
| Odd file errors inside OneDrive | Move the project to a normal folder such as `C:\cv_project1`. |

---

## Limitations

- Classical methods are tied to specific object types. They cannot detect arbitrary categories such as cars or animals.
- For general multi-class detection, a deep model such as YOLO or SSD would be the next step (outside this syllabus).
