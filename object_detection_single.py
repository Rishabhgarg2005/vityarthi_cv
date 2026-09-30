"""
Classical Object Detection Toolkit - SINGLE FILE VERSION
=========================================================
Install :  pip install numpy opencv-contrib-python
Run     :  python object_detection_single.py demo
           python object_detection_single.py shapes   --image objects.jpg
           python object_detection_single.py template --template logo.jpg --image scene.jpg
           python object_detection_single.py people   --image street.jpg
           python object_detection_single.py faces    --image group.jpg
Results are saved in the 'outputs' folder.
"""
import argparse
import os
import sys

import cv2
import numpy as np

OUT = "outputs"


# =============================================================================
# 1. UTILITIES
# =============================================================================
def nms(boxes, scores, iou_thr=0.4):
    """Non-Maximum Suppression: keep best box, drop boxes overlapping it (IoU)."""
    if len(boxes) == 0:
        return []
    b = np.array(boxes, dtype=float)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    areas = b[:, 2] * b[:, 3]
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return keep


def draw_box(img, box, label, color=(0, 255, 0)):
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print(f"  saved -> {path}")


# =============================================================================
# 2. DEMO DATA (synthetic images, no downloads needed)
# =============================================================================
def demo_shapes_scene():
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.circle(img, (100, 100), 55, (0, 0, 220), -1)
    cv2.rectangle(img, (220, 60), (400, 140), (220, 60, 0), -1)
    cv2.rectangle(img, (450, 50), (540, 140), (0, 160, 0), -1)
    cv2.fillPoly(img, [np.array([[100, 330], [40, 230], [160, 230]])], (0, 140, 255))
    pent = np.array([[330 + 60 * np.cos(2 * np.pi * k / 5 - np.pi / 2),
                      290 + 60 * np.sin(2 * np.pi * k / 5 - np.pi / 2)]
                     for k in range(5)], np.int32)
    cv2.fillPoly(img, [pent], (150, 0, 150))
    return img


def demo_template():
    rng = np.random.default_rng(1)
    t = np.full((200, 200, 3), 230, np.uint8)
    for _ in range(40):
        p1 = tuple(int(v) for v in rng.integers(0, 200, 2))
        p2 = tuple(int(v) for v in rng.integers(0, 200, 2))
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(t, p1, p2, col, -1)
        else:
            cv2.circle(t, p1, int(rng.integers(5, 30)), col, -1)
    cv2.rectangle(t, (0, 0), (199, 199), (0, 0, 0), 3)
    return t


def demo_template_scene(template):
    rng = np.random.default_rng(7)
    scene = rng.integers(60, 200, (480, 640, 3), dtype=np.uint8)
    scene = cv2.GaussianBlur(scene, (9, 9), 0)
    for _ in range(25):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(scene, (int(rng.integers(640)), int(rng.integers(480))),
                   int(rng.integers(8, 25)), c, -1)
    M = cv2.getRotationMatrix2D((100, 100), 30, 1.1)   # rotate 30 deg, scale 1.1
    M[:, 2] += (330, 130)
    warped = cv2.warpAffine(template, M, (640, 480))
    mask = cv2.warpAffine(np.full(template.shape[:2], 255, np.uint8), M, (640, 480))
    scene[mask > 0] = warped[mask > 0]
    return scene


# =============================================================================
# 3. DETECTION METHODS
# =============================================================================
def detect_shapes(img, min_area=500):
    """Method 1: threshold -> morphology -> contours -> polygon approximation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    mode = cv2.THRESH_BINARY_INV if border.mean() > 127 else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        n = len(approx)
        circularity = 4 * np.pi * area / (peri * peri)
        if n == 3:
            label = "triangle"
        elif n == 4:
            _, _, w, h = cv2.boundingRect(approx)
            label = "square" if 0.9 <= w / h <= 1.1 else "rectangle"
        elif n == 5:
            label = "pentagon"
        elif circularity > 0.8:
            label = "circle"
        else:
            label = f"polygon({n})"
        results.append({"box": cv2.boundingRect(c), "label": label, "area": area})
    return results, mask


def find_object(template, scene, min_matches=10, ratio=0.75):
    """Method 2: SIFT -> ratio-test matching -> RANSAC homography."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None:
        return None

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < min_matches:
        return {"found": False, "matches": len(good)}

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return {"found": False, "matches": len(good)}

    h, w = template.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return {"found": True, "matches": len(good), "inliers": int(inliers.sum()),
            "polygon": np.int32(cv2.perspectiveTransform(corners, H))}


def detect_people(img):
    """Method 3: HOG features + pre-trained linear SVM, sliding window, NMS."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    if len(boxes) == 0:
        return []
    weights = np.array(weights).flatten()
    keep = nms(boxes.tolist(), weights)
    return [(tuple(boxes[i]), float(weights[i])) for i in keep]


def detect_faces(img):
    """Method 4: Haar cascade (Viola-Jones) on a histogram-equalized image."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


# =============================================================================
# 4. RUNNERS
# =============================================================================
def run_shapes(path):
    img = load_image(path) if path else demo_shapes_scene()
    results, mask = detect_shapes(img)
    out = img.copy()
    for r in results:
        draw_box(out, r["box"], r["label"])
        print(f"  {r['label']:<10} box={r['box']} area={r['area']:.0f}")
    save("shapes_result.png", out)
    save("shapes_mask.png", mask)


def run_template(template_path, scene_path):
    if template_path and scene_path:
        template, scene = load_image(template_path), load_image(scene_path)
    else:
        template = demo_template()
        scene = demo_template_scene(template)
    res = find_object(template, scene)
    out = scene.copy()
    if res and res["found"]:
        cv2.polylines(out, [res["polygon"]], True, (0, 255, 0), 3)
        print(f"  FOUND: {res['matches']} good matches, {res['inliers']} RANSAC inliers")
    else:
        print("  object NOT found")
    save("template_result.png", out)


def run_people(path):
    img = load_image(path)
    out = img.copy()
    dets = detect_people(img)
    for box, score in dets:
        draw_box(out, box, f"person {score:.2f}", (0, 0, 255))
    print(f"  {len(dets)} person(s) detected")
    save("people_result.png", out)


def run_faces(path):
    img = load_image(path)
    out = img.copy()
    faces = detect_faces(img)
    for f in faces:
        draw_box(out, f, "face", (255, 0, 0))
    print(f"  {len(faces)} face(s) detected")
    save("faces_result.png", out)


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Classical object detection toolkit")
    p.add_argument("mode", nargs="?", default="demo",
                   choices=["shapes", "template", "people", "faces", "demo"])
    p.add_argument("--image", help="input image (scene)")
    p.add_argument("--template", help="template image for 'template' mode")
    a = p.parse_args()

    if a.mode == "shapes":
        run_shapes(a.image)
    elif a.mode == "template":
        run_template(a.template, a.image)
    elif a.mode in ("people", "faces"):
        if not a.image:
            sys.exit(f"{a.mode} mode needs --image")
        (run_people if a.mode == "people" else run_faces)(a.image)
    else:
        print("[shapes]")
        run_shapes(None)
        print("[template]")
        run_template(None, None)


if __name__ == "__main__":
    main()
