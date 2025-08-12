# 02_calibrate_camera.py
import cv2
import numpy as np
import glob
import argparse
import os

def parse_args():
    ap = argparse.ArgumentParser("Calibrate camera from chessboard images")
    ap.add_argument("--images_glob", type=str, default="data/calib/*.png", help="Путь-шаблон к фото")
    ap.add_argument("--cols", type=int, default=6, help="внутренние углы по горизонтали")
    ap.add_argument("--rows", type=int, default=9, help="внутренние углы по вертикали")
    ap.add_argument("--square_size_mm", type=float, default=28, help="сторона КЛЕТКИ, мм")
    ap.add_argument("--save", type=str, default="calibration_cam0.npz", help="файл для сохранения параметров")
    return ap.parse_args()

def main():
    args = parse_args()
    pattern_size = (args.cols, args.rows)
    square_size_m = args.square_size_mm / 1000.0

    images = sorted(glob.glob(args.images_glob))
    if not images:
        raise SystemExit("Нет изображений по шаблону: " + args.images_glob)

    objp = np.zeros((args.rows*args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []
    imgpoints = []
    used = []

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    im_size = None
    for p in images:
        img = cv2.imread(p)
        if img is None: 
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if im_size is None:
            im_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not found:
            print("[SKIP] no corners:", p)
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)
        used.append(p)
        print("[OK]", p)

    if len(objpoints) < 10:
        raise SystemExit(f"Недостаточно кадров с углами: {len(objpoints)} (минимум 10–15)")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, im_size, None, None,
        flags=cv2.CALIB_RATIONAL_MODEL
    )

    # Пересчёт ошибки на кадр
    errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        e = np.mean(np.linalg.norm(proj.reshape(-1,2) - imgpoints[i].reshape(-1,2), axis=1))
        errors.append(e)
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))

    print("\n=== CALIBRATION RESULT ===")
    print("Image size: ", im_size)
    print("RMS:       ", ret)
    print("Mean err:  ", mean_err, "px")
    print("Max err:   ", max_err, "px")
    print("K:\n", K)
    print("dist:\n", dist.ravel())

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    np.savez(args.save,
             K=K.astype(np.float32),
             dist=dist.astype(np.float32),
             im_width=im_size[0], im_height=im_size[1],
             pattern_cols=args.cols, pattern_rows=args.rows,
             square_size_m=square_size_m,
             rms=float(ret), mean_err=mean_err, max_err=max_err,
             used_images=np.array(used))
    print("\nSaved ->", args.save)

if __name__ == "__main__":
    main()
