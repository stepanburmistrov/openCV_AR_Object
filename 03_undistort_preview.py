# 03_undistort_preview.py
import cv2
import numpy as np
import argparse

def parse_args():
    ap = argparse.ArgumentParser("Preview undistortion")
    ap.add_argument("--params", type=str, default="calibration_cam0.npz")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    return ap.parse_args()

def main():
    args = parse_args()
    data = np.load(args.params, allow_pickle=True)
    K = data["K"]
    dist = data["dist"]
    calib_w, calib_h = int(data["im_width"]), int(data["im_height"])

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Не удалось открыть камеру")

    show_undist = True
    print("[INFO] U=toggle undistort, ESC=exit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # Подгоняем K, если текущее разрешение отличается от калибровочного
        K_adj = K.copy()
        if (w, h) != (calib_w, calib_h):
            sx = w / calib_w
            sy = h / calib_h
            K_adj[0,0] *= sx; K_adj[1,1] *= sy
            K_adj[0,2] *= sx; K_adj[1,2] *= sy

        if show_undist:
            newK, roi = cv2.getOptimalNewCameraMatrix(K_adj, dist, (w,h), 1)
            und = cv2.undistort(frame, K_adj, dist, None, newK)
            vis = und
            cv2.putText(vis, "UNDISTORTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            vis = frame.copy()
            cv2.putText(vis, "RAW", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Undistort preview", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        elif k in (ord('u'), ord('U')):
            show_undist = not show_undist

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
