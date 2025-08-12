# 01_capture_chessboard_images.py
import cv2
import os
import time
import argparse

def parse_args():
    ap = argparse.ArgumentParser("Capture chessboard images for calibration")
    ap.add_argument("--out", type=str, default="data/calib", help="Папка для сохранения")
    ap.add_argument("--cam", type=int, default=0, help="Индекс камеры")
    ap.add_argument("--width", type=int, default=640, help="Ширина захвата")
    ap.add_argument("--height", type=int, default=480, help="Высота захвата")
    ap.add_argument("--cols", type=int, default=6, help="ЧИСЛО внутренних углов по горизонтали (у вас 6)")
    ap.add_argument("--rows", type=int, default=9, help="ЧИСЛО внутренних углов по вертикали (у вас 9)")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit("Не удалось открыть камеру")

    pattern_size = (args.cols, args.rows)  # ВНИМАНИЕ: (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

    saved = 0
    print("[INFO] SPACE=сохранить, ESC=выход")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            cv2.putText(vis, "DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(vis, "NOT FOUND (try move/angle/light)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(vis, f"Saved: {saved}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,200,255), 2)
        cv2.imshow("Capture chessboard", vis)

        k = cv2.waitKey(1)
        if k == 27:  # ESC
            break
        elif k == 32 and found:  # SPACE
            fname = os.path.join(args.out, f"img_{int(time.time()*1000)}.png")
            cv2.imwrite(fname, frame)
            saved += 1
            print("[SAVED]", fname)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
