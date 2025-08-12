#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_detect_aruco_and_cube_follow_v2.py
- UNDISTORT (alpha=0) + ROI
- Detect on undistorted view
- Pose from solvePnPGeneric (IPPE)
- Orientation lock across IPPE branches
- ALWAYS extrude cube TOWARD camera (auto z-sign)
- No HOLD drawing: if marker not detected this frame -> no cube
- Smoothing: SLERP (rotation) + EMA (translation), weaker on noisy frames
"""

import argparse
import cv2
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser("ArUco + Cube (follow v2: up & reacquire)")
    ap.add_argument("--params", type=str, default="calibration_cam0.npz")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--marker_size_mm", type=float, default=38.0)
    ap.add_argument("--dict", type=str, default="4X4_50")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--alpha_bad", type=float, default=0.12)
    ap.add_argument("--err_bad", type=float, default=5.0)
    ap.add_argument("--min_side_px", type=float, default=14.0)
    return ap.parse_args()

def get_aruco_api(dictionary_name: str):
    ar = cv2.aruco
    name = dictionary_name.upper()
    dic_map = {
        "4X4_50": ar.DICT_4X4_50, "4X4_100": ar.DICT_4X4_100, "4X4_250": ar.DICT_4X4_250, "4X4_1000": ar.DICT_4X4_1000,
        "5X5_50": ar.DICT_5X5_50, "5X5_100": ar.DICT_5X5_100, "5X5_250": ar.DICT_5X5_250, "5X5_1000": ar.DICT_5X5_1000,
        "6X6_50": ar.DICT_6X6_50, "6X6_100": ar.DICT_6X6_100, "6X6_250": ar.DICT_6X6_250, "6X6_1000": ar.DICT_6X6_1000,
        "7X7_50": ar.DICT_7X7_50, "7X7_100": ar.DICT_7X7_100, "7X7_250": ar.DICT_7X7_250, "7X7_1000": ar.DICT_7X7_1000,
        "APRILTAG_36H11": getattr(ar, "DICT_APRILTAG_36h11", None)
    }
    dic_const = dic_map.get(name)
    if dic_const is None:
        raise SystemExit("Unknown dictionary: " + dictionary_name)
    dictionary = ar.getPredefinedDictionary(dic_const) if hasattr(ar, "getPredefinedDictionary") \
                 else ar.Dictionary_get(dic_const)
    try:
        parameters = ar.DetectorParameters()
    except AttributeError:
        parameters = ar.DetectorParameters_create()
    detector = ar.ArucoDetector(dictionary, parameters) if hasattr(ar, "ArucoDetector") else None
    return ar, dictionary, parameters, detector

def r2R(r): R,_ = cv2.Rodrigues(r); return R
def R2r(R): r,_ = cv2.Rodrigues(R); return r

def slerp_R(R_prev, R_curr, alpha):
    if R_prev is None: return R_curr
    R_delta = R_prev.T @ R_curr
    r_delta,_ = cv2.Rodrigues(R_delta)
    r_delta = r_delta * float(alpha)
    R_step,_ = cv2.Rodrigues(r_delta)
    return R_prev @ R_step

def solve_ippe(obj_pts, img_pts, K):
    # Return list of (rvec,tvec,err,idx)
    out = []
    if hasattr(cv2, "solvePnPGeneric") and hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok and rvecs is not None:
            for i,(rv,tv) in enumerate(zip(rvecs, tvecs)):
                if tv[2] <= 0: continue
                proj,_ = cv2.projectPoints(obj_pts, rv, tv, K, None)
                err = float(np.mean(np.linalg.norm(proj.reshape(-1,2) - img_pts.reshape(-1,2), axis=1)))
                out.append((rv,tv,err,i))
    else:
        ok, rv, tv = cv2.solvePnP(obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok and tv[2] > 0:
            proj,_ = cv2.projectPoints(obj_pts, rv, tv, K, None)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1,2) - img_pts.reshape(-1,2), axis=1)))
            out.append((rv,tv,err,0))
    return out

def pick_with_lock(cands, K, prev_pose, pref_idx, size_m):
    # Score candidates by error + closeness + lock
    if not cands: return None
    if prev_pose is None and pref_idx is None:
        return min(cands, key=lambda x: x[2])

    best = None; best_score = 1e9
    for rv,tv,err,idx in cands:
        score = err
        if pref_idx is not None and idx != pref_idx:
            score *= 1.8
        if prev_pose is not None:
            R_prev = r2R(prev_pose[0]); R = r2R(rv)
            R_rel = R_prev.T @ R
            r_rel,_ = cv2.Rodrigues(R_rel)
            ang = float(np.linalg.norm(r_rel)) * 180.0/np.pi
            d = float(np.linalg.norm(tv - prev_pose[1]))
            score += 0.03*ang + 0.5*(d/max(size_m,1e-6))
        if score < best_score:
            best = (rv,tv,err,idx); best_score = score
    return best

def main():
    args = parse_args()

    data = np.load(args.params, allow_pickle=True)
    K0 = data["K"].astype(np.float32)
    dist = data["dist"].astype(np.float32)
    calib_w, calib_h = int(data["im_width"]), int(data["im_height"])

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened(): raise SystemExit("Cannot open camera")
    ok, first = cap.read()
    if not ok: raise SystemExit("Camera returned no frame")
    h,w = first.shape[:2]

    # scale K
    K = K0.copy()
    if (w,h)!=(calib_w,calib_h):
        sx,sy = w/calib_w, h/calib_h
        K[0,0]*=sx; K[1,1]*=sy; K[0,2]*=sx; K[1,2]*=sy

    # UNDISTORT
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w,h), cv2.CV_16SC2)
    x,y,w2,h2 = roi; use_roi = (w2>0 and h2>0)
    newK_sub = newK.copy()
    if use_roi: newK_sub[0,2]-=x; newK_sub[1,2]-=y

    # ArUco
    ar, dictionary, parameters, detector = get_aruco_api(args.dict)

    # Model
    size_m = args.marker_size_mm/1000.0
    half = size_m/2.0
    obj_pts = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]], np.float32)
    cube_bottom = np.float32([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]])

    # state
    prev_pose = {}    # id -> (rvec,tvec)
    pref_idx = {}     # id -> 0/1

    while True:
        ok, raw = cap.read()
        if not ok: break
        ud = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
        ud_view = ud[y:y+h2, x:x+w2] if use_roi else ud
        gray = cv2.cvtColor(ud_view, cv2.COLOR_BGR2GRAY)

        if detector is not None:
            corners, ids, rej = detector.detectMarkers(gray)
        else:
            corners, ids, rej = ar.detectMarkers(gray, dictionary, parameters=parameters)

        status = ""

        if ids is not None and len(ids)>0:
            ar.drawDetectedMarkers(ud_view, corners, ids)

            for c, mid in zip(corners, ids.flatten()):
                pid = int(mid)
                img_pts = c.reshape(4,2).astype(np.float32)

                side_px = float(max(np.linalg.norm(img_pts[0]-img_pts[1]), np.linalg.norm(img_pts[1]-img_pts[2])))
                if side_px < args.min_side_px and pid not in prev_pose:
                    status = f"ID:{pid} init skipped: small {side_px:.1f}px"
                    continue

                cands = solve_ippe(obj_pts, img_pts, newK_sub)
                best = pick_with_lock(cands, newK_sub, prev_pose.get(pid), pref_idx.get(pid), size_m)
                if best is None:
                    # marker есть, но позу не получили — пропускаем кадр
                    continue

                rvec, tvec, err, idx = best

                # lock branch
                if pid not in pref_idx:
                    pref_idx[pid] = idx

                # refine
                if hasattr(cv2, "solvePnPRefineLM"):
                    cv2.solvePnPRefineLM(obj_pts, img_pts.reshape(-1,1,2), newK_sub, None, rvec, tvec)

                # smoothing
                alpha = args.alpha_bad if err > args.err_bad else args.alpha
                if pid in prev_pose:
                    R_sm = slerp_R(r2R(prev_pose[pid][0]), r2R(rvec), alpha)
                    rvec = R2r(R_sm)
                    tvec = (1.0-alpha)*prev_pose[pid][1] + alpha*tvec

                prev_pose[pid] = (rvec, tvec)

                # ---------- DRAW ----------
                # 1) projected marker for sanity
                proj_m,_ = cv2.projectPoints(obj_pts, rvec, tvec, newK_sub, None)
                pm = proj_m.reshape(-1,2).astype(int)
                cv2.polylines(ud_view, [pm], True, (0,255,255), 2)

                # 2) axes
                if hasattr(ar, "drawAxis"):
                    ar.drawAxis(ud_view, newK_sub, None, rvec, tvec, size_m/2)
                else:
                    cv2.drawFrameAxes(ud_view, newK_sub, None, rvec, tvec, size_m/2)

                # 3) cube that always extrudes TOWARD the camera
                R = r2R(rvec)
                zc = R[2,2]  # z-component of marker Z in camera coords
                top_z = (-size_m) if zc > 0 else (+size_m)  # if +Z increases camera Z -> push top to negative Z
                cube_top = np.float32([[-half,-half,top_z],[half,-half,top_z],[half,half,top_z],[-half,half,top_z]])
                cube_pts3d = np.vstack((cube_bottom, cube_top))

                proj,_ = cv2.projectPoints(cube_pts3d, rvec, tvec, newK_sub, None)
                p = proj.reshape(-1,2).astype(int)
                cv2.drawContours(ud_view, [p[:4]], -1, (0,255,0), 2)
                cv2.drawContours(ud_view, [p[4:]], -1, (255,0,0), 2)
                for i in range(4): cv2.line(ud_view, tuple(p[i]), tuple(p[i+4]), (0,0,255), 2)

                status = f"ID:{pid} side:{side_px:.1f}px err:{err:.2f} idx:{idx} zc:{zc:+.2f} alpha:{alpha:.2f}"

        # If marker completely lost this frame -> we DO NOT draw old cube; state stays for next re-acquire

        if status:
            cv2.putText(ud_view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210,210,210), 2)
        cv2.putText(ud_view, f"Res: {ud_view.shape[1]}x{ud_view.shape[0]}  (calib {calib_w}x{calib_h})",
                    (10, ud_view.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("ArUco + Cube (follow v2: up & reacquire)", ud_view)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
