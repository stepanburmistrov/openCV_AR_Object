#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_ar_aruco_obj_textured.py
AR with textured OBJ (diffuse map_Kd) inside the cube.
- Undistort + ROI
- Stable pose (iterative from previous, IPPE fallback), "always-up" cube
- Renders OBJ as: wireframe / flat-fill / textured triangles (affine per-triangle)
Keys: ESC exit | C cube | W wire | F fill | T textures | +/- scale | Z/X z-bias | Arrows yaw/pitch | Q/E roll
"""
import argparse, os, cv2, numpy as np

# ---------- ArUco resolver ----------
def get_aruco_api(dictionary_name: str):
    ar = cv2.aruco
    name = dictionary_name.upper()
    dic_map = {
        "4X4_50": getattr(ar, "DICT_4X4_50", None),
        "4X4_100": getattr(ar, "DICT_4X4_100", getattr(ar, "DICT_4X_100", None)),
        "4X4_250": getattr(ar, "DICT_4X4_250", getattr(ar, "DICT_4X_250", None)),
        "4X4_1000": getattr(ar, "DICT_4X4_1000", None),
        "5X5_50": getattr(ar, "DICT_5X5_50", None),
        "5X5_100": getattr(ar, "DICT_5X5_100", None),
        "5X5_250": getattr(ar, "DICT_5X5_250", None),
        "5X5_1000": getattr(ar, "DICT_5X5_1000", None),
        "6X6_50": getattr(ar, "DICT_6X6_50", None),
        "6X6_100": getattr(ar, "DICT_6X6_100", None),
        "6X6_250": getattr(ar, "DICT_6X6_250", None),
        "6X6_1000": getattr(ar, "DICT_6X6_1000", None),
        "7X7_50": getattr(ar, "DICT_7X7_50", None),
        "7X7_100": getattr(ar, "DICT_7X7_100", None),
        "7X7_250": getattr(ar, "DICT_7X7_250", None),
        "7X7_1000": getattr(ar, "DICT_7X7_1000", None),
        "APRILTAG_36H11": getattr(ar, "DICT_APRILTAG_36h11", getattr(ar, "DICT_APRILTAG_36H11", None)),
    }
    dic_const = dic_map.get(name)
    if dic_const is None:
        raise SystemExit(f"Unknown dictionary: {dictionary_name}")
    dictionary = ar.getPredefinedDictionary(dic_const) if hasattr(ar, "getPredefinedDictionary") else ar.Dictionary_get(dic_const)
    try: parameters = ar.DetectorParameters()
    except AttributeError: parameters = ar.DetectorParameters_create()
    detector = ar.ArucoDetector(dictionary, parameters) if hasattr(ar, "ArucoDetector") else None
    return ar, dictionary, parameters, detector

# ---------- math ----------
def euler_deg_to_R(yaw, pitch, roll):
    rz = np.deg2rad(yaw); ry = np.deg2rad(pitch); rx = np.deg2rad(roll)
    cz,sz = np.cos(rz), np.sin(rz)
    cy,sy = np.cos(ry), np.sin(ry)
    cx,sx = np.cos(rx), np.sin(rx)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], np.float32)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def r2R(r): R,_ = cv2.Rodrigues(r); return R
def R2r(R): r,_ = cv2.Rodrigues(R); return r

def slerp_R(R_prev, R_curr, alpha):
    if R_prev is None: return R_curr
    R_delta = R_prev.T @ R_curr
    r_delta,_ = cv2.Rodrigues(R_delta)
    R_step,_ = cv2.Rodrigues(r_delta * float(alpha))
    return R_prev @ R_step

# ---------- OBJ + MTL (diffuse) ----------
def load_obj_with_uv_and_mtl(obj_path):
    V=[]; VT=[]; Fv=[]; Fvt=[]; Fmtl=[]; mtl_name=None
    obj_dir = os.path.dirname(obj_path)
    materials = {}  # name -> dict(texture: np.ndarray or None)
    cur_mtl = None

    def load_mtl(path):
        tex = None
        name = None
        cur = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith("#"): continue
                if line.startswith("newmtl "):
                    name = line.split()[1]
                    materials[name] = {"texture": None}
                    cur = materials[name]
                elif line.lower().startswith("map_kd"):
                    tex_file = line.split(None,1)[1].strip().split()[-1]
                    tex_path = tex_file if os.path.isabs(tex_file) else os.path.join(obj_dir, tex_file)
                    img = cv2.imread(tex_path, cv2.IMREAD_COLOR)
                    if img is not None and cur is not None:
                        cur["texture"] = img
        return materials

    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            if line.startswith("mtllib "):
                mtl_filename = line.split()[1]
                mtl_path = mtl_filename if os.path.isabs(mtl_filename) else os.path.join(obj_dir, mtl_filename)
                if os.path.exists(mtl_path):
                    load_mtl(mtl_path)
            elif line.startswith("usemtl "):
                cur_mtl = line.split()[1]
            elif line.startswith("v "):
                _,x,y,z = line.split()[:4]
                V.append([float(x),float(y),float(z)])
            elif line.startswith("vt "):
                parts = line.split()
                # vt u v [w]
                u = float(parts[1]); v = float(parts[2])
                VT.append([u,v])
            elif line.startswith("f "):
                parts=line.split()[1:]
                vidx=[]; vtidx=[]
                for p in parts:
                    # v/vt/vn or v//vn or v/vt
                    toks = p.split("/")
                    vi = int(toks[0]) - 1
                    vti = int(toks[1]) - 1 if len(toks)>=2 and toks[1] != "" else -1
                    vidx.append(vi); vtidx.append(vti)
                # triangulate
                for i in range(1, len(vidx)-1):
                    Fv.append([vidx[0], vidx[i], vidx[i+1]])
                    Fvt.append([vtidx[0], vtidx[i], vtidx[i+1]])
                    Fmtl.append(cur_mtl)
    V=np.array(V,np.float32)
    VT=np.array(VT,np.float32) if len(VT)>0 else None
    Fv=np.array(Fv,np.int32); Fvt=np.array(Fvt,np.int32) if VT is not None else None
    return V, VT, Fv, Fvt, Fmtl, materials

def normalize_center(V):
    Vc = V.copy()
    c = (Vc.min(0)+Vc.max(0))*0.5
    Vc -= c
    return Vc

# ---------- pose helpers ----------
def reproj_err(obj_pts, rvec, tvec, K, img_pts):
    proj,_ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
    return float(np.mean(np.linalg.norm(proj.reshape(-1,2)-img_pts.reshape(-1,2), axis=1)))

def iterative_from_prev(obj_pts, img_pts, K, prev_rvec, prev_tvec):
    r = prev_rvec.copy(); t = prev_tvec.copy()
    ok, r, t = cv2.solvePnP(obj_pts, img_pts, K, None, r, t, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    e = reproj_err(obj_pts, r, t, K, img_pts)
    return (r,t,e,-1)

def ippe_candidates(obj_pts, img_pts, K):
    cand=[]
    if hasattr(cv2,"solvePnPGeneric") and hasattr(cv2,"SOLVEPNP_IPPE_SQUARE"):
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok and rvecs is not None:
            for i,(rv,tv) in enumerate(zip(rvecs,tvecs)):
                e=reproj_err(obj_pts, rv, tv, K, img_pts)
                cand.append((rv,tv,e,i))
    else:
        ok, rv, tv = cv2.solvePnP(obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            e=reproj_err(obj_pts, rv, tv, K, img_pts)
            cand.append((rv,tv,e,0))
    return cand

def pick_stable(cands, prev_pose, prefer_facing=True):
    if not cands: return None
    if prev_pose is None:
        best=None; best_score=1e9
        for rv,tv,e,idx in cands:
            zc = r2R(rv)[2,2]
            score = e + (5.0 if (prefer_facing and zc>0) else 0.0)
            if score<best_score: best=(rv,tv,e,idx); best_score=score
        return best
    Rprev=r2R(prev_pose[0]); tprev=prev_pose[1]
    best=None; best_score=1e9
    for rv,tv,e,idx in cands:
        R=r2R(rv); zc=R[2,2]
        Rrel=Rprev.T@R; rrel,_=cv2.Rodrigues(Rrel)
        ang=float(np.linalg.norm(rrel))*180.0/np.pi
        d=float(np.linalg.norm(tv-tprev))
        score = e + 0.03*ang + 2.0*d + (10.0 if (prefer_facing and zc>0) else 0.0)
        if score<best_score: best=(rv,tv,e,idx); best_score=score
    return best

# ---------- rendering ----------
def draw_cube_toward_camera(frame, K, rvec, tvec, size_m, thickness=1):
    half=size_m/2.0
    zc = r2R(rvec)[2,2]
    top_z = -size_m if zc>0 else +size_m
    bottom=np.float32([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]])
    top   =np.float32([[-half,-half,top_z],[half,-half,top_z],[half,half,top_z],[-half,half,top_z]])
    P,_=cv2.projectPoints(np.vstack((bottom,top)), rvec, tvec, K, None)
    p=P.reshape(-1,2).astype(int)
    cv2.drawContours(frame,[p[:4]],-1,(0,255,0),thickness)
    cv2.drawContours(frame,[p[4:]],-1,(255,0,0),thickness)
    for i in range(4): cv2.line(frame, tuple(p[i]), tuple(p[i+4]), (0,0,255), thickness)

def draw_model_wire_fill(frame, K, rvec, tvec, V, F, Rm, scale, z_bias_m, cube_size_m, wire=True, fill=False):
    Rmkr=r2R(rvec)
    up_sign = -1.0 if Rmkr[2,2]>0 else +1.0
    offset = np.array([0,0, up_sign*(cube_size_m*0.5) + z_bias_m], np.float32)
    Vm = (V*scale)@Rm.T + offset
    Vp,_ = cv2.projectPoints(Vm, rvec, tvec, K, None)
    P = Vp.reshape(-1,2)
    Vc = (Vm @ Rmkr.T) + tvec.reshape(1,3)
    depths = Vc[F].mean(axis=1)[:,2]
    order = np.argsort(depths)[::-1]
    if fill:
        for f in order:
            tri = P[F[f]].astype(np.int32)
            cv2.fillConvexPoly(frame, tri, (60,140,255), lineType=cv2.LINE_AA)
    if wire:
        done=set()
        for f in order:
            a,b,c=F[f]
            for u,v in ((a,b),(b,c),(c,a)):
                key=(min(u,v),max(u,v)); 
                if key in done: continue
                done.add(key)
                p1=tuple(P[u].astype(int)); p2=tuple(P[v].astype(int))
                cv2.line(frame,p1,p2,(0,255,255),1, lineType=cv2.LINE_AA)

def warp_triangle(dst, tex, t_src, t_dst):
    """
    t_src: 3x2 float32 в координатах текстуры (пиксели)
    t_dst: 3x2 float32 в координатах изображения (пиксели)
    Рисует аффинно натянутый треугольник с учётом выхода за границы кадра.
    """
    H, W = dst.shape[:2]

    # Прямоугольники вокруг треугольников
    r_dst = cv2.boundingRect(t_dst)  # (x, y, w, h) в кадре
    r_src = cv2.boundingRect(t_src)  # (x, y, w, h) в текстуре

    # Вырожденные случаи
    if r_dst[2] <= 0 or r_dst[3] <= 0 or r_src[2] <= 0 or r_src[3] <= 0:
        return

    # Локальные координаты вершин в пределах своих прямоугольников
    dst_local = t_dst.copy()
    dst_local[:, 0] -= r_dst[0]
    dst_local[:, 1] -= r_dst[1]

    src_local = t_src.copy()
    src_local[:, 0] -= r_src[0]
    src_local[:, 1] -= r_src[1]

    # Вырезаем кусок текстуры под источник
    tex_crop = tex[r_src[1]:r_src[1]+r_src[3], r_src[0]:r_src[0]+r_src[2]]
    if tex_crop.size == 0:
        return

    # Аффинное преобразование в полное окно r_dst
    M = cv2.getAffineTransform(src_local.astype(np.float32),
                               dst_local.astype(np.float32))
    warped_full = cv2.warpAffine(
        tex_crop, M, (r_dst[2], r_dst[3]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    # Полная маска под r_dst
    mask_full = np.zeros((r_dst[3], r_dst[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask_full, dst_local.astype(np.int32), 255,
                       lineType=cv2.LINE_AA)

    # --- ВАЖНО: зажимаем r_dst в пределах изображения и
    #            выравниваем вырезанные области одинаково ---

    x0, y0, w, h = r_dst
    x1, y1 = x0 + w, y0 + h

    x0c = max(0, x0); y0c = max(0, y0)
    x1c = min(W, x1); y1c = min(H, y1)

    wc = x1c - x0c
    hc = y1c - y0c
    if wc <= 0 or hc <= 0:
        return

    # Насколько отрезали слева/сверху относительно r_dst
    dx = x0c - x0
    dy = y0c - y0

    # Кропим warped и mask тем же сдвигом и размером
    warped = warped_full[dy:dy+hc, dx:dx+wc]
    mask   = mask_full  [dy:dy+hc, dx:dx+wc]

    # ROI точно такого же размера
    roi = dst[y0c:y1c, x0c:x1c]

    # Сборка
    # mask — uint8, одинаковый (hc, wc) → всё ок
    inv = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    add = cv2.bitwise_and(warped, warped, mask=mask)
    roi[...] = inv + add


def draw_model_textured(frame, K, rvec, tvec, V, VT, Fv, Fvt, Fmtl, materials, Rm, scale, z_bias_m, cube_size_m):
    if VT is None or Fvt is None or not materials: 
        return  # nothing to do
    Rmkr=r2R(rvec)
    up_sign = -1.0 if Rmkr[2,2]>0 else +1.0
    offset = np.array([0,0, up_sign*(cube_size_m*0.5) + z_bias_m], np.float32)
    Vm = (V*scale)@Rm.T + offset
    # project model-space vertices (for P) and get camera-space for depth sort
    Vp,_ = cv2.projectPoints(Vm, rvec, tvec, K, None)
    P = Vp.reshape(-1,2).astype(np.float32)

    Vc = (Vm @ Rmkr.T) + tvec.reshape(1,3)
    depths = Vc[Fv].mean(axis=1)[:,2]
    order = np.argsort(depths)[::-1]  # far -> near

    # For each face: get its material texture if available; map VT->pixels then warp
    for f in order:
        tri_idx = Fv[f]     # indices into V
        uv_idx  = Fvt[f]    # indices into VT
        matname = Fmtl[f]
        mat = materials.get(matname) if matname in materials else None
        if mat is None or mat.get("texture") is None or (uv_idx<0).any(): 
            # fall back: simple fill
            tri = P[tri_idx].astype(np.int32)
            cv2.fillConvexPoly(frame, tri, (60,140,255), lineType=cv2.LINE_AA)
            continue
        tex = mat["texture"]
        th, tw = tex.shape[:2]
        # Destination triangle in image
        t_dst = P[tri_idx].astype(np.float32)
        # Source triangle in texture pixels (note: OBJ v has origin bottom-left, OpenCV uses top-left;
        # many OBJs come with V increasing upward; to be safe, flip v: v_tex = (1 - v) * th)
        uv = VT[uv_idx].copy().astype(np.float32)
        #uv[:,1] = 1.0 - uv[:,1]
        t_src = np.stack([uv[:,0]*tw, uv[:,1]*th], axis=1).astype(np.float32)
        warp_triangle(frame, tex, t_src, t_dst)

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="calibration_cam0.npz")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--dict", type=str, default="4X4_50")
    ap.add_argument("--marker_size_mm", type=float, default=38)
    ap.add_argument("--obj", type=str, default="sample_textured_cube.obj",)
    #ap.add_argument("--obj", type=str, default="sample.obj",)
    ap.add_argument("--fit_to_cube", type=int, default=1)
    ap.add_argument("--init_yaw", type=float, default=0.0)
    ap.add_argument("--init_pitch", type=float, default=0.0)
    ap.add_argument("--init_roll", type=float, default=0.0)
    ap.add_argument("--textured", type=int, default=1)
    args=ap.parse_args()

    data=np.load(args.params, allow_pickle=True)
    K0=data["K"].astype(np.float32); dist=data["dist"].astype(np.float32)
    calib_w,calib_h=int(data["im_width"]),int(data["im_height"])

    cap=cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    ok,first=cap.read()
    if not ok: raise SystemExit("Camera failed")
    h,w=first.shape[:2]

    K=K0.copy()
    if (w,h)!=(calib_w,calib_h):
        sx,sy=w/calib_w,h/calib_h
        K[0,0]*=sx; K[1,1]*=sy; K[0,2]*=sx; K[1,2]*=sy

    newK,roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),alpha=0)
    map1,map2=cv2.initUndistortRectifyMap(K,dist,None,newK,(w,h),cv2.CV_16SC2)
    x,y,w2,h2=roi; use_roi=(w2>0 and h2>0)
    newK_sub=newK.copy()
    if use_roi: newK_sub[0,2]-=x; newK_sub[1,2]-=y

    ar, dictionary, parameters, detector = get_aruco_api(args.dict)

    size_m=args.marker_size_mm/1000.0
    half=size_m/2.0
    obj_pts=np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]],np.float32)

    # OBJ + MTL
    V, VT, Fv, Fvt, Fmtl, materials = load_obj_with_uv_and_mtl(args.obj)
    V = normalize_center(V)
    if args.fit_to_cube:
        bb=(V.max(0)-V.min(0)).max()
        scale=(size_m*0.85)/float(bb if bb!=0 else 1.0)
    else:
        scale=0.03
    yaw,pitch,roll=args.init_yaw,args.init_pitch,args.init_roll
    Rm = euler_deg_to_R(yaw,pitch,roll)

    prev_pose={}
    show_cube=True; wire=False; fill=False; textured=bool(args.textured)
    z_bias_m=0.0

    print("ESC exit | C cube | W wire | F fill | T textures | +/- scale | Z/X z | I/K/J/L yaw/pitch | Q/E roll")

    while True:
        ok,raw=cap.read()
        if not ok: break
        ud=cv2.remap(raw,map1,map2,cv2.INTER_LINEAR)
        view=ud[y:y+h2,x:x+w2] if use_roi else ud
        gray=cv2.cvtColor(view,cv2.COLOR_BGR2GRAY)

        if detector is not None:
            corners,ids,rej=detector.detectMarkers(gray)
        else:
            corners,ids,rej=ar.detectMarkers(gray,dictionary,parameters=parameters)

        if ids is not None and len(ids)>0:
            ar.drawDetectedMarkers(view,corners,ids)
            for c, mid in zip(corners, ids.flatten()):
                pid=int(mid); img_pts=c.reshape(4,2).astype(np.float32)

                # Pose: iterative from previous, fallback IPPE
                if pid in prev_pose:
                    pose = iterative_from_prev(obj_pts, img_pts, newK_sub, prev_pose[pid][0], prev_pose[pid][1])
                    if pose is None or pose[2] > 6.0:
                        best = pick_stable(ippe_candidates(obj_pts,img_pts,newK_sub), prev_pose[pid], True)
                        if best is not None: pose=best
                else:
                    pose = pick_stable(ippe_candidates(obj_pts,img_pts,newK_sub), None, True)

                if pose is None: continue
                rvec,tvec,e,_=pose
                if hasattr(cv2,"solvePnPRefineLM"):
                    cv2.solvePnPRefineLM(obj_pts, img_pts.reshape(-1,1,2), newK_sub, None, rvec, tvec)
                # small smoothing
                if pid in prev_pose:
                    R_sm=slerp_R(r2R(prev_pose[pid][0]), r2R(rvec), 0.25)
                    rvec=R2r(R_sm); tvec=0.75*prev_pose[pid][1]+0.25*tvec
                prev_pose[pid]=(rvec,tvec)

                if show_cube:
                    draw_cube_toward_camera(view,newK_sub,rvec,tvec,size_m,thickness=1)

                # model orientation (user can change keys -> update Rm)
                # draw textured if available (preferred), else wire/fill
                if textured and (VT is not None) and (Fvt is not None) and materials:
                    draw_model_textured(view, newK_sub, rvec, tvec, V, VT, Fv, Fvt, Fmtl, materials, Rm, scale, z_bias_m, size_m)
                else:
                    draw_model_wire_fill(view, newK_sub, rvec, tvec, V, Fv, Rm, scale, z_bias_m, size_m, wire=wire, fill=fill)

                cv2.putText(view,f"ID:{pid}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210,210,210), 2)

        cv2.imshow("AR OBJ (textured)", view)
        k=cv2.waitKey(1)&0xFF
        if k==27: break
        elif k in (ord('c'),ord('C')): show_cube=not show_cube
        elif k in (ord('w'),ord('W')): wire=not wire
        elif k in (ord('f'),ord('F')): fill=not fill
        elif k in (ord('t'),ord('T')): textured = not textured
        elif k in (ord('+'),ord('=')): scale *= 1.05
        elif k in (ord('-'),ord('_')): scale /= 1.05
        elif k in (ord('z'),ord('Z')): z_bias_m += 0.002
        elif k in (ord('x'),ord('X')): z_bias_m -= 0.002
        elif k in (ord('i'),ord('I')): yaw -= 5; Rm = euler_deg_to_R(yaw,pitch,roll)
        elif k in (ord('l'),ord('L')): yaw += 5; Rm = euler_deg_to_R(yaw,pitch,roll)
        elif k in (ord('j'),ord('J')): pitch += 5; Rm = euler_deg_to_R(yaw,pitch,roll)
        elif k in (ord('k'),ord('K')): pitch -= 5; Rm = euler_deg_to_R(yaw,pitch,roll)
        elif k in (ord('q'),ord('Q')): roll -= 5; Rm = euler_deg_to_R(yaw,pitch,roll)
        elif k in (ord('e'),ord('E')): roll += 5; Rm = euler_deg_to_R(yaw,pitch,roll)

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
