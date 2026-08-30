import cv2, sys, time
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    sys.exit("FAIL: cannot open camera 0 (passthrough layer or another app may hold it)")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FPS, 60)
ok, f = cap.read()
if not ok: sys.exit("FAIL: opened but no frame")
print(f"negotiated : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f}")
print(f"frame shape: {f.shape}  -> {f.shape[1]//2}x{f.shape[0]} per eye")
n=0; t=time.time()
while time.time()-t < 3.0:
    if cap.read()[0]: n+=1
print(f"measured   : {n/(time.time()-t):.1f} fps")
cap.release()
print("OK" if f.shape[1]==3200 else "WARNING: not 3200 wide")
