import cv2, json, numpy as np
p = json.load(open("markers.json"))
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
img = cv2.imread("aruco_markers_A4.png", cv2.IMREAD_GRAYSCALE)

print("scale   detected  missing")
for scale, note in [(1.0,"as printed"), (0.25,"~75 dpi"), (0.12,"far"), (0.08,"very far")]:
    s = img if scale==1.0 else cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    c,i,_ = det.detectMarkers(s)
    found = sorted(i.flatten().tolist()) if i is not None else []
    missing = [x for x in p["ids"] if x not in found]
    print(f"{scale:5.2f}   {len(found):2d}/12     {missing if missing else 'none':<20} {note}")

# simulate a poor print: blur + noise + low contrast
s = cv2.resize(img,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
s = cv2.GaussianBlur(s,(3,3),0)
s = np.clip(s.astype(np.int16)*0.6+70 + np.random.normal(0,6,s.shape),0,255).astype(np.uint8)
c,i,_ = det.detectMarkers(s)
print(f"\ndegraded (blur + 60% contrast + noise): {0 if i is None else len(i)}/12 detected")
