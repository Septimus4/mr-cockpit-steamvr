"""Find the ELP by capability rather than by index - indices move when devices come and go."""
import cv2
def find_elp(maxdev=6, verbose=True):
    for i in range(maxdev):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if verbose:
            tag = "<-- ELP dual-lens" if (w, h) == (3200, 1200) else ""
            print(f"  index {i}: {w}x{h} {tag}")
        if (w, h) == (3200, 1200):
            return i
    return None
if __name__ == "__main__":
    i = find_elp()
    print(f"\nELP at index {i}" if i is not None else "\nELP NOT FOUND - check the USB connection")
