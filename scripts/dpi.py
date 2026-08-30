"""
Make the process per-monitor DPI aware, and SAY SO if it fails.

Getting this wrong is silent: Windows hands back logical coordinates scaled by the
display setting, so a "1:1" window is quietly resized.  Every caller must check the
returned mode rather than assume it worked.
"""
import ctypes

def make_dpi_aware():
    u32, shcore = ctypes.windll.user32, None
    try: shcore = ctypes.windll.shcore
    except Exception: pass
    # per-monitor v2: handle is POINTER-sized, must not be passed as c_int
    try:
        fn = u32.SetProcessDpiAwarenessContext
        fn.argtypes = [ctypes.c_void_p]; fn.restype = ctypes.c_bool
        if fn(ctypes.c_void_p(-4)): return "per-monitor-v2"
    except Exception: pass
    try:
        if shcore is not None and shcore.SetProcessDpiAwareness(2) == 0: return "per-monitor"
    except Exception: pass
    try:
        if u32.SetProcessDPIAware(): return "system"
    except Exception: pass
    return "NONE"

def monitors():
    from ctypes import wintypes
    class MONITORINFOEXW(ctypes.Structure):
        _fields_=[("cbSize",wintypes.DWORD),("rcMonitor",wintypes.RECT),
                  ("rcWork",wintypes.RECT),("dwFlags",wintypes.DWORD),("szDevice",wintypes.WCHAR*32)]
    out=[]
    P = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                           ctypes.POINTER(wintypes.RECT), ctypes.c_double)
    def cb(h,hdc,lprc,d):
        mi=MONITORINFOEXW(); mi.cbSize=ctypes.sizeof(MONITORINFOEXW)
        ctypes.windll.user32.GetMonitorInfoW(ctypes.c_ulong(h), ctypes.byref(mi))
        r=mi.rcMonitor
        out.append(dict(dev=mi.szDevice, x=r.left, y=r.top,
                        w=r.right-r.left, h=r.bottom-r.top, primary=bool(mi.dwFlags&1)))
        return 1
    ctypes.windll.user32.EnumDisplayMonitors(0,0,P(cb),0)
    return out

if __name__ == "__main__":
    mode = make_dpi_aware()
    print(f"  DPI awareness: {mode}")
    if mode == "NONE": print("  WARNING: coordinates below are SCALED, not physical")
    for i,m in enumerate(monitors()):
        o = "portrait" if m['h']>m['w'] else "landscape"
        print(f"  monitor {i}: {m['dev']}  origin ({m['x']},{m['y']})  {m['w']}x{m['h']} {o}"
              f"{'  PRIMARY' if m['primary'] else ''}")
