def open_camera(index: int = 0, width: int = 1280, height: int = 720, req_fps: int = 60) -> cv.VideoCapture:
    backends = [cv.CAP_DSHOW, cv.CAP_MSMF, cv.CAP_ANY]
    last_err = None

    def try_open(idx, backend):
        cap = cv.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            return None
        # Request props
        cap.set(cv.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv.CAP_PROP_FPS,          req_fps)
        # Verify by actually reading one frame (driver FPS lies sometimes)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return None
        return cap

    # First try the user-provided index quickly across backends
    for backend in backends:
        cap = try_open(index, backend)
        if cap:
            print(f"[video_io] Opened camera index {index} with backend {backend}")
            return cap

    # If that failed, scan indices 0-5 across backends
    for idx in range(0, 6):
        for backend in backends:
            cap = try_open(idx, backend)
            if cap:
                print(f"[video_io] Opened camera index {idx} with backend {backend}")
                return cap

    raise RuntimeError("No camera found. Tried indices 0-5 with DSHOW/MSMF/ANY. "
                       "Close other apps and check Windows camera privacy settings.")
