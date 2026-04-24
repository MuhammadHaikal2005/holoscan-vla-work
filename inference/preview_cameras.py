"""
Quick preview of both USB cameras used for inference.

Usage:
    python inference/preview_cameras.py              # cam0=0, cam1=2
    python inference/preview_cameras.py --cam0 0 --cam1 2

Keys:
    q or ESC — quit
"""

import argparse

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam0", type=int, default=0)
    p.add_argument("--cam1", type=int, default=2)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    for cap in (cap0, cap1):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap0.isOpened():
        print(f"ERROR: cannot open camera /dev/video{args.cam0 * 2}")
        return
    if not cap1.isOpened():
        print(f"ERROR: cannot open camera /dev/video{args.cam1 * 2}")
        return

    print(f"cam0 = /dev/video{args.cam0 * 2}   (left window)")
    print(f"cam1 = /dev/video{args.cam1 * 2}   (right window)")
    print("Press 'q' or ESC to quit.")

    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            print("Camera read failed")
            break

        cv2.putText(f0, "cam0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(f1, "cam1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        combined = cv2.hconcat([f0, f1])
        cv2.imshow("cam0 | cam1", combined)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
