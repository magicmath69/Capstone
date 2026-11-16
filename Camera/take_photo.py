"""Requires opencv-python. Run: python Camera/take_photo.py [-i CAMERA_INDEX]."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Take photos with the Arducam.")
    parser.add_argument("-i", "--index", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "photos"
    window = "Arducam - s: save | q / Esc: quit"
    cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)

    try:
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {args.index}. Try -i 1 if needed.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        cap.set(cv2.CAP_PROP_FPS, 30)

        output_dir.mkdir(exist_ok=True)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 800)
        print("Click the camera window, then press s to save or q / Esc to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Cannot read a frame from the camera.")
                break

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                filename = datetime.now().strftime("photo_%Y-%m-%d_%H-%M-%S_%f.png")
                path = output_dir / filename
                if cv2.imwrite(str(path), frame):
                    print(f"Saved {path}")
                else:
                    print(f"Failed to save {path}")
            elif key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
