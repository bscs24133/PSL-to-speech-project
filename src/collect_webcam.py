# src/collect_webcam.py
import cv2, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from inference import ROI_X1, ROI_Y1, ROI_X2, ROI_Y2

SIGNS = [
    "1-Hay","Ain","Alif","Bay","Byeh","Chay","Cyeh","Daal","Dal",
    "Dochahay","Fay","Gaaf","Ghain","Hamza","Kaf","Khay","Kiaf",
    "Lam","Meem","Nuun","Nuungh","Pay","Ray","Say","Seen","Sheen",
    "Suad","Taay","Tay","Tuey","Wao","Zaal","Zaey","Zay","Zuad","Zuey"
]

SAMPLES_PER_SIGN = 40
SAVE_DIR         = "webcam_train"

cap = cv2.VideoCapture(0)
print("SPACE = save sample | N = skip to next sign | Q = quit")
print(f"Need {SAMPLES_PER_SIGN} samples per sign, {len(SIGNS)} signs total\n")

sign_idx = 0

while sign_idx < len(SIGNS):
    sign     = SIGNS[sign_idx]
    save_dir = os.path.join(SAVE_DIR, sign)
    os.makedirs(save_dir, exist_ok=True)

    # Resume from existing count
    count = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
    if count >= SAMPLES_PER_SIGN:
        print(f"[SKIP] {sign} already done ({count} samples)")
        sign_idx += 1
        continue

    print(f"\n[{sign_idx+1}/{len(SIGNS)}] Show sign: {sign}  ({count}/{SAMPLES_PER_SIGN})")

    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

        display = frame.copy()
        cv2.rectangle(display, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)

        # Show ROI preview in corner
        preview = cv2.resize(roi, (120, 120))
        display[10:130, 10:130] = preview

        cv2.putText(display, f"Sign: {sign}", (140, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"Saved: {count}/{SAMPLES_PER_SIGN}", (140, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display, f"Sign {sign_idx+1} of {len(SIGNS)}", (140, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, "SPACE=save  N=next  Q=quit", (10, display.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Flash green when saved
        cv2.imshow("Collect PSL Signs", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            path = os.path.join(save_dir, f"{count:04d}.jpg")
            cv2.imwrite(path, roi)
            count += 1
            print(f"  {sign}: {count}/{SAMPLES_PER_SIGN}", end='\r')

            # Flash feedback
            flash = display.copy()
            cv2.rectangle(flash, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 4)
            cv2.putText(flash, "SAVED!", (ROI_X1+50, ROI_Y1+140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("Collect PSL Signs", flash)
            cv2.waitKey(120)

        elif key == ord('n'):
            print(f"\n  Skipping {sign} (saved {count})")
            break

        elif key == ord('q'):
            print("\nQuitting — run again to resume from where you left off")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    sign_idx += 1
    print(f"\n  Done with {sign}!")

cap.release()
cv2.destroyAllWindows()
print(f"\nAll done! Samples saved to '{SAVE_DIR}/'")
print("Now run: python src/finetune.py")