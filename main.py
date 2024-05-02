import cv2
import numpy as np


def traffic_count(camera_indices):
    caps = []
    num_cameras = len(camera_indices)
    counters = [0] * num_cameras  # Initialize the counters list

    # Create a single background subtractor for both cameras
    algo = cv2.createBackgroundSubtractorMOG2()

    # Configure camera properties and initialize capture objects
    for index in camera_indices:
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set lower resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set lower resolution
        caps.append(cap)

    count_line_pos = 150  # Adjust count line position
    line_thickness = 2
    min_width = 40  # Adjust minimum width and height for detection
    min_height = 40
    offset = 6

    while True:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 5)

            img_sub = algo.apply(blur)
            dil = cv2.dilate(img_sub, np.ones((5, 5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dil_2 = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel)
            dil_2 = cv2.morphologyEx(dil_2, cv2.MORPH_CLOSE, kernel)

            # Define a region of interest (ROI) where vehicles are expected
            roi = dil_2[120:240, :]

            contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(frame, (0, count_line_pos), (frame.shape[1], count_line_pos), (0, 255, 0), line_thickness)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                validate_counter = w >= min_width and h >= min_height
                if not validate_counter:
                    continue

                cv2.rectangle(frame, (x, y + 120), (x + w, y + h + 120), (0, 255, 0), 2)
                center = (x + int(w / 2), y + int(h / 2) + 120)

                if count_line_pos - offset < center[1] < count_line_pos + offset:
                    counters[i] += 1
                    print(f'Camera {camera_indices[i]} - Vehicle Counter: {counters[i]}')

                cv2.line(frame, (0, count_line_pos), (frame.shape[1], count_line_pos), (0, 0, 255), line_thickness)

            cv2.imshow(f'Camera {camera_indices[i]}', frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    for cap in caps:
        cap.release()

    return counters


counters = traffic_count([0,1.2,3])
print(counters)
