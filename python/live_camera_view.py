"""Display DroneA/DroneB front cameras in a separate desktop window.

Press Q or Escape in the monitor window to close it. The simulator may be paused;
the last rendered frame remains viewable in that case.
"""
import time
import airsim
import cv2
import numpy as np

from balloon_detector import detect_balloon, annotate


def frame(client, vehicle):
    data = client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name=vehicle)
    if not data:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    result = detect_balloon(image)
    image = annotate(image, result)
    status = "DETECTED" if result["detected"] else "NO TARGET"
    cv2.putText(image, vehicle + " / front_center / " + status, (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return image


def main():
    client = airsim.MultirotorClient(timeout_value=10)
    # Move the virtual lens beyond the rotor guards so the body does not cover
    # the lower half of the forward image.
    for vehicle in ("DroneA", "DroneB"):
        client.simSetCameraPose("front_center",
                                airsim.Pose(airsim.Vector3r(1.0, 0.0, -0.05), airsim.Quaternionr()),
                                vehicle_name=vehicle)
    cv2.namedWindow("AirSim cameras - Q/Esc to close", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AirSim cameras - Q/Esc to close", 1280, 480)
    while True:
        left, right = frame(client, "DroneA"), frame(client, "DroneB")
        cv2.imshow("AirSim cameras - Q/Esc to close", np.hstack((left, right)))
        key = cv2.waitKey(80) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
