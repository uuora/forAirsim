"""Minimal two-drone AirSim smoke test for the Blocks environment."""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import airsim


DRONES = ("DroneA", "DroneB")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"dual_smoke_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("airsim.dual_smoke")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Log file: %s", log_path)
    return logger


def main() -> None:
    logger = configure_logging()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    vehicles = client.listVehicles()
    logger.info("Vehicles reported by AirSim: %s", vehicles)
    missing = [name for name in DRONES if name not in vehicles]
    if missing:
        raise RuntimeError(f"Missing configured vehicles: {missing}")

    for name in DRONES:
        client.enableApiControl(True, vehicle_name=name)
        client.armDisarm(True, vehicle_name=name)

    logger.info("Taking off...")
    futures = [client.takeoffAsync(vehicle_name=name) for name in DRONES]
    for future in futures:
        future.join()

    logger.info("Moving to separated test positions...")
    futures = [
        client.moveToPositionAsync(-4, -4, -4, 2, vehicle_name="DroneA"),
        client.moveToPositionAsync(-4, 4, -4, 2, vehicle_name="DroneB"),
    ]
    for future in futures:
        future.join()

    for name in DRONES:
        state = client.getMultirotorState(vehicle_name=name)
        logger.info("%s position: %s", name, state.kinematics_estimated.position)

    logger.info("Hovering for 3 seconds...")
    for name in DRONES:
        client.hoverAsync(vehicle_name=name).join()
    time.sleep(3)

    logger.info("Returning to home height and disarming...")
    for name in DRONES:
        # A short controlled descent is more reliable than waiting for the
        # simulator's landing detector in a first-run smoke test.
        client.moveToZAsync(-1.0, 1.0, vehicle_name=name).join()
        client.armDisarm(False, vehicle_name=name)
        client.enableApiControl(False, vehicle_name=name)

    logger.info("DUAL_DRONE_SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()
