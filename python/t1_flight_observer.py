"""Airborne receipt wrapper around the RGB-only continuous observer.

The core observer has no flight API. This wrapper corrects its ground-pilot
receipt label; neither component grants motion permission.
"""
from t1_continuous_semantic_observer import create_observer


class FlightObserver:
    def __init__(self, detector_config_path):
        self.core = create_observer(detector_config_path)
        self.close_receipt = None

    def begin(self, *args, **kwargs):
        return self.core.begin(*args, **kwargs)

    def process(self, observation):
        result = dict(self.core.process(observation))
        result.update(ground_only=False, platform_stage="AIRBORNE_RGB_OBSERVATION",
                      flight_authorized=False,
                      motion_authority="separate_current_frame_and_self_state_guard")
        return result

    def close(self):
        self.core.close()
        self.close_receipt = dict(self.core.close_receipt)
        self.close_receipt.update(ground_only=False, platform_stage="AIRBORNE_RGB_OBSERVATION",
                                 flight_authorized=False)

