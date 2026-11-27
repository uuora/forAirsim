"""Read-only AirSim actuator sampling on a connection separate from control."""
import json
import math
from threading import Event, Thread
import time


class RotorRecorder:
    rate_hz = 10

    def __init__(self, options, vehicle, start_ns):
        self.start_ns = start_ns
        self.stop = Event()
        self.rows = []
        self.errors = []
        self.duplicates = 0
        self.thread = Thread(target=self._poll, args=(options, vehicle), daemon=True)
        self.thread.start()

    def _poll(self, options, vehicle):
        import airsim

        client = None
        try:
            client = airsim.MultirotorClient(ip=options['host'], port=options['port'], timeout_value=.6)
            previous = self.start_ns - 1
            while not self.stop.is_set():
                began = time.monotonic()
                try:
                    state = client.getRotorStates(vehicle_name=vehicle)
                    stamp = int(state.timestamp)
                    rotors = [[float(r[field]) for field in ('thrust', 'speed')] for r in state.rotors]
                    if len(rotors) != 4 or not all(math.isfinite(v) and v >= 0 for r in rotors for v in r):
                        raise ValueError('Invalid quadrotor sample')
                    if stamp > previous:
                        self.rows.append(dict(sim_timestamp_ns=stamp, rotors=rotors))
                        previous = stamp
                    elif stamp == previous:
                        self.duplicates += 1
                    else:
                        self.errors.append('Rotor timestamp precedes the tracking start or previous sample')
                except Exception as exc:
                    self.errors.append(str(exc))
                self.stop.wait(max(0., 1/self.rate_hz - (time.monotonic()-began)))
        except Exception as exc:
            self.errors.append(str(exc))
        finally:
            if client is not None:
                client.client.close()

    def save(self, directory, end_ns, repeated_states):
        self.stop.set()
        self.thread.join(timeout=2)
        rows = [r for r in list(self.rows) if r['sim_timestamp_ns'] <= end_ns]
        (directory/'rotors.json').write_text(json.dumps(dict(
            source='AirSim getRotorStates', target_hz=self.rate_hz, start_ns=self.start_ns, end_ns=end_ns,
            rotor_order='AirSim array order; M1 through M4', columns=['thrust_n', 'speed_rad_s'],
            samples=rows, repeated_rotor_timestamps=self.duplicates, repeated_tracking_reads=repeated_states,
            errors=list(self.errors), closed=not self.thread.is_alive()), separators=(',', ':'), allow_nan=False),
            encoding='utf-8')
