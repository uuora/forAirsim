"""Lightweight desktop adapter for one persistent, frozen v24 trial.

Importing this module never imports Qt, NumPy, a detector or a flight owner.
Call bootstrap/start_trial/refresh_scene on a GUI worker thread. Polling only
reads published records. Closing the UI requests a safe return and leaves its
independent trial owner alive until the driver verifies ground release.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/fixed_map_perception_yolo_gpu640_v24.json"
CONFIG_SHA256 = "260fcf3147c550c46b34f7e7a649532c1454d4ca2a8da77ebec8d2aa0a397789"
DETECTOR_CONFIG_SHA256 = "261dd8cb07228a85c7cfd6217e9b24dc8ad9b04059e04aa7c0206f461e176d6d"
WEIGHTS_SHA256 = "92b0dc0f4e17ff14e9c1e110b9cad9ef249274136ee9a28d99b7885dfb396844"
TRAINING_RECEIPT_SHA256 = "f98db6fb0f916bd00ea74ee36596fb222d662f3a913d36610d9e96aa7c4b22a5"
OWNER_CLAIM = ROOT / "reports/interface_v24_owner.json"
_RUNTIME_IMPORT_LOCK = threading.Lock()
_LOCAL_OWNER_LOCK = threading.RLock()


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _write(path, value):
    path = _workspace_path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _workspace_path(path):
    path = Path(path).resolve()
    if ROOT.drive.upper() != "D:" or not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("V24_OUTPUT_REQUIRES_D_WORKSPACE")
    return path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(pid):
    import psutil
    process = psutil.Process(pid)
    return dict(pid=pid, created=process.create_time())


def _alive(identity):
    import psutil
    try:
        return _identity(identity["pid"]) == identity
    except (KeyError, TypeError, psutil.NoSuchProcess):
        return False
    # AccessDenied is deliberately not treated as a dead owner.


@contextmanager
def _owner_lock():
    """Serialize stale-claim cleanup and replacement across desktop windows."""
    with _LOCAL_OWNER_LOCK:
        with (ROOT / "reports/interface_v24_owner.lock").open("a+b") as handle:
            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class V24Session:
    def __init__(self, output=None):
        selected = output or ROOT / "reports" / ("interface_v24_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self.output = _workspace_path(selected)
        if not self.output.is_relative_to(ROOT / "reports"):
            raise ValueError("V24_SESSION_REQUIRES_REPORTS_DIRECTORY")
        self.output.mkdir(parents=True, exist_ok=False)
        self.config_path = CONFIG_PATH
        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._bootstrap = dict(status="NOT_STARTED", phase="NOT_STARTED", scene_bound=False,
                               ground={}, engine={}, error=None)
        self._engine = None
        self._scene_version = None
        self._process = None
        self._log = None
        self._trial_root = None
        self._trial_output = None
        self._preparing = False
        self._stop_requested = False
        self._closed = False
        self._error = None
        self._claim_token = None
        self._trial_number = 0
        self._last_records = {}
        self._previous_cleanup_required = False
        _write(self.output / "SESSION.json", dict(utc=_utc(), schema="v24_desktop_persistent_owner",
            config=str(self.config_path), config_sha256=CONFIG_SHA256, output=str(self.output),
            flight_started_on_window_open=False, storage_location="D_WORKSPACE_ONLY"))

    def _verify_frozen(self):
        if _sha(self.config_path) != CONFIG_SHA256:
            raise ValueError("V24_FROZEN_CONFIG_HASH_MISMATCH")
        config = _read(self.config_path)
        detector_path = _workspace_path(ROOT / config["detector_config"])
        if _sha(detector_path) != DETECTOR_CONFIG_SHA256:
            raise ValueError("V24_FROZEN_DETECTOR_CONFIG_HASH_MISMATCH")
        detector = _read(detector_path)
        if detector.get("weights_sha256") != WEIGHTS_SHA256:
            raise ValueError("V24_FROZEN_WEIGHTS_DECLARATION_MISMATCH")
        weights = _workspace_path(detector["weights"])
        if _sha(weights) != WEIGHTS_SHA256:
            raise ValueError("V24_FROZEN_WEIGHTS_HASH_MISMATCH")
        receipt = _workspace_path(detector["training_receipt"])
        if detector.get("training_receipt_sha256") != TRAINING_RECEIPT_SHA256 or _sha(receipt) != TRAINING_RECEIPT_SHA256:
            raise ValueError("V24_FROZEN_TRAINING_RECEIPT_HASH_MISMATCH")
        return config

    def _environment(self):
        base = _workspace_path(ROOT / ".runtime/v24_desktop")
        environment = dict(os.environ)
        paths = {"TEMP": base / "temp", "TMP": base / "temp", "TMPDIR": base / "temp",
                 "YOLO_CONFIG_DIR": base / "yolo", "XDG_CACHE_HOME": base / "cache",
                 "HF_HOME": base / "huggingface", "HUGGINGFACE_HUB_CACHE": base / "huggingface/hub",
                 "TRANSFORMERS_CACHE": base / "huggingface/transformers", "TORCH_HOME": base / "torch",
                 "MPLCONFIGDIR": base / "matplotlib", "UE-LocalDataCachePath": ROOT / ".unreal-cache"}
        for name, path in paths.items():
            path = _workspace_path(path)
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
        environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        return environment

    @staticmethod
    def _python():
        executable = Path(sys.executable)
        if executable.name.lower() == "pythonw.exe":
            executable = executable.with_name("python.exe")
        return str(executable)

    def _operation(self):
        if self._closed:
            raise RuntimeError("V24_SESSION_CLOSED")
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("V24_SESSION_OPERATION_IN_PROGRESS")
        with self._state_lock:
            if self._process is not None and self._process.poll() is None:
                self._operation_lock.release()
                raise RuntimeError("V24_TRIAL_OWNER_ALREADY_RUNNING")
            if self._process is not None:
                final = _read(self._trial_output / "result.json") if self._trial_output else {}
                if self._cleanup_passed(final):
                    self._release_claim()
                if self._log is not None:
                    self._log.close()
                    self._log = None
                self._process = None

    def _set_bootstrap(self, **values):
        with self._state_lock:
            self._bootstrap.update(values)
            record = dict(self._bootstrap, utc=_utc())
        _write(self.output / "BOOTSTRAP_STATUS.json", record)
        return record

    def _ground_and_bind(self, config):
        from fixed_map_trial_cleanup import probe_ground
        from fixed_map_scene_binding import bind_scene_config, current_engine_identity
        ground = probe_ground()
        self._set_bootstrap(ground=ground)
        if not ground.get("passed"):
            raise ValueError("V24_REQUIRES_FRESH_LANDED_RELEASED_GROUND")
        engine = current_engine_identity()
        info = _read(self.output / "bootstrap/active_engine.json")
        info.update(engine)
        self._engine = engine
        self._set_bootstrap(engine=info)
        try:
            bound = bind_scene_config(config, _read(ROOT / "configs/fixed_map_scene_state.json"))
        except ValueError as error:
            self._scene_version = None
            return self._set_bootstrap(status="READY_SCENE_SYNC_REQUIRED", phase="READY_SCENE_SYNC_REQUIRED",
                scene_bound=False, error=str(error), message="UE 已连接并确认地面安全；请点击同步地面场景后再开始任务。")
        self._scene_version = bound["runtime_scene_version"]
        _write(self.output / "bootstrap/SCENE_BINDING.json", dict(utc=_utc(), engine_identity=engine,
            scene_version=self._scene_version, config_sha256=CONFIG_SHA256, scene_bound=True))
        return self._set_bootstrap(status="READY", phase="READY", scene_bound=True, error=None,
            scene_version=self._scene_version, message="v24 已就绪，模型和飞行将在点击开始后启动。")

    def bootstrap(self):
        self._operation()
        try:
            self._error = None
            self._set_bootstrap(status="STARTING", phase="VERIFYING_FROZEN_INPUTS", error=None, scene_bound=False)
            config = self._verify_frozen()
            with _RUNTIME_IMPORT_LOCK:
                os.environ.update(self._environment())
                from fixed_map_cpu_runtime import configure_math_runtime
                configure_math_runtime(config)
                from fixed_map_runtime import configure_session
                configure_session(self.output / "bootstrap")
                import fixed_map_mission as mission
                self._set_bootstrap(phase="STARTING_UE")
                mission.ensure_scene()
                self._set_bootstrap(phase="CHECKING_GROUND_AND_SCENE")
                return self._ground_and_bind(config)
        except Exception as error:
            self._error = str(error)
            return self._set_bootstrap(status="FAILED", phase="FAILED", error=str(error), scene_bound=False)
        finally:
            self._operation_lock.release()

    def _command(self, arguments, log_path):
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run([self._python(), *arguments], cwd=ROOT, env=self._environment(),
                stdout=handle, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError("V24_PREPARATION_FAILED_" + str(result.returncode) + "; see " + str(log_path))

    def refresh_scene(self, randomize=False, seed=None):
        self._operation()
        try:
            self._error = None
            if not self._engine or self._bootstrap["status"] not in {"READY", "READY_SCENE_SYNC_REQUIRED"}:
                raise RuntimeError("V24_BOOTSTRAP_REQUIRED_BEFORE_SCENE_SYNC")
            config = self._verify_frozen()
            from fixed_map_scene_binding import current_engine_identity
            from fixed_map_trial_cleanup import probe_ground
            if current_engine_identity() != self._engine:
                raise ValueError("V24_ENGINE_CHANGED_BOOTSTRAP_AGAIN")
            if not probe_ground().get("passed"):
                raise ValueError("V24_SCENE_SYNC_REQUIRES_FRESH_RELEASED_GROUND")
            self._claim_owner()
            self._set_bootstrap(status="STARTING", phase="SYNCHRONIZING_GROUND_SCENE", scene_bound=False)
            folder = self.output / ("scene_sync_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            arguments=[str(ROOT / "python/fixed_map_random_region.py"), "--output", str(folder)]
            if randomize:
                if type(seed) is not int or not 0<=seed<=2147483647:raise ValueError('RANDOM_SEED_REQUIRED')
                arguments += ['--seed', str(seed)]
            else:arguments += ['--restore']
            self._command(arguments,
                          self.output / (folder.name + ".log"))
            return self._ground_and_bind(config)
        except Exception as error:
            self._error = str(error)
            return self._set_bootstrap(status="FAILED", phase="FAILED", error=str(error), scene_bound=False)
        finally:
            self._release_claim()
            self._operation_lock.release()

    def _assert_no_owner(self):
        import psutil
        # Existing experiment runners also own the control loop, even if
        # their model warmup has not requested takeoff yet.
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = process.info.get("cmdline") or []
                script_names = {Path(argument).name.lower() for argument in command}
                if process.pid != os.getpid():
                    if "run_yolo_language_mission_v8.py" in script_names:
                        raise RuntimeError("V24_EXISTING_TRIAL_OWNER_RUNNING_PID_" + str(process.pid))
                    if "fixed_map_perception_process.py" in script_names and "--worker" in command:
                        raise RuntimeError("V24_EXISTING_SENSOR_WORKER_RUNNING_PID_" + str(process.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not OWNER_CLAIM.exists():
            return
        claim = _read(OWNER_CLAIM)
        if not claim or not claim.get("identity"):
            raise RuntimeError("V24_OWNER_CLAIM_UNREADABLE_OR_PENDING")
        if _alive(claim["identity"]):
            raise RuntimeError("V24_ANOTHER_DESKTOP_OWNER_RUNNING_PID_" + str(claim["identity"]["pid"]))
        if claim.get("stage") == "TRIAL_OWNER":
            previous_output = _workspace_path(claim["output"])
            if not self._cleanup_passed(_read(previous_output / "result.json")):
                raise RuntimeError("V24_PREVIOUS_OWNER_FINAL_CLEANUP_REQUIRED")
        # A dead owner cannot resume. The locked claimant removes this stale
        # record; a read-only availability check must not remove another claim.

    def _claim_owner(self):
        with _owner_lock():
            self._assert_no_owner()
            OWNER_CLAIM.unlink(missing_ok=True)
            token = uuid.uuid4().hex
            claim = dict(utc=_utc(), token=token, identity=_identity(os.getpid()),
                         stage="PREPARING", session=str(self.output))
            try:
                descriptor = os.open(str(OWNER_CLAIM), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                raise RuntimeError("V24_ANOTHER_DESKTOP_START_IN_PROGRESS") from None
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(claim, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            self._claim_token = token

    def _release_claim(self):
        if self._claim_token:
            with _owner_lock():
                if _read(OWNER_CLAIM).get("token") == self._claim_token:
                    OWNER_CLAIM.unlink(missing_ok=True)
                self._claim_token = None

    def _check_start_cancelled(self, cancel_event=None):
        if self._closed or self._stop_requested or (cancel_event is not None and cancel_event.is_set()):
            raise RuntimeError("V24_START_CANCELLED_BEFORE_OWNER_LAUNCH")

    @staticmethod
    def _cleanup_passed(final):
        if final.get("terminal_phase") != "FINAL" or final.get("ground_verification_passed") is not True or final.get("cleanup_errors") != []:
            return False
        sensor = final.get("sensor_cleanup", {})
        return sensor.get("status") == "PASS" or (final.get("flight_start_requested") is False and not sensor)

    def start_trial(self, instruction, cancel_event=None):
        # The GUI sets the per-request Event before scheduling a worker. A
        # cancelled queued worker must not clear that request when it wakes.
        self._check_start_cancelled(cancel_event)
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1000:
            raise ValueError("请输入 1 至 1000 字任务指令")
        self._operation()
        try:
            if self._previous_cleanup_required:
                final = _read(self._trial_output / "result.json") if self._trial_output else {}
                if not self._cleanup_passed(final):
                    raise RuntimeError("V24_PREVIOUS_TRIAL_FINAL_CLEANUP_REQUIRED")
                self._previous_cleanup_required = False
            with self._state_lock:
                self._release_claim()
                if self._log is not None:
                    self._log.close()
                    self._log = None
                self._process = None
                self._preparing = True
                self._trial_root = None
                self._trial_output = None
                self._error = None
                self._last_records = {}
            if self._bootstrap["status"] != "READY" or not self._bootstrap["scene_bound"]:
                raise RuntimeError("V24_READY_SCENE_BINDING_REQUIRED")
            self._check_start_cancelled(cancel_event)
            self._verify_frozen()
            from fixed_map_scene_binding import current_engine_identity, bind_scene_config
            from fixed_map_trial_cleanup import probe_ground
            if current_engine_identity() != self._engine:
                self._set_bootstrap(status="READY_SCENE_SYNC_REQUIRED", phase="READY_SCENE_SYNC_REQUIRED", scene_bound=False,
                    error="V24_ENGINE_CHANGED_BOOTSTRAP_AGAIN")
                raise ValueError("UE 已重启，请重新连接并同步地面场景。")
            bound = bind_scene_config(_read(self.config_path), _read(ROOT / "configs/fixed_map_scene_state.json"))
            if bound["runtime_scene_version"] != self._scene_version:
                raise ValueError("V24_SCENE_EPOCH_CHANGED_BOOTSTRAP_AGAIN")
            if not probe_ground().get("passed"):
                raise ValueError("V24_START_REQUIRES_FRESH_LANDED_RELEASED_GROUND")
            self._claim_owner()
            self._trial_number += 1
            trial = self.output / ("trial_%02d" % self._trial_number)
            with self._state_lock:
                self._trial_root = trial
                self._trial_output = trial / "online_trial"
            self._check_start_cancelled(cancel_event)
            self._command([str(ROOT / "python/prepare_closed_loop_retest.py"), str(trial),
                "--config", str(self.config_path), "--previous", str(ROOT / "reports/v24_gpu_repeatability_20261012")],
                self.output / (trial.name + "_prepare.log"))
            self._check_start_cancelled(cancel_event)
            self._verify_frozen()
            ground = probe_ground()
            if not ground.get("passed") or current_engine_identity() != self._engine:
                raise ValueError("V24_GROUND_OR_ENGINE_CHANGED_DURING_PREPARATION")
            bound = bind_scene_config(_read(self.config_path), _read(ROOT / "configs/fixed_map_scene_state.json"))
            if bound["runtime_scene_version"] != self._scene_version:
                raise ValueError("V24_SCENE_EPOCH_CHANGED_DURING_PREPARATION")
            _write(trial / "INTERFACE_REQUEST.json", dict(utc=_utc(), instruction=instruction.strip(),
                config_sha256=CONFIG_SHA256, engine_identity=self._engine, scene_version=self._scene_version,
                frozen_config_bytes_unchanged=True, actor="manual_v24_desktop", maximum_new_takeoffs=1))
            with self._state_lock:
                self._check_start_cancelled(cancel_event)
                self._log = (trial / "owner.log").open("w", encoding="utf-8")
                arguments = [self._python(), str(ROOT / "python/run_yolo_language_mission_v8.py"),
                    "--config", str(self.config_path), "--output", str(self._trial_output),
                    "--instruction", instruction.strip(), "--execute-one-trial", "--stop-request", str(trial / "stop.request"),
                    "--interface-request", str(trial / "INTERFACE_REQUEST.json"),
                    "--interface-request-sha256", _sha(trial / "INTERFACE_REQUEST.json")]
                options = dict(cwd=ROOT, env=self._environment(), stdin=subprocess.DEVNULL,
                               stdout=self._log, stderr=subprocess.STDOUT, close_fds=True)
                if os.name == "nt":
                    startup = subprocess.STARTUPINFO()
                    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startup.wShowWindow = 0
                    options.update(startupinfo=startup, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
                else:
                    options["start_new_session"] = True
                self._check_start_cancelled(cancel_event)
                self._process = subprocess.Popen(arguments, **options)
                self._previous_cleanup_required = True
                # If the Event was set during Popen itself, its independent
                # owner sees this external request before loading/takeoff.
                if cancel_event is not None and cancel_event.is_set():
                    (trial / "stop.request").write_text("cancelled_during_owner_launch " + _utc(), encoding="utf-8")
                claim = dict(utc=_utc(), token=self._claim_token, identity=_identity(self._process.pid),
                    stage="TRIAL_OWNER", session=str(self.output), trial=str(trial), output=str(self._trial_output))
                with _owner_lock():
                    if _read(OWNER_CLAIM).get("token") != self._claim_token:
                        raise RuntimeError("V24_OWNER_CLAIM_CHANGED_AFTER_LAUNCH")
                    _write(OWNER_CLAIM, claim)
                _write(trial / "OWNER_PROCESS.json", dict(claim, command=arguments,
                    ui_close_terminates_owner=False, stop_file=str(trial / "stop.request")))
                self._preparing = False
            return self.poll()
        except Exception as error:
            with self._state_lock:
                self._error = str(error)
                self._preparing = False
                alive = self._process is not None and self._process.poll() is None
                if not alive:
                    self._stop_requested = False
            if not alive:
                self._release_claim()
                if self._log is not None:
                    self._log.close()
                    self._log = None
            else:
                # A metadata failure cannot abandon a newly launched owner.
                # Its persistent driver must perform the usual safe return.
                try:
                    self.request_stop()
                except OSError as stop_error:
                    self._error = str(error) + "; STOP_REQUEST_WRITE_FAILED: " + str(stop_error)
            _write(self.output / "LAST_START_ERROR.json", dict(utc=_utc(), error=str(error),
                owner_running=alive, trial=str(self._trial_root) if self._trial_root else None))
            raise
        finally:
            self._operation_lock.release()

    def request_stop(self):
        with self._state_lock:
            trial = self._trial_root
            process = self._process
            preparing = self._preparing
            active = preparing or (process is not None and process.poll() is None)
            if active:
                self._stop_requested = True
        if trial is not None and trial.is_dir() and (preparing or (process is not None and process.poll() is None)):
            (trial / "stop.request").write_text("manual_v24_desktop_safe_return " + _utc(), encoding="utf-8")
            return dict(requested=True, path=str(trial / "stop.request"), terminates_owner=False)
        return dict(requested=preparing, pending_prelaunch=preparing, terminates_owner=False)

    def poll(self):
        with self._state_lock:
            process = self._process
            output = self._trial_output
            bootstrap = dict(self._bootstrap)
            preparing = self._preparing
            error = self._error
            cached = dict(self._last_records)
        running = process is not None and process.poll() is None
        records = {}
        cached_names = []
        publication_current_read = False
        if output is not None:
            for key, relative in [("status", "trial_status.json"), ("recovery", "recovery_status.json"),
                ("publication", "online/publication.json"), ("final", "result.json"), ("failure", "failure.json")]:
                value = _read(output / relative)
                if value:
                    cached[key] = value
                    with self._state_lock:
                        if self._trial_output == output:
                            self._last_records[key] = value
                    if key == "publication":
                        publication_current_read = True
                elif key in cached:
                    cached_names.append(key)
                records[key] = cached.get(key, {})
        status = records.get("status", {})
        recovery = records.get("recovery", {})
        final = records.get("final", {})
        failure = records.get("failure", {})
        stage = recovery.get("stage") if running and recovery else status.get("stage")
        stage = stage or ("PREPARING" if preparing else "INITIALIZING" if running else "FINISHED" if process else bootstrap["phase"])
        if not running and process is not None:
            with self._state_lock:
                if self._process is process and not self._preparing:
                    self._stop_requested = False
                    self._previous_cleanup_required = not self._cleanup_passed(final)
                    if self._log is not None:
                        self._log.close()
                        self._log = None
                    if not self._previous_cleanup_required:
                        self._release_claim()
            if not final and not error:
                error = failure.get("reason") or "OWNER_EXITED_WITHOUT_FINAL_RECORD; see owner.log"
        latest = dict(records.get("publication", {}).get("latest", {}))
        captured = latest.get("capture", {}).get("received_monotonic_s")
        source_age = time.monotonic() - captured if isinstance(captured, (int, float)) else None
        latest_current = bool(publication_current_read and running and
            not records.get("publication", {}).get("closed") and
            source_age is not None and 0 <= source_age <= .5)
        if latest and not latest_current:
            latest.update(valid=False, display_current=False,
                          publication_cache_stale=not publication_current_read)
        elif latest:
            latest["display_current"] = True
        return dict(running=running or preparing, owner_running=running, preparing=preparing,
            pid=process.pid if process is not None else None,
            output=str(output or self.output), session_output=str(self.output),
            trial_output=str(output) if output else None, bootstrap=bootstrap,
            scene_bound=bootstrap.get("scene_bound", False), engine_info=bootstrap.get("engine", {}),
            stage=stage, mission=status.get("mission") or final.get("mission") or {},
            state=recovery.get("state") if running and recovery else status.get("state") or final.get("state") or {},
            latest=latest, final=final, records_cached=cached_names,
            publication_current_read=publication_current_read, latest_current=latest_current,
            latest_source_age_s=source_age,
            error=error or failure.get("reason") or final.get("issue"),
            exit_code=process.poll() if process is not None else None)

    def close(self):
        self._closed = True
        receipt = self.request_stop()
        with self._state_lock:
            running = self._process is not None and self._process.poll() is None
            if not running and not self._preparing and not self._operation_lock.locked():
                final = _read(self._trial_output / "result.json") if self._trial_output else {}
                if not self._previous_cleanup_required or self._cleanup_passed(final):
                    self._release_claim()
                if self._log is not None:
                    self._log.close()
                    self._log = None
        return dict(receipt, owner_retained=running)
