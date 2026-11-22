"""Install only this task's native mouse observer; preserve all other actors."""
import hashlib
import json
from pathlib import Path
import traceback
import unreal as ue

ROOT = Path(__file__).resolve().parents[2]
MAP = '/Game/DroneControlOverview/Maps/DroneControlOverview'
CLASS = '/Script/Blocks.DroneControlObserver'
TAG = 'DCO_ObserverOwned'
VERSION_TAG = 'DCO_MouseObserver_v1'
RECEIPT = ROOT / 'logs/dual_target_mouse_observer_install.json'
receipt = dict(status='INSTALLING', map=MAP, observer_class=CLASS, version=1,
               flight_commands=False)

try:
    kind = ue.load_class(None, CLASS)
    if not kind:
        raise RuntimeError('Compile the Blocks observer module before installing the map actor.')
    if not ue.EditorLevelLibrary.load_level(MAP):
        raise RuntimeError('Cannot load the isolated control map.')
    actors = ue.EditorLevelLibrary.get_all_level_actors()
    marker = next((a for a in actors if a.get_actor_label() == 'DC_OverviewMarker'), None)
    if marker is None or 'DCO_BuilderOwned' not in {str(t) for t in marker.tags}:
        raise RuntimeError('Refusing to alter a map without its control-task ownership marker.')
    observers = [a for a in actors if a.get_class() == kind]
    if len(observers) > 1:
        raise RuntimeError('The control map has multiple observer actors.')
    if observers:
        observer = observers[0]
        if TAG not in {str(t) for t in observer.tags}:
            raise RuntimeError('Refusing to modify an observer not owned by this installer.')
    else:
        observer = ue.EditorLevelLibrary.spawn_actor_from_class(kind, ue.Vector(0, 0, 0))
    observer.set_actor_label('DC_MouseObserver')
    observer.set_editor_property('tags', [ue.Name(TAG), ue.Name(VERSION_TAG)])
    observer.set_folder_path(ue.Name('DroneControlOverview'))
    if not ue.EditorLevelLibrary.save_current_level():
        raise RuntimeError('Unable to save the native observer actor.')
    map_file = ROOT / 'external/AirSim/Unreal/Environments/Blocks/Content/DroneControlOverview/Maps/DroneControlOverview.umap'
    receipt.update(status='PASS', actor=observer.get_name(),
                   map_sha256=hashlib.sha256(map_file.read_bytes()).hexdigest())
except Exception:
    receipt.update(status='FAIL', error=traceback.format_exc())
finally:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    ue.log(str(receipt))
