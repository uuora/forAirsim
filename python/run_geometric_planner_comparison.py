"""Prepare three references or explicitly fly them; default never calls AirSim."""
import argparse
import json
from pathlib import Path

from geometric_validation.campaign import PLANNER_CONFIG, TASK_CONFIG, prepare, run_campaign
from geometric_validation.planners import METHODS
from geometric_validation.runtime import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--environment", help="Task preset ID from --list-environments")
    p.add_argument("--list-environments", action="store_true")
    p.add_argument("--prepare-environments", action="store_true", help="Offline generation of every preset; no flight")
    p.add_argument("--task", type=Path, default=TASK_CONFIG)
    p.add_argument("--config", type=Path, default=PLANNER_CONFIG)
    p.add_argument("--airsim", action="store_true")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.launch and not a.airsim:
        p.error("--launch requires --airsim")
    if a.list_environments or a.prepare_environments or a.environment:
        from geometric_validation.environments import load_catalog, config_for, prepare_environment, prepare_all, load_environment_config
        if a.prepare_environments and a.airsim:
            p.error('--prepare-environments is offline only; select one environment explicitly for flight')
        if a.list_environments:
            print(json.dumps(load_catalog(),ensure_ascii=False,indent=2))
            return 0
        if a.prepare_environments:
            print(json.dumps(dict(configs=[str(x) for x in prepare_all()],rpc=False,flight_started=False),indent=2))
            return 0
        a.config=prepare_environment(a.environment) if a.prepare else config_for(a.environment)
        load_environment_config(a.environment)
    elif a.prepare:
        prepare(a.task, a.config)
    cfg = load_config(a.config)
    if not a.airsim:
        print(json.dumps(dict(check="PASS", rpc=False,environment=cfg.get('comparison',{}).get('environment',{'id':'baseline'}),
                              methods=[v["method"] for v in cfg["cases"].values()]),ensure_ascii=False,indent=2))
        return 0
    folder, rows = run_campaign(a.config, a.output, [a.method] if a.method else METHODS, a.launch)
    print(json.dumps(dict(directory=str(folder), results=rows), ensure_ascii=False, indent=2))
    return 0 if all(r["accepted"] is True for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
