"""Generate a replayable three-round protocol fixture, without AirSim.

The inputs, clocks, scene inventory acknowledgements, ETA values and contact
events are synthetic. No simulator, camera, model API or controller is invoked.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from refreshed_target_round import RefreshedTargetRound, RoundPhase, RoundThresholds


ARTIFACT_KIND = "PURE_PROTOCOL_FIXTURE"
ETA_PAIRS = ((3.0, 4.0), (5.0, 2.0), (2.0, 3.0))
POLICIES = ("fixed_A", "alternating", "nearest_eta")
EXPECTED_EXECUTORS = {
    "fixed_A": ("DroneA", "DroneA", "DroneA"),
    "alternating": ("DroneA", "DroneB", "DroneA"),
    "nearest_eta": ("DroneA", "DroneB", "DroneA"),
}
UNVERIFIED = [
    "AirSim 中旧 actor 的删除、新 actor 的生成及任意时刻仅一个目标的场景实测",
    "两机实际取消动作、安全保持和就绪标记的可信获取",
    "相机帧、YOLO/VLX 结果与目标 generation 的在线绑定及实际延迟",
    "模拟器碰撞事件、目标身份与碰撞后位置刷新的实际链路",
    "基于观测的 ETA 估计误差、轨迹优化收益、多机避障和连续运行稳定性",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class FixtureDriver:
    """Only fabricates messages and checks the protocol's returned decisions."""

    def __init__(self, policy: str):
        self.policy = policy
        self.machine = RefreshedTargetRound(RoundThresholds())
        self.clock = 1000.0
        self.sequence = 0
        self.checks = []

    def check(self, name: str, passed: bool, details: Dict[str, Any]) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "details": details})
        if not passed:
            raise RuntimeError(f"{self.policy}: {name}: {details}")

    def send(self, event: str, expected_reason: str = "accepted", **fields):
        self.clock = round(self.clock + 0.01, 6)
        self.sequence += 1
        payload = dict(round_id=self.machine.round_id, target_generation=self.machine.target_generation,
                       now=self.clock)
        timestamps = {
            "prepare_round": ("prepared_at",),
            "acknowledge_hold": ("recorded_at",),
            "observe": ("captured_at", "estimate_at"),
            "assign": ("created_at",),
            "dispatch": ("issued_at",),
            "mark_contact": ("issued_at",),
            "confirm_hit": ("reported_at",),
            "acknowledge_cancel": ("recorded_at",),
        }
        for key in timestamps[event]:
            payload[key] = self.clock
        payload.update(fields)
        decision = self.machine.submit(event, payload)
        self.check(
            f"event_{self.sequence:03d}_{event}",
            decision.reason == expected_reason and decision.accepted == (expected_reason == "accepted"),
            {"event_log_index": len(self.machine.event_log) - 1, "expected_reason": expected_reason,
             "actual_reason": decision.reason, "accepted": decision.accepted},
        )
        if not decision.accepted:
            self.check(f"event_{self.sequence:03d}_rejection_emits_no_actions", not decision.actions,
                       {"actions": list(decision.actions)})
        return decision

    def prepare(self, expected_reason="accepted"):
        next_round = self.machine.round_id + 1
        return self.send("prepare_round", expected_reason,
                         round_id=next_round, target_generation=self.machine.target_generation + 1,
                         actor_name=self.machine.actor_for_round(next_round),
                         active_target_actors=[self.machine.actor_for_round(next_round)])

    def ack(self, drone, cancel=False):
        return self.send("acknowledge_cancel" if cancel else "acknowledge_hold",
                         drone=drone, request_token=self.machine.cancel_token if cancel else self.machine.hold_token,
                         old_actions_cancelled=True, safe_hold=True, ready=True)

    def observe(self, drone, eta, *, old_generation=False):
        fields = dict(drone=drone, actor_name=self.machine.actor_name,
                      message_id=f"synthetic:{self.policy}:r{self.machine.round_id}:{drone}:m{self.sequence + 1}",
                      source="synthetic_vlx" if old_generation else "synthetic_perception",
                      identity_confirmed=True, ready=True, eta_seconds=eta)
        if old_generation:
            fields["target_generation"] = self.machine.target_generation - 1
        return self.send("observe", "target_generation_mismatch" if old_generation else "accepted", **fields)

    def plan(self):
        return {key: self.machine.assignment[key] for key in ("executor", "assignment_id", "plan_id")}

    def hit(self, stamp, expected_reason="accepted"):
        return self.send("confirm_hit", expected_reason, **self.plan(), actor_name=self.machine.actor_name,
                         has_collided=True, collision_timestamp=stamp)

    def run(self) -> Dict[str, Any]:
        rounds = []
        for index, eta_pair in enumerate(ETA_PAIRS, 1):
            scene_decision = self.prepare()
            self.check(f"round_{index}_new_scene_requests_two_holds",
                       len(scene_decision.actions) == 2
                       and {action["drone"] for action in scene_decision.actions} == set(self.machine.DRONES)
                       and all(action["operation"] == "CANCEL_AND_HOLD" for action in scene_decision.actions),
                       {"actions": list(scene_decision.actions)})
            for drone in self.machine.DRONES:
                self.ack(drone)
            if index > 1:
                self.observe("DroneA", eta_pair[0], old_generation=True)
            for drone, eta in zip(self.machine.DRONES, eta_pair):
                self.observe(drone, eta)
            assignment = self.send("assign", policy=self.policy,
                                   assignment_id=f"synthetic:{self.policy}:assignment:{index}",
                                   plan_id=f"synthetic:{self.policy}:plan:{index}")
            expected_executor = EXPECTED_EXECUTORS[self.policy][index - 1]
            self.check(f"round_{index}_executor_selection",
                       self.machine.assignment["executor"] == expected_executor,
                       {"expected_executor": expected_executor,
                        "actual_executor": self.machine.assignment["executor"], "synthetic_eta_pair": list(eta_pair)})
            self.check(f"round_{index}_one_executor_one_observer",
                       {self.machine.assignment["executor"], self.machine.assignment["observer"]} == set(self.machine.DRONES)
                       and len(assignment.actions) == 1,
                       {"executor": self.machine.assignment["executor"], "observer": self.machine.assignment["observer"]})
            baseline = max(1000, self.machine.last_collision_timestamps[expected_executor])
            dispatch = self.send("dispatch", **self.plan(), collision_baseline_timestamp=baseline)
            self.check(f"round_{index}_only_executor_receives_plan",
                       len([action for action in dispatch.actions if action["operation"] == "DISPATCH_PLAN"]) == 1,
                       {"actions": list(dispatch.actions)})
            self.hit(baseline, "old_collision_timestamp")
            self.send("mark_contact", **self.plan())
            hit_decision = self.hit(baseline + 1)
            self.check(f"round_{index}_hit_requests_cancel_before_refresh",
                       self.machine.phase == RoundPhase.CANCEL_PENDING
                       and {action["operation"] for action in hit_decision.actions} == {"CANCEL_AND_HOLD"},
                       {"phase": self.machine.phase.value, "actions": list(hit_decision.actions)})
            self.hit(baseline + 2, "duplicate_hit")
            self.prepare("cancellation_barrier_not_complete")
            self.ack("DroneA", cancel=True)
            self.prepare("cancellation_barrier_not_complete")
            refresh = self.ack("DroneB", cancel=True)
            self.check(f"round_{index}_refresh_after_both_cancel_confirmations",
                       self.machine.phase == RoundPhase.REFRESH_PENDING and len(refresh.actions) == 1
                       and refresh.actions[0]["operation"] == "REQUEST_TARGET_REFRESH",
                       {"phase": self.machine.phase.value, "actions": list(refresh.actions)})
            rounds.append({
                "round_id": self.machine.round_id,
                "target_generation": self.machine.target_generation,
                "actor_name": self.machine.actor_name,
                "synthetic_input": True,
                "eta_seconds": dict(zip(self.machine.DRONES, eta_pair)),
                "executor": self.machine.assignment["executor"],
                "observer": self.machine.assignment["observer"],
                "accepted_synthetic_contact_timestamp": baseline + 1,
                "final_phase": self.machine.phase.value,
            })
        replayed = RefreshedTargetRound.replay(self.machine.event_log)
        self.check("full_event_log_replay", replayed.snapshot() == self.machine.snapshot(),
                   {"event_count_including_initialize": len(self.machine.event_log)})
        self.check("exactly_three_synthetic_hits", self.machine.accepted_hit_rounds == {1, 2, 3},
                   {"accepted_hit_rounds": sorted(self.machine.accepted_hit_rounds)})
        decisions = [event["decision"] for event in self.machine.event_log[1:]]
        reasons = {}
        for decision in decisions:
            if not decision["accepted"]:
                reasons[decision["reason"]] = reasons.get(decision["reason"], 0) + 1
        return {
            "policy": self.policy,
            "status": "PASS",
            "synthetic_input": True,
            "rounds": rounds,
            "event_count": len(decisions),
            "accepted_event_count": sum(decision["accepted"] for decision in decisions),
            "rejected_event_count": sum(not decision["accepted"] for decision in decisions),
            "rejection_reason_counts": reasons,
            "check_count": len(self.checks),
            "all_checks_passed": all(check["passed"] for check in self.checks),
            "replay_verified": True,
            "checks": self.checks,
        }


def render_summary(report: Dict[str, Any]) -> str:
    lines = [
        "# 连续刷新目标协议：纯脚本演示",
        "",
        f"**{ARTIFACT_KIND} · {report['status']} · synthetic_input=true**",
        "",
        "本工件使用同步合成时钟、合成 ETA、合成场景库存/保持确认和合成接触消息。",
        "它验证状态机准入、角色分配、取消屏障和事件重放；没有连接 AirSim，也没有执行飞行、接触或目标刷新。",
        "不能将下面的合成事件数量计入 AirSim 实验成功率，不能据此声称气球真实破裂或连续运行安全。",
        "",
        "## 输入与结果",
        "",
        "三个轮次的 DroneA/DroneB 合成 ETA 分别为 (3, 4)、(5, 2)、(2, 3) 秒。阈值固定，未输入目标或无人机坐标。",
        "",
        "| 分配策略 | 第 1/2/3 轮执行机 | 合成接触事件 | 拒绝事件 | 成功检查 | 重放 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for policy in report["policies"]:
        executors = " / ".join(row["executor"] for row in policy.get("rounds", []))
        lines.append(f"| {policy['policy']} | {executors} | {len(policy.get('rounds', []))} | "
                     f"{policy.get('rejected_event_count', 0)} | {policy.get('check_count', 0)} | "
                     f"{'PASS' if policy.get('replay_verified') else 'NOT_RUN'} |")
    lines.extend([
        "",
        "每轮新场景先确认两机已取消旧动作并安全保持，再收取当前 generation 的新鲜观测。",
        "有效合成接触后两机再次取消并保持；两机确认完整后才产生刷新请求。每轮仅一个执行机、一个观察机。",
        "",
        "插入并核验的拒绝事件：旧 generation 的 VLX 消息、旧碰撞时间戳、重复接触，以及零个/仅一个取消确认时提前准备新目标。",
        "",
        "## 尚未实测的环节",
        "",
    ])
    lines.extend(f"- {item}。" for item in UNVERIFIED)
    lines.extend(["", "## 工件与溯源", "", "report.json 保存输入、源 SHA-256、全部检查与每策略事件文件哈希。",
                  "三个 *_events.json 保存完整状态机事件流，可用 RefreshedTargetRound.replay() 重放。",
                  "代码只产生操作请求，实际执行须由后续 AirSim 适配层完成。", "",
                  "AI使用说明：协议代码、合成事件脚本与报告由AI辅助编写；验证来自本机纯逻辑运行，不是AirSim实验。", ""])
    if report.get("errors"):
        lines.extend(["## 失败信息", "", *[f"- {error}" for error in report["errors"]], ""])
    return "\n".join(lines)


def main(output_dir: Path = None) -> int:
    root = Path(__file__).resolve().parent.parent
    if output_dir is None:
        output_dir = root / "logs" / f"refreshed_target_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    source_files = (Path(__file__).resolve(), Path(__file__).with_name("refreshed_target_round.py").resolve())
    report = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "ai_assistance": "protocol code, synthetic fixture and report authored with AI assistance",
        "synthetic_input": True,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "airsim_connected": False,
        "flight_commands_executed": False,
        "target_refresh_executed": False,
        "sources": [{"path": str(path), "sha256": sha256_file(path)} for path in source_files],
        "inputs": {
            "eta_pairs_seconds": [list(pair) for pair in ETA_PAIRS],
            "clock": {"kind": "synthetic_monotonic_seconds", "start": 1000.0, "event_step": 0.01},
            "thresholds": asdict(RoundThresholds()),
            "scene_inventory": "synthetic single actor name per round, without positions",
            "hold_and_cancel_attestations": "synthetic true flags, without controller execution",
            "collision_events": "synthetic integer timestamps, without AirSim contact",
        },
        "policies": [],
        "unverified": UNVERIFIED,
        "errors": [],
    }
    for policy in POLICIES:
        driver = FixtureDriver(policy)
        try:
            result = driver.run()
        except Exception as exc:
            report["status"] = "FAIL"
            report["errors"].append(f"{policy}: {type(exc).__name__}: {exc}")
            result = {"policy": policy, "status": "FAIL", "synthetic_input": True,
                      "all_checks_passed": False, "checks": driver.checks,
                      "check_count": len(driver.checks), "replay_verified": False}
        event_path = output_dir / f"{policy}_events.json"
        write_json(event_path, {
            "schema_version": 1,
            "artifact_kind": ARTIFACT_KIND,
            "synthetic_input": True,
            "policy": policy,
            "events": driver.machine.event_log,
        })
        result["event_file"] = str(event_path)
        result["event_file_sha256"] = sha256_file(event_path)
        report["policies"].append(result)
    report["all_checks_passed"] = all(policy["all_checks_passed"] for policy in report["policies"])
    report["replay_verified"] = all(policy["replay_verified"] for policy in report["policies"])
    report_path = output_dir / "report.json"
    write_json(report_path, report)
    (output_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
    print(f"{ARTIFACT_KIND}: {report['status']}")
    print(f"Report: {report_path}")
    print(f"Summary: {output_dir / 'summary.md'}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing directories are refused")
    args = parser.parse_args()
    raise SystemExit(main(args.output_dir))
