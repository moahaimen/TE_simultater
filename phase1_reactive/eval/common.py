"""Common config helpers for Phase-1 reactive runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from phase1_reactive.data.topology_loader import Phase1ConfigBundle, Phase1TopologySpec, get_topology_specs, load_phase1_config
from phase1_reactive.data.traffic_loader import load_reactive_dataset
from phase1_reactive.drl.dqn_selector import DQNConfig
from phase1_reactive.drl.reward import ReactiveRewardConfig
from phase1_reactive.env.offline_env import ReactiveEnvConfig
from phase1_reactive.routing.path_cache import build_dataset_paths
from phase3.ppo_agent import PPOConfig
from phase3.state_builder import TelemetryConfig


DRL_ALIAS = "our_drl"
PPO_METHOD = "our_drl_ppo"
DQN_METHOD = "our_drl_dqn"
PPO_PRETRAIN_METHOD = "our_drl_ppo_pretrained"
DQN_PRETRAIN_METHOD = "our_drl_dqn_pretrained"
DUAL_GATE_METHOD = "our_drl_dual_gate"


def load_bundle(config_path: str | Path) -> Phase1ConfigBundle:
    return load_phase1_config(config_path)


def max_steps_from_args(bundle: Phase1ConfigBundle, override: int | None) -> int | None:
    if override is not None:
        return int(override)
    exp = bundle.raw.get("experiment", {})
    return int(exp.get("max_steps")) if isinstance(exp, dict) and exp.get("max_steps") is not None else None


def build_reactive_env_cfg(bundle: Phase1ConfigBundle) -> ReactiveEnvConfig:
    exp = bundle.raw.get("experiment", {}) if isinstance(bundle.raw.get("experiment"), dict) else {}
    reward_cfg = bundle.raw.get("reward", {}) if isinstance(bundle.raw.get("reward"), dict) else {}
    telemetry_cfg = bundle.raw.get("telemetry", {}) if isinstance(bundle.raw.get("telemetry"), dict) else {}
    return ReactiveEnvConfig(
        k_crit=int(exp.get("k_crit", 20)),
        lp_time_limit_sec=int(exp.get("lp_time_limit_sec", 20)),
        telemetry=TelemetryConfig(**telemetry_cfg),
        reward=ReactiveRewardConfig(**reward_cfg),
    )


def build_ppo_cfg(bundle: Phase1ConfigBundle) -> PPOConfig:
    drl = bundle.raw.get("drl", {}) if isinstance(bundle.raw.get("drl"), dict) else {}
    return PPOConfig(**drl)


def build_dqn_cfg(bundle: Phase1ConfigBundle) -> DQNConfig:
    dqn = bundle.raw.get("dqn", {}) if isinstance(bundle.raw.get("dqn"), dict) else {}
    return DQNConfig(**dqn)


def load_named_dataset(bundle: Phase1ConfigBundle, spec: Phase1TopologySpec, max_steps: int | None):
    dataset = load_reactive_dataset(spec, bundle, max_steps=max_steps)
    path_library = build_dataset_paths(dataset, k_paths=int(bundle.raw.get("experiment", {}).get("k_paths", 3)))
    return dataset, path_library


def write_config_snapshot(bundle: Phase1ConfigBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle.raw, indent=2) + "\n", encoding="utf-8")


def collect_specs(bundle: Phase1ConfigBundle, field_name: str) -> list[Phase1TopologySpec]:
    return get_topology_specs(bundle, field_name)


def normalize_method_list(methods: Sequence[str], drl_method: str | None) -> list[str]:
    out: list[str] = []
    mode = str(drl_method or "ppo").lower()
    for method in methods:
        key = str(method)
        expanded = [key]
        if key == DRL_ALIAS:
            if mode == "both":
                expanded = [PPO_METHOD, DQN_METHOD]
            elif mode == "dqn":
                expanded = [DQN_METHOD]
            else:
                expanded = [PPO_METHOD]
        for item in expanded:
            if item not in out:
                out.append(item)
    return out


def checkpoint_map_from_train_dir(train_dir: Path | str) -> dict[str, Path]:
    base = Path(train_dir)
    mapping: dict[str, Path] = {}

    ppo_ckpt = base / "ppo" / "policy.pt"
    if not ppo_ckpt.exists():
        shared_ckpt = base / "shared" / "policy.pt"
        if shared_ckpt.exists():
            ppo_ckpt = shared_ckpt
    if ppo_ckpt.exists():
        mapping[PPO_METHOD] = ppo_ckpt
        mapping[DRL_ALIAS] = ppo_ckpt

    dqn_ckpt = base / "dqn" / "qnet.pt"
    if dqn_ckpt.exists():
        mapping[DQN_METHOD] = dqn_ckpt

    ppo_pre = base / "ppo_pretrained" / "policy.pt"
    if ppo_pre.exists():
        mapping[PPO_PRETRAIN_METHOD] = ppo_pre

    dqn_pre = base / "dqn_pretrained" / "qnet.pt"
    if dqn_pre.exists():
        mapping[DQN_PRETRAIN_METHOD] = dqn_pre

    return mapping
