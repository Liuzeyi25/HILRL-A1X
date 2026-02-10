#!/usr/bin/env python3
"""External CuRobo IK service for A1_X.

Run this script inside the conda environment that has torch + curobo installed.
It exposes a ZMQ REP endpoint for IK queries from the ROS2 node.
"""

import argparse
import os
import sys
import time
from typing import Any

import numpy as np

import torch
import zmq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from a1_x_kenimetic_haoyuan import A1Kinematics


def _build_response(status: str, **kwargs: Any) -> dict:
    payload = {"status": status}
    payload.update(kwargs)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="CuRobo IK Service")
    parser.add_argument(
        "--bind",
        type=str,
        default="tcp://*:6202",
        help="Bind address for ZMQ REP server",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf",
        help="URDF path for IK solver",
    )
    parser.add_argument(
        "--pos-threshold",
        type=float,
        default=None,
        help="Override IK position threshold (meters)",
    )
    parser.add_argument(
        "--rot-threshold",
        type=float,
        default=None,
        help="Override IK rotation threshold (radians)",
    )
    args = parser.parse_args()

    print(
        "[CuRobo IK Service] torch cuda:",
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )

    ik_solver = A1Kinematics(
        urdf_file=args.urdf,
        base_link="base_link",
        ee_link="arm_link6",
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)

    print(f"[CuRobo IK Service] Listening on {args.bind}")

    try:
        while True:
            request = socket.recv_json()
            cmd = request.get("cmd", "solve_ik")

            if cmd == "shutdown":
                socket.send_json(_build_response("ok", message="shutting down"))
                break

            if cmd != "solve_ik":
                socket.send_json(_build_response("error", error="unknown cmd"))
                continue

            target_pos = request.get("target_pos")
            target_quat = request.get("target_quat")
            current_joints = request.get("current_joints")

            if target_pos is None or target_quat is None or current_joints is None:
                socket.send_json(
                    _build_response("error", error="missing target or joints")
                )
                continue

            try:
                # 调用 A1Kinematics.solve_ik，传入当前关节以优化求解
                ik_solver.prev_q = torch.as_tensor(
                    current_joints[:6], **ik_solver.tensor_args.as_torch_dict()
                ).unsqueeze(0)
                result = ik_solver.solve_ik(
                    pos=np.array(target_pos, dtype=float),
                    quat=np.array(target_quat, dtype=float)
                )
            except Exception as exc:
                socket.send_json(_build_response("error", error=str(exc)))
                continue

            # A1Kinematics 返回结果的访问方式：
            # - result.success: bool tensor
            # - result.js_solution.position: 最佳解（已由solve_ik设置）
            # - result.solution: 所有解的tensor
            if result.success.cpu().numpy().any():
                # 获取最佳解（已经存储在 js_solution.position）
                solution = result.js_solution.position.cpu().numpy()[:6].tolist()
                
                # 计算误差（需要从所有解中获取）
                pos_error = None
                rot_error = None
                try:
                    if hasattr(result, "position_error"):
                        pos_error = float(result.position_error.cpu().numpy().min())
                except Exception:
                    pass
                try:
                    if hasattr(result, "rotation_error"):
                        rot_error = float(result.rotation_error.cpu().numpy().min())
                except Exception:
                    pass
                
                socket.send_json(
                    _build_response(
                        "ok",
                        solution=solution,
                        pos_error=pos_error if pos_error is not None else 0.0,
                        rot_error=rot_error if rot_error is not None else 0.0,
                        timestamp=time.time(),
                    )
                )
            else:
                pos_error = None
                rot_error = None
                try:
                    if hasattr(result, "position_error"):
                        pos_error = float(result.position_error.cpu().numpy().min())
                except Exception:
                    pass
                try:
                    if hasattr(result, "rotation_error"):
                        rot_error = float(result.rotation_error.cpu().numpy().min())
                except Exception:
                    pass
                socket.send_json(
                    _build_response(
                        "error",
                        error="ik_failed",
                        target_pos=target_pos,
                        target_quat=target_quat,
                        current_joints=current_joints,
                        pos_error=pos_error,
                        rot_error=rot_error,
                    )
                )

    finally:
        try:
            socket.close()
        except Exception:
            pass
        context.term()


if __name__ == "__main__":
    main()
