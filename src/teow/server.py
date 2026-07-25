"""浏览器观战服务(v1.2):FastAPI + WebSocket 回放流。

照 alicization server/ 的组织,但 TEOW 状态量小(N=128 实体),用 JSON over WS
即可,不必二进制协议/WebGL。当前只做**回放模式**(读 run 目录 trajectory.npz);
live 对战流留待后续版本。

用法:.venv/bin/python src/run.py serve <run_dir> [--port 8000]
浏览器打开 http://127.0.0.1:8000/
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import numpy as np

# fastapi 的类型必须在模块级导入:本文件开了 future annotations(注解成字符串),
# FastAPI 用 get_type_hints 从模块 globals 解析 WebSocket——函数内局部导入会
# 解析失败,websocket 路由被当成依赖错误直接 403(实测坑)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

WEB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "web"


def load_replay(run_dir: pathlib.Path) -> dict:
    """把 run 目录读成可 JSON 化的 meta + 帧列表。"""
    from .config import Config
    from .map import build_map

    cfg = Config(**json.loads((run_dir / "resolved_config.json").read_text()))
    m = build_map(cfg)
    raw = np.load(run_dir / "trajectory.npz")
    # 必须一次性物化:npz 是惰性解压,循环里每次 data[k] 都会重新解压整个数组,
    # 帧循环直接变平方级(实测 1359 帧卡 148s → 物化后 <2s)
    data = {k: raw[k] for k in raw.files}
    n_frames = int(data["tick"].shape[0])

    meta = {
        "kind": "meta",
        "grid": [cfg.grid_h, cfg.grid_w],
        "e_max": cfg.e_max,
        "record_every": int(data["record_every"]),
        "node_pos": m.node_pos.tolist(),
        "node_type": m.node_type.tolist(),
        "hq_pos": m.hq_pos.tolist(),
        "n_frames": n_frames,
        "run": run_dir.name,
    }

    has_flags = "flag_active" in data  # v1.3 起才录;旧 run 目录发空表
    frames = []
    for i in range(n_frames):
        alive = data["alive"][i]
        idx = np.flatnonzero(alive)
        frames.append({
            "kind": "frame",
            "i": i,
            "tick": int(data["tick"][i]),
            "res": data["resources"][i].tolist(),
            "upgrades": data["upgrades"][i].tolist(),
            "node_owner": data["node_owner"][i].tolist(),
            "winner": int(data["winner"][i]),
            "flags": [
                [int(p), round(float(data["flag_pos"][i][p][j][0]), 2),
                 round(float(data["flag_pos"][i][p][j][1]), 2)]
                for p in range(2)
                for j in range(data["flag_active"].shape[2])
                if data["flag_active"][i][p][j]
            ] if has_flags else [],
            "ents": [
                {
                    "s": int(j),
                    "t": int(data["etype"][i][j]),
                    "p": [round(float(data["pos"][i][j][0]), 2),
                          round(float(data["pos"][i][j][1]), 2)],
                    "hp": int(data["hp"][i][j]),
                    "lv": int(data["level"][i][j]),
                    "in": bool(data["inside"][i][j]),
                    "cg": int(data["cargo"][i][j]),
                    "bt": int(data["btype"][i][j]),
                }
                for j in idx
            ],
        })
    return {"meta": meta, "frames": frames}


def create_app(run_dir: pathlib.Path, fps: int = 20):
    replay = load_replay(run_dir)
    app = FastAPI()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            await sock.send_text(json.dumps(replay["meta"]))
            for fr in replay["frames"]:
                await sock.send_text(json.dumps(fr))
                await asyncio.sleep(1.0 / fps)
            await sock.send_text(json.dumps({"kind": "end"}))
        except WebSocketDisconnect:
            pass

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


def serve(run_dir: pathlib.Path, host: str = "127.0.0.1",
          port: int = 8000, fps: int = 20) -> None:
    import uvicorn

    print(f"观战: http://{host}:{port}/   (run={run_dir.name}, fps={fps})")
    uvicorn.run(create_app(run_dir, fps), host=host, port=port, log_level="warning")
