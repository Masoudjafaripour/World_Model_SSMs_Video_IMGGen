"""LIBERO simulator check: build one task per suite, reset to a benchmark init
state, step a few no-op actions, and save the agent-view frame headless (EGL).

The saved PNGs are compared by eye with the training frames: the training data
has images rotated 180 degrees, and eval must feed the policy the same view.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

out = Path(os.environ["WM_RUNS"]) / "sim_test"
out.mkdir(parents=True, exist_ok=True)

suites = benchmark.get_benchmark_dict()
for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    suite = suites[name]()
    task = suite.get_task(0)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                                    task.bddl_file),
        camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(10):
        obs, reward, done, info = env.step(np.zeros(7))
    img = obs["agentview_image"]
    Image.fromarray(img).save(out / f"{name}_raw.png")
    Image.fromarray(img[::-1, ::-1]).save(out / f"{name}_rot180.png")
    print(f"{name}: {suite.get_num_tasks()} tasks, '{task.language}', frame {img.shape}, "
          f"state keys {sorted(k for k in obs if 'image' not in k)[:6]}...")
    env.close()
print(f"SIM TEST PASSED -> {out}")
