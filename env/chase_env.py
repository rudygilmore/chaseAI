"""Chase: N×N grid world with a green agent and a blue 3×3 X target (Gymnasium)."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Fixed framebuffer size (each logical cell is an integer number of pixels).
CANVAS_SIZE: int = 512

# RGB
_COLOR_BG = np.array([32, 32, 32], dtype=np.uint8)
_COLOR_AGENT = np.array([0, 255, 0], dtype=np.uint8)
_COLOR_X = np.array([30, 144, 255], dtype=np.uint8)

# Local (row, col) offsets within the 3×3 target patch for the five X cells.
_X_LOCAL_CELLS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 2),
    (1, 1),
    (2, 0),
    (2, 2),
)

# Actions: 0 = up, 1 = down, 2 = left, 3 = right (see README / execplan.md).
_ACTION_DELTA: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

# Reward shaping: small bonus when Manhattan distance to the goal center decreases.
_REWARD_GOAL = 1.0
_REWARD_STEP = -0.01
_REWARD_MOVE_TOWARD = 0.02


def _manhattan(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r1 - r2) + abs(c1 - c2)


def _validate_grid_side(n: int) -> None:
    """Require N = 2^k with integer k >= 3 (so N >= 8) and N | 512 for pixel mapping."""
    if n < 8:
        raise ValueError("n must be at least 8 (grid side N = 2^k with k >= 3).")
    if n & (n - 1) != 0:
        raise ValueError(f"n must be a power of 2 (got n={n}).")
    if CANVAS_SIZE % n != 0:
        raise ValueError(
            f"n must divide {CANVAS_SIZE} evenly (got n={n}); "
            "each logical cell must map to an integer pixel square."
        )


class ChaseEnv(gym.Env):
    """Discrete grid chase environment with 512×512 RGB observations."""

    metadata = {
        "render_modes": ("human", "rgb_array"),
        "render_fps": 30,
    }

    def __init__(
        self,
        n: int = 64,
        render_mode: str | None = None,
        max_episode_steps: int | None = 500,
    ) -> None:
        super().__init__()
        _validate_grid_side(n)

        self.n = n
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(CANVAS_SIZE, CANVAS_SIZE, 3),
            dtype=np.uint8,
        )

        self.cell_px = CANVAS_SIZE // n
        self._agent_row: int = 0
        self._agent_col: int = 0
        self._target_tr: int = 0
        self._target_tc: int = 0
        self._center_row: int = 1
        self._center_col: int = 1
        self._episode_steps: int = 0

        self._last_frame: np.ndarray | None = None

    # --- Gymnasium API -------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._episode_steps = 0
        n = self.n

        max_tl = n - 3
        self._target_tr = int(self.np_random.integers(0, max_tl + 1))
        self._target_tc = int(self.np_random.integers(0, max_tl + 1))
        self._center_row = self._target_tr + 1
        self._center_col = self._target_tc + 1

        for _ in range(10_000):
            ar = int(self.np_random.integers(0, n))
            ac = int(self.np_random.integers(0, n))
            if (ar, ac) != (self._center_row, self._center_col):
                self._agent_row, self._agent_col = ar, ac
                break
        else:
            raise RuntimeError("Could not place agent off the goal cell; check RNG / grid size.")

        obs = self._get_obs()
        info = {
            "agent": (self._agent_row, self._agent_col),
            "target_center": (self._center_row, self._center_col),
            "target_top_left": (self._target_tr, self._target_tc),
        }
        return obs, info

    def step(self, action: int | np.integer) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        if int(action) not in (0, 1, 2, 3):
            raise ValueError(f"action must be in 0..3, got {action}")

        dr, dc = _ACTION_DELTA[int(action)]
        n = self.n
        dist_before = _manhattan(
            self._agent_row,
            self._agent_col,
            self._center_row,
            self._center_col,
        )
        # Clipping at boundaries: moves that would leave the grid leave position unchanged.
        self._agent_row = int(np.clip(self._agent_row + dr, 0, n - 1))
        self._agent_col = int(np.clip(self._agent_col + dc, 0, n - 1))

        self._episode_steps += 1

        dist_after = _manhattan(
            self._agent_row,
            self._agent_col,
            self._center_row,
            self._center_col,
        )

        at_goal = (self._agent_row, self._agent_col) == (self._center_row, self._center_col)
        terminated = bool(at_goal)
        truncated = False
        if self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps:
            truncated = True

        if at_goal:
            reward = _REWARD_GOAL
        else:
            reward = _REWARD_STEP
            if dist_after < dist_before:
                reward += _REWARD_MOVE_TOWARD

        obs = self._get_obs()
        info = {
            "agent": (self._agent_row, self._agent_col),
            "target_center": (self._center_row, self._center_col),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            gym.logger.warning("render() called with render_mode=None; nothing to display.")
            return None

        frame = self._render_frame()
        self._last_frame = frame

        if self.render_mode == "human":
            try:
                import cv2  # type: ignore[import-untyped]

                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Chase", bgr)
                cv2.waitKey(1)
            except Exception:
                gym.logger.warning(
                    "render_mode=human requires OpenCV (cv2) for display; "
                    "install opencv-python or use render_mode='rgb_array'."
                )
        return frame

    def close(self) -> None:
        if self.render_mode == "human":
            try:
                import cv2  # type: ignore[import-untyped]

                cv2.destroyWindow("Chase")
            except Exception:
                pass

    # --- Internals -----------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        img = np.empty((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
        img[...] = _COLOR_BG

        cp = self.cell_px
        tr, tc = self._target_tr, self._target_tc

        for dr, dc in _X_LOCAL_CELLS:
            r, c = tr + dr, tc + dc
            self._fill_cell(img, r, c, _COLOR_X)

        self._fill_cell(img, self._agent_row, self._agent_col, _COLOR_AGENT)

        return img

    def _fill_cell(self, img: np.ndarray, row: int, col: int, color: np.ndarray) -> None:
        cp = self.cell_px
        r0, r1 = row * cp, (row + 1) * cp
        c0, c1 = col * cp, (col + 1) * cp
        img[r0:r1, c0:c1] = color


def register_chase_v0() -> None:
    """Register ``Chase-v0`` with Gymnasium if not already registered."""
    try:
        gym.register(
            id="Chase-v0",
            entry_point="env.chase_env:ChaseEnv",
            kwargs={"n": 64},
        )
    except Exception as exc:  # noqa: BLE001 — idempotent re-import / notebook re-runs
        msg = str(exc).lower()
        if "already" in msg or "duplicate" in msg or "registered" in msg:
            return
        raise
