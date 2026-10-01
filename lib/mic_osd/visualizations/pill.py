"""Compact monochrome pill visualization for mic-osd."""

import math
import time

import cairo
import numpy as np

from .base import AutoGain, BaseVisualization, StateManager, VisualizerState


class PillVisualization(BaseVisualization):
    """A small black capsule with a state dot and live white level bars.

    Only recording animates (the bars follow the mic). Every other state is
    drawn static, so the window stops repainting once the bars have settled.
    """

    show_preview = True
    preview_mode = "pill"

    # Distance from the screen edge to the bottom of the layer surface. The
    # pill sits PILL_PAD above the surface bottom, so it ends ~56px up.
    BOTTOM_MARGIN = 52
    PILL_PAD = 4.0

    PILL_WIDTH = 108
    PILL_HEIGHT = 34
    DOT_RADIUS = 4.0
    NUM_BARS = 9
    BAR_WIDTH = 2.5
    BAR_GAP = 3.5
    MIN_HEIGHT = 2.5
    MAX_HEIGHT = 16.0
    NOISE_GATE = 0.006
    INPUT_GAIN = 18.0

    # Palette is part of the style, so it works without a theme file.
    BACKGROUND = (0.0, 0.0, 0.0, 0.90)
    BORDER = (1.0, 1.0, 1.0, 0.15)
    RED = (1.0, 0.271, 0.227, 1.0)          # #FF453A
    DOT_COLORS = {
        VisualizerState.RECORDING: RED,
        VisualizerState.PAUSED: (1.0, 0.624, 0.039, 0.70),   # Dim amber
        VisualizerState.PROCESSING: (1.0, 1.0, 1.0, 0.55),
        VisualizerState.ERROR: RED,
        VisualizerState.SUCCESS: (1.0, 1.0, 1.0, 1.0),
    }
    BAR_COLORS = {
        VisualizerState.RECORDING: (1.0, 1.0, 1.0, 0.94),
        VisualizerState.PAUSED: (1.0, 1.0, 1.0, 0.35),
        VisualizerState.PROCESSING: (1.0, 1.0, 1.0, 0.45),
        VisualizerState.ERROR: (1.0, 0.271, 0.227, 0.80),
        VisualizerState.SUCCESS: (1.0, 1.0, 1.0, 0.94),
    }

    # Bar heights below this per-frame change count as settled.
    SETTLE_EPSILON = 0.002

    def __init__(self):
        super().__init__()
        self.num_bars = self.NUM_BARS
        self.bar_heights = np.zeros(self.num_bars, dtype=np.float64)
        self.state_manager = StateManager()
        self._last_update = time.monotonic()
        # Read by OSDWindow.update(): False once a static state has settled.
        self.needs_redraw = True
        # INPUT_GAIN was the old fixed value, now the floor for hot mics
        self.auto_gain = AutoGain(min_gain=self.INPUT_GAIN, noise_floor=0.002)

    @staticmethod
    def _rounded_rect(cr: cairo.Context, x: float, y: float, width: float,
                      height: float, radius: float):
        radius = min(radius, width / 2.0, height / 2.0)
        cr.new_sub_path()
        cr.arc(x + width - radius, y + radius, radius, -math.pi / 2, 0)
        cr.arc(x + width - radius, y + height - radius, radius, 0, math.pi / 2)
        cr.arc(x + radius, y + height - radius, radius, math.pi / 2, math.pi)
        cr.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
        cr.close_path()

    @staticmethod
    def _resample(values: np.ndarray, count: int) -> np.ndarray:
        if values is None or len(values) == 0:
            return np.zeros(count, dtype=np.float64)
        values = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
        if len(values) == count:
            return values
        old_x = np.linspace(0.0, 1.0, len(values))
        new_x = np.linspace(0.0, 1.0, count)
        return np.interp(new_x, old_x, values)

    def set_state(self, state_str: str):
        self.state_manager.set_state_from_string(state_str or "recording")
        self.needs_redraw = True

    def update(self, level: float, samples: np.ndarray = None):
        super().update(level, samples)
        now = time.monotonic()
        dt = min(0.05, max(0.001, now - self._last_update))
        self._last_update = now

        state = self.state_manager.current_state
        positions = np.linspace(0.0, 1.0, self.num_bars)

        if state == VisualizerState.RECORDING:
            audio = self._resample(samples, self.num_bars)
            # level is pre-scaled (raw RMS x10), so this substitute is on a
            # different scale than the buckets — keep it out of the gain
            # envelope, which would read it as a 10x louder mic.
            from_samples = bool(np.any(audio))
            if not from_samples:
                audio[:] = max(0.0, float(level))

            # Gate mic noise out of the RMS feed, then compress so speech fills
            # the pill. Both track the input's own level: a fixed gate would
            # swallow a quiet mic whole, whose speech sits under it.
            if from_samples:
                gain = self.auto_gain.update(float(audio.max()))
                gate = self.auto_gain.gate(self.NOISE_GATE)
            else:
                gain, gate = self.INPUT_GAIN, self.NOISE_GATE
            energy = np.sqrt(np.clip(
                np.maximum(audio - gate, 0.0) * gain,
                0.0,
                1.0,
            ))
            center_envelope = 0.62 + 0.38 * np.sin(math.pi * positions)
            target = energy * center_envelope
        else:
            # Paused, processing, error and success are static: flat bars,
            # with the state shown by the dot and bar colour.
            target = np.zeros(self.num_bars)

        rise = 1.0 - math.exp(-dt * 18.0)
        fall = 1.0 - math.exp(-dt * 9.0)
        blend = np.where(target > self.bar_heights, rise, fall)
        delta = (target - self.bar_heights) * blend
        self.bar_heights += delta

        self.needs_redraw = (
            state == VisualizerState.RECORDING
            or float(np.max(np.abs(delta))) > self.SETTLE_EPSILON
            or self._success_fading()
        )

    def _pill_geometry(self, width: int, height: int):
        pill_w = min(self.PILL_WIDTH, width - 4)
        pill_h = min(self.PILL_HEIGHT, height - 4)
        # Whole pixels keep the 1px hairline border crisp.
        x = float(round((width - pill_w) / 2.0))
        # Bottom-aligned: any space above is reserved for the preview text.
        y = height - pill_h - self.PILL_PAD
        return x, y, pill_w, pill_h

    def _success_elapsed(self) -> float:
        return time.time() - self.state_manager.state_changed_at

    def _success_fading(self) -> bool:
        return (
            self.state_manager.current_state == VisualizerState.SUCCESS
            and self._success_elapsed() <= 1.0
        )

    def _success_fade(self) -> float:
        if self.state_manager.current_state != VisualizerState.SUCCESS:
            return 1.0
        elapsed = self._success_elapsed()
        if elapsed <= 0.72:
            return 1.0
        return max(0.0, 1.0 - (elapsed - 0.72) / 0.28)

    def draw_background(self, cr: cairo.Context, width: int, height: int):
        x, y, pill_w, pill_h = self._pill_geometry(width, height)
        alpha = self._success_fade()

        r, g, b, a = self.BACKGROUND
        self._rounded_rect(cr, x, y, pill_w, pill_h, pill_h / 2.0)
        cr.set_source_rgba(r, g, b, a * alpha)
        cr.fill()

        # Inset by half a pixel so the 1px stroke lands on whole pixels.
        r, g, b, a = self.BORDER
        self._rounded_rect(
            cr, x + 0.5, y + 0.5, pill_w - 1.0, pill_h - 1.0,
            (pill_h - 1.0) / 2.0,
        )
        cr.set_source_rgba(r, g, b, a * alpha)
        cr.set_line_width(1.0)
        cr.stroke()

    def draw(self, cr: cairo.Context, width: int, height: int):
        x, y, pill_w, pill_h = self._pill_geometry(width, height)
        center_y = y + pill_h / 2.0
        state = self.state_manager.current_state
        alpha = self._success_fade()

        # State dot, centred in the left cap of the capsule.
        dot_x = x + pill_h / 2.0
        r, g, b, a = self.DOT_COLORS.get(state, self.RED)
        cr.arc(dot_x, center_y, self.DOT_RADIUS, 0, 2 * math.pi)
        cr.set_source_rgba(r, g, b, a * alpha)
        cr.fill()

        # Bars fill the space between the dot and the right cap.
        total_width = (
            self.num_bars * self.BAR_WIDTH
            + (self.num_bars - 1) * self.BAR_GAP
        )
        area_left = dot_x + self.DOT_RADIUS
        area_right = x + pill_w - pill_h / 2.0 + self.DOT_RADIUS
        start_x = (area_left + area_right - total_width) / 2.0

        r, g, b, a = self.BAR_COLORS.get(state, self.BAR_COLORS[
            VisualizerState.RECORDING
        ])
        cr.set_source_rgba(r, g, b, a * alpha)
        for i in range(self.num_bars):
            bar_x = start_x + i * (self.BAR_WIDTH + self.BAR_GAP)
            height_norm = float(np.clip(self.bar_heights[i], 0.0, 1.0))
            bar_h = self.MIN_HEIGHT + height_norm * (
                self.MAX_HEIGHT - self.MIN_HEIGHT
            )
            self._rounded_rect(
                cr,
                bar_x,
                center_y - bar_h / 2.0,
                self.BAR_WIDTH,
                bar_h,
                self.BAR_WIDTH / 2.0,
            )
        cr.fill()
