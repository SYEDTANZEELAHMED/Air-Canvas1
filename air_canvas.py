import cv2
import time
import os
import math
from typing import List, Tuple, Dict

import numpy as np

try:
	import mediapipe as mp
except ImportError:
	raise SystemExit("mediapipe is required. Run: pip install -r requirements.txt")


# ------------------------------
# Configuration
# ------------------------------
WINDOW_NAME = "Air Canvas"
HEADER_HEIGHT = 70
BUTTON_PADDING = 6
BUTTON_HEIGHT = HEADER_HEIGHT - 2 * BUTTON_PADDING
BUTTON_TEXT_COLOR = (255, 255, 255)
LANDMARK_COLOR = (0, 255, 0)
CURSOR_COLOR = (0, 255, 0)
CURSOR_RADIUS = 6

# Default palette (BGR)
PALETTE: List[Tuple[int, int, int]] = [
	(255, 0, 0),   # Blue
	(0, 255, 0),   # Green
	(0, 0, 255),   # Red
	(0, 255, 255), # Yellow
	(0, 0, 0),     # Black (for UI text contrast; not a drawing color here)
]
ERASER_COLOR = (255, 255, 255)

DEFAULT_COLOR = PALETTE[0]
DEFAULT_THICKNESS = 6
THICKNESS_MIN = 2
THICKNESS_MAX = 32

PINCH_THRESHOLD_PX = 40  # Distance between index tip and thumb tip to be considered a "click"

# Buttons layout will be computed dynamically based on frame width


# ------------------------------
# Utilities
# ------------------------------
class Button:
	def __init__(self, label: str, color: Tuple[int, int, int], x1: int, y1: int, x2: int, y2: int):
		self.label = label
		self.color = color
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

	def draw(self, img: np.ndarray, invert_text: bool = False) -> None:
		cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), self.color, thickness=-1)
		text_color = (0, 0, 0) if invert_text else BUTTON_TEXT_COLOR
		cv2.putText(
			img,
			self.label,
			(self.x1 + 10, self.y1 + int(BUTTON_HEIGHT * 0.65)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			text_color,
			2,
			cv2.LINE_AA,
		)

	def contains(self, x: int, y: int) -> bool:
		return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
	return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ------------------------------
# Stroke model
# ------------------------------
class Stroke:
	def __init__(self, color: Tuple[int, int, int], thickness: int):
		self.color = color
		self.thickness = thickness
		self.points: List[Tuple[int, int]] = []

	def add_point(self, pt: Tuple[int, int]) -> None:
		self.points.append(pt)

	def is_empty(self) -> bool:
		return len(self.points) < 2


# ------------------------------
# UI construction based on frame size
# ------------------------------


def build_buttons(frame_width: int) -> Dict[str, Button]:
	buttons: Dict[str, Button] = {}
	x = BUTTON_PADDING

	def add_button(key: str, label: str, color: Tuple[int, int, int], width: int) -> None:
		nonlocal x
		buttons[key] = Button(
			label=label,
			color=color,
			x1=x,
			y1=BUTTON_PADDING,
			x2=x + width,
			y2=BUTTON_PADDING + BUTTON_HEIGHT,
		)
		x += width + BUTTON_PADDING

	# Control buttons
	add_button("clear", "CLEAR", (60, 60, 60), 90)
	add_button("undo", "UNDO", (90, 90, 90), 80)
	add_button("save", "SAVE", (120, 120, 120), 80)
	add_button("mode", "MODE", (70, 70, 70), 80)

	# Palette buttons
	for idx, col in enumerate(PALETTE[:4]):
		add_button(f"color_{idx}", ["BLUE", "GREEN", "RED", "YELLOW"][idx], col, 90)

	# Eraser
	add_button("eraser", "ERASER", (200, 200, 200), 100)

	# Thickness down/up
	add_button("thick_down", "-", (50, 50, 50), 34)
	add_button("thick_label", "THK", (30, 30, 30), 50)
	add_button("thick_up", "+", (50, 50, 50), 34)

	return buttons


# ------------------------------
# Main application
# ------------------------------

def main() -> None:
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise SystemExit("Could not open webcam. Ensure a camera is connected and not in use.")

	# Probe a frame to initialize sizes
	ok, frame = cap.read()
	if not ok:
		cap.release()
		raise SystemExit("Failed to read from webcam.")

	frame_height, frame_width = frame.shape[:2]
	canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
	buttons = build_buttons(frame_width)

	mp_hands = mp.solutions.hands
	mp_draw = mp.solutions.drawing_utils
	hands = mp_hands.Hands(
		static_image_mode=False,
		max_num_hands=1,
		min_detection_confidence=0.6,
		min_tracking_confidence=0.5,
	)

	current_color = DEFAULT_COLOR
	current_thickness = DEFAULT_THICKNESS
	strokes: List[Stroke] = []
	redo_stack: List[Stroke] = []
	active_stroke: Stroke = None  # type: ignore
	was_pinching = False
	always_draw_mode = False  # False: pinch-to-click UI, draw when not pinching. True: always draw without pinch
	pinch_threshold = PINCH_THRESHOLD_PX

	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(WINDOW_NAME, frame_width, frame_height)

	last_save_time = 0.0

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		frame = cv2.flip(frame, 1)
		raw_frame = frame.copy()

		# Draw header background
		header = frame.copy()
		cv2.rectangle(header, (0, 0), (frame_width, HEADER_HEIGHT), (30, 30, 30), thickness=-1)
		# Draw buttons
		for key, btn in buttons.items():
			invert = key in {"clear", "undo", "save", "thick_down", "thick_label", "thick_up"}
			btn.draw(header, invert_text=invert)
		# Show thickness value
		thk_btn = buttons["thick_label"]
		cv2.putText(
			header,
			str(current_thickness),
			(thk_btn.x1 + 6, thk_btn.y1 + int(BUTTON_HEIGHT * 0.62)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)

		# Overlay header onto frame
		frame[0:HEADER_HEIGHT, :, :] = header[0:HEADER_HEIGHT, :, :]

		# Status text (mode + pinch threshold)
		status_text = f"MODE: {'ALWAYS-DRAW' if always_draw_mode else 'PINCH UI / DRAW NO-PINCH'} | THRESH: {pinch_threshold}px"
		cv2.putText(
			frame,
			status_text,
			(10, HEADER_HEIGHT + 24),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(60, 60, 60),
			2,
			cv2.LINE_AA,
		)

		# Draw last selection toast if available
		if 'last_selection' in locals():
			msg, msg_time = last_selection
			if time.time() - msg_time < 1.2:
				cv2.rectangle(frame, (10, HEADER_HEIGHT + 34), (10 + 260, HEADER_HEIGHT + 64), (30, 30, 30), -1)
				cv2.putText(frame, msg, (18, HEADER_HEIGHT + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

		# Hand processing
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		result = hands.process(rgb)

		# Prepare PiP preview area (top-right)
		# Place preview at bottom-right (avoid header overlap)
		pip_w = max(160, min(240, frame_width // 5))
		pip_h = int(pip_w * 0.75)
		pip_x2 = frame_width - 10
		pip_y2 = frame_height - 10
		pip_x1 = pip_x2 - pip_w
		pip_y1 = pip_y2 - pip_h
		preview = cv2.resize(raw_frame, (pip_w, pip_h))
		cv2.rectangle(frame, (pip_x1 - 2, pip_y1 - 2), (pip_x2 + 2, pip_y2 + 2), (40, 40, 40), 2)
		frame[pip_y1:pip_y2, pip_x1:pip_x2] = preview

		index_tip = None  # type: Tuple[int, int] | None
		thumb_tip = None  # type: Tuple[int, int] | None

		if result.multi_hand_landmarks:
			for hand_landmarks in result.multi_hand_landmarks:
				# Draw landmarks
				mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

				# Collect landmark positions (normalized -> pixel)
				pts = []
				for lm in hand_landmarks.landmark:
					x = int(lm.x * frame_width)
					y = int(lm.y * frame_height)
					pts.append((x, y))

				index_tip = pts[8]
				thumb_tip = pts[4]

		if index_tip is not None:
			# Smooth cursor with exponential moving average
			if 'smooth_pt' not in locals():
				smooth_pt = index_tip
			else:
				alpha = 0.35  # smoothing factor
				smooth_pt = (
					int((1 - alpha) * smooth_pt[0] + alpha * index_tip[0]),
					int((1 - alpha) * smooth_pt[1] + alpha * index_tip[1]),
				)
			cv2.circle(frame, smooth_pt, CURSOR_RADIUS, CURSOR_COLOR, -1)

		# Determine pinch (click)
		is_pinching = False
		if index_tip is not None and thumb_tip is not None:
			is_pinching = distance(index_tip, thumb_tip) < pinch_threshold

		# Handle interactions
		if index_tip is not None:
			x, y = index_tip

			# Click on header buttons when pinching in header area (UI interaction)
			if y <= HEADER_HEIGHT and is_pinching:
				for key, btn in buttons.items():
					if btn.contains(x, y):
						if key == "clear":
							strokes.clear()
							redo_stack.clear()
							canvas[:] = 255
							last_selection = ("CLEARED", time.time())
						elif key == "undo":
							if strokes:
								redo_stack.append(strokes.pop())
								last_selection = ("UNDO", time.time())
						elif key == "save":
							# Debounce save to avoid multiple on a single press
							if time.time() - last_save_time > 0.75:
								last_save_time = time.time()
								# Save the pure canvas (without header/landmarks)
								stamp = time.strftime("%Y%m%d_%H%M%S")
								filename = f"air_canvas_{stamp}.png"
								cv2.imwrite(filename, canvas)
								print(f"Saved: {filename}")
								last_selection = ("SAVED", time.time())
						elif key.startswith("color_"):
							idx = int(key.split("_")[1])
							current_color = PALETTE[idx]
							last_selection = ((["BLUE", "GREEN", "RED", "YELLOW"][idx]), time.time())
						elif key == "eraser":
							current_color = ERASER_COLOR
							last_selection = ("ERASER", time.time())
						elif key == "thick_down":
							current_thickness = max(THICKNESS_MIN, current_thickness - 2)
							last_selection = (f"THK {current_thickness}", time.time())
						elif key == "thick_up":
							current_thickness = min(THICKNESS_MAX, current_thickness + 2)
							last_selection = (f"THK {current_thickness}", time.time())
						elif key == "mode":
							always_draw_mode = not always_draw_mode
							last_selection = (("MODE: ALWAYS" if always_draw_mode else "MODE: PINCH"), time.time())

				# When clicking buttons, end any active stroke
				active_stroke = None

			# Drawing area interaction (below header)
			elif y > HEADER_HEIGHT:
				should_draw = always_draw_mode or (not is_pinching)
				if should_draw:
					# Drawing mode
					if active_stroke is None:
						active_stroke = Stroke(current_color, current_thickness)
						strokes.append(active_stroke)
						redo_stack.clear()
					# Use smoothed point if available
					pt_to_add = smooth_pt if 'smooth_pt' in locals() else (x, y)
					active_stroke.add_point(pt_to_add)
				else:
					# End stroke when not drawing
					active_stroke = None

		# Keyboard shortcuts
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		elif key == ord('c'):
			strokes.clear()
			redo_stack.clear()
			canvas[:] = 255
		elif key == ord('u'):
			if strokes:
				redo_stack.append(strokes.pop())
		elif key == ord('r'):
			if redo_stack:
				strokes.append(redo_stack.pop())
		elif key == ord('s'):
			stamp = time.strftime("%Y%m%d_%H%M%S")
			filename = f"air_canvas_{stamp}.png"
			cv2.imwrite(filename, canvas)
			print(f"Saved: {filename}")
		elif key == ord('m'):
			always_draw_mode = not always_draw_mode
		elif key == ord('['):
			pinch_threshold = max(10, pinch_threshold - 5)
		elif key == ord(']'):
			pinch_threshold = min(150, pinch_threshold + 5)
		elif key == ord('1'):
			current_color = PALETTE[0]; last_selection = ("BLUE", time.time())
		elif key == ord('2'):
			current_color = PALETTE[1]; last_selection = ("GREEN", time.time())
		elif key == ord('3'):
			current_color = PALETTE[2]; last_selection = ("RED", time.time())
		elif key == ord('4'):
			current_color = PALETTE[3]; last_selection = ("YELLOW", time.time())
		elif key == ord('e'):
			current_color = ERASER_COLOR; last_selection = ("ERASER", time.time())

		# Redraw canvas from strokes
		canvas[:] = 255
		for s in strokes:
			if len(s.points) >= 2:
				for i in range(1, len(s.points)):
					cv2.line(canvas, s.points[i - 1], s.points[i], s.color, s.thickness, lineType=cv2.LINE_AA)

		# Compose output: put canvas under header, then overlay on frame lightly
		output = frame.copy()
		alpha = 0.25
		# Mix only the drawing region (below header)
		roi = output[HEADER_HEIGHT:, :, :]
		roi_canvas = canvas[HEADER_HEIGHT:, :, :]
		cv2.addWeighted(roi_canvas, 1.0, roi, alpha, 0, dst=roi)
		output[HEADER_HEIGHT:, :, :] = roi

		# Indicate current color in header by drawing border around the active color/eraser
		active_keys = []
		if current_color == ERASER_COLOR:
			active_keys.append("eraser")
		else:
			for i, col in enumerate(PALETTE[:4]):
				if current_color == col:
					active_keys.append(f"color_{i}")
		for k in active_keys:
			btn = buttons[k]
			cv2.rectangle(output, (btn.x1, btn.y1), (btn.x2, btn.y2), (255, 255, 255), thickness=2)

		cv2.imshow(WINDOW_NAME, output)

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
