# Air Canvas using OpenCV + MediaPipe (Windows)

Draw in the air with your index finger and see it appear on a canvas. Control colors, thickness, undo, clear, and save — all using hand gestures (pinch) and an on-screen toolbar.

## Features
- Hand tracking via MediaPipe Hands (single hand)
- Draw with index finger; pinch index + thumb to "click"
- Toolbar with: Clear, Undo, Save, 4 colors, Eraser, Thickness -/+
- Keyboard shortcuts: `q` quit, `c` clear, `u` undo, `r` redo, `s` save
- Saves canvas as PNG without the toolbar/landmarks

## Installation (Windows)
1. Open PowerShell in this folder.
2. (Recommended) Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

If you face issues installing `mediapipe`, ensure you have Python 3.9–3.11 and latest pip:
```powershell
py -m pip install --upgrade pip
```

## Run
```powershell
py air_canvas.py
```

- Raise your index finger to draw.
- Pinch the index and thumb to click UI buttons in the top header.
- Pinch in the drawing area to lift the pen (end the current stroke).

## Tips
- Ensure only one app uses the webcam at a time.
- Lighting matters — better lighting improves hand detection.
- Adjust the pinch threshold inside `air_canvas.py` if clicks misfire.

## Troubleshooting
- Cannot open camera: close other apps and verify device permissions.
- ImportError: mediapipe: run `pip install -r requirements.txt`. On some systems, `mediapipe` requires specific Python versions.
- Window not responding: click the window first; use `q` to quit.

## Credits
- Built with OpenCV and MediaPipe by Google.
