# Android Porting Notes

## Feasibility

**Short answer: Yes, but the effort depends heavily on the approach.**

---

## What Needs to Be Replaced

The core logic (`raytracing.py`, `geom.py`, `mirror.py`, etc.) is pure math — NumPy array ops. That's the good news. The bad news is everything `scope.py` touches for I/O:

| Current | Android replacement |
|---|---|
| `cv2.namedWindow` / `cv2.imshow` | Android `SurfaceView` or `TextureView` |
| `cv2.waitKey` / mouse callbacks | Android touch events |
| `cv2.VideoCapture` (camera) | Android Camera2 API or CameraX |
| `cv2.remap` (pixel map lookup) | RenderScript, OpenGL ES shader, or Vulkan compute |
| `argparse` CLI | Android settings UI |
| `scipy.ndimage` filters (stain mask) | Custom convolution or OpenCV Android |
| `threading.Thread` + `ThreadPoolExecutor` | Android coroutines / `ExecutorService` |

---

## Option 1: Kivy + Python-for-Android (easiest, worst performance)

- Wrap the existing Python code in a [Kivy](https://kivy.org/) UI
- **OpenCV for Android** exists but is heavy; `cv2.remap` works but slowly
- NumPy works via Python-for-Android (p4a)
- The render loop in `scope.py:start()` is a tight `while` loop calling `cv2.imshow` — replace with Kivy's clock/canvas
- **Reality:** Will probably run at 5–15 FPS at low resolution. `cv2.remap` on a full frame per tick is the bottleneck.

**Effort:** ~2–4 weeks for a working prototype. Not production quality.

---

## Option 2: BeeWare / Briefcase (Python → Android packaging)

- Similar story to Kivy — Python stays, UI layer changes
- Less mature than Kivy for real-time rendering
- Not recommended for a camera + live render app

---

## Option 3: Native Android (Kotlin/Java) — best performance, most work

- Rewrite the raytracer in Kotlin or C++ (via NDK)
- The ray map (`float_map`) is computed once and cached — fine to do in a background coroutine
- `cv2.remap` becomes a **GLSL fragment shader** (just a texture lookup) — trivially fast on GPU
- Camera via **CameraX** (modern, well-documented)
- Touch gestures replace mouse callbacks
- **This is the right approach** if you want a real app

**Effort:** 4–8 weeks for an experienced Android dev who understands the raytrace math.

---

## Option 4: Flutter + Dart with FFI to C++ raytracer (good middle ground)

- Write the raytracer in C++ (portable from Python — it's all array math)
- Call it from Dart via FFI
- Use Flutter's `CustomPainter` or a texture for display
- `cv2.remap` → custom fragment shader or CPU remap in C++
- Cross-platform: Android + iOS + desktop

**Effort:** 4–6 weeks if comfortable with C++.

---

## Key Architectural Insight

The `scope.py` render loop is already well-structured for porting. The entire hot path per frame is just three lines (`scope.py:374-376`):

```python
frame_out = cv2.remap(input_frame, x_map, y_map, interp)  # → GPU shader
frame_out[oob_mask] = BKG                                   # → shader clip
self._apply_art_layers(...)                                 # → shader pass
```

The **ray map is computed once** (background thread, `scope.py:128-138`), then `cv2.remap` is just a table lookup applied every frame. On Android GPU this would be trivially fast. The hard part is the one-time raytrace (NumPy-heavy), which you'd either rewrite in C++/Kotlin or accept running slowly once at startup.

---

## Recommendation

- **For a real Android app:** Option 3 or 4. The raytrace math is straightforward to port to C++ (mostly array indexing and dot products). The UI in `user_interface.py` (menu, drag controls) is the most tedious part to rewrite for touch idioms.
- **For a quick demo on Android:** Kivy + p4a — lower your resolution expectations to ~480p.
