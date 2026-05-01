Now I have everything. Here's the complete frontend audit:

---

## 🚨 FRONTEND — COMPLETE ISSUE REPORT

---

**BUG 1 — CRITICAL: WebSocket double-connects (camera never works correctly)**

`useWebSocket.js` has a `useEffect` that calls `connect()` on mount automatically. `WebcamDetector` also calls `connect()` manually when user clicks "Start Camera". So the WebSocket opens **twice** — once silently on page load, once on button click. The second `connect()` creates a new `WebSocket` object but `wsRef.current` now points to the new one. The first connection leaks and stays open. The backend receives frames from both, creating duplicate inference calls and response confusion.

Fix — remove the auto-connect `useEffect` from `useWebSocket.js`:
```javascript
// DELETE these lines from useWebSocket.js:
useEffect(() => {
  connect();
  return () => { disconnect(); };
}, [connect, disconnect]);
```
Let `WebcamDetector` control connect/disconnect explicitly via `startCamera()` and `stopCamera()`.

---

**BUG 2 — CRITICAL: Canvas dual-use erases bounding boxes every 100ms**

`captureAndSendFrame()` sets `canvas.width = video.videoWidth` every `frameInterval` ms. Setting `.width` on a canvas **clears it entirely**. So every 100ms the bounding boxes that `drawDetectionBoxes()` just drew get wiped. Boxes are never visible.

Fix — use a separate offscreen canvas for capture only:
```javascript
const captureAndSendFrame = () => {
  if (!videoRef.current) return;
  const offscreen = document.createElement('canvas');
  offscreen.width = videoRef.current.videoWidth || 640;
  offscreen.height = videoRef.current.videoHeight || 480;
  offscreen.getContext('2d').drawImage(videoRef.current, 0, 0);
  const frameBase64 = offscreen.toDataURL('image/jpeg', 0.8).split(',')[1];
  if (frameBase64) sendFrame(frameBase64);
  // DO NOT touch canvasRef here
};
```
Set the display canvas size once when video loads:
```javascript
videoRef.current.onloadedmetadata = () => {
  canvasRef.current.width = videoRef.current.videoWidth;
  canvasRef.current.height = videoRef.current.videoHeight;
};
```

---

**BUG 3 — CRITICAL: Training status field mismatch**

`Settings.jsx` line 51:
```javascript
setIsTraining(status.is_running); // ← undefined always
```
Backend returns `{ "status": "running" }` not `{ "is_running": true }`. So `isTraining` is always `false`. The UI never shows training as active, progress bar never appears, Start button never disables.

Fix:
```javascript
setIsTraining(status.status === "running");
```

---

**BUG 4 — MAJOR: `Detection.jsx` page is fully white/light themed**

The entire Detection page uses light mode Tailwind classes:
```
text-gray-900 / text-gray-600 / bg-white / border-gray-200
text-blue-600 / text-green-600 / text-yellow-600 / text-purple-600
bg-gray-200 / border-gray-300
```
On the dark background it renders as a bright white box. Also the "Detection Statistics" card shows **100% hardcoded values** (24, 89%, 3, 0.2s) that never update from real data.

Fix — rewrite `Detection.jsx` to use CSS variables and pull real stats from `WebcamDetector`'s `lastDetection` state.

---

**BUG 5 — MAJOR: `ActivityFeed.jsx` is fully white themed + 100% fake data**

```javascript
className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
className="text-gray-900"
className="text-yellow-600 bg-yellow-50"
className="hover:bg-gray-50"
```
Also the `activities` array is hardcoded (iPhone 13, Samsung S22 etc.) and never fetches from `/api/alerts`. This component sits on the main Dashboard and shows fake data permanently.

Fix — rewrite to dark theme CSS variables + fetch from `getAlerts()`.

---

**BUG 6 — MAJOR: `RealTimeMonitor` shows fake devices with random usage animation**

```javascript
const [devices, setDevices] = useState([
  { id: 1, name: "iPhone 13", usage: 45, ... },
  { id: 2, name: "Samsung S22", usage: 78, ... },
  ...
]);
useEffect(() => {
  setInterval(() => {
    // randomises usage every 5s ← completely fake
  }, 5000);
}, []);
```
This component is on the Dashboard and shows fake device data. It should fetch from `GET /api/devices`.

Fix — replace hardcoded state with `useEffect` calling `getDevices()` from `api.js`, same pattern as `Devices.jsx` (which already does this correctly).

---

**BUG 7 — MEDIUM: `Reports.jsx` never fetches real data**

Chart and summary cards are 100% hardcoded: `[65, 59, 80, 81, 56, 55, 40]`. Never calls `getDashboardTimeline()` or `getDashboardStats()`. Fix — add `useEffect` to fetch real data on mount.

---

**BUG 8 — MEDIUM: `Detection.jsx` title invisible on dark background**

```jsx
<h1 className="text-2xl md:text-3xl font-bold text-gray-900">
<p className="text-gray-600 mt-2">
```
`text-gray-900` is near-black — invisible on dark `--bg-page`. Fix — replace with `style={{ color: 'var(--text-primary)' }}`.

---

## 📋 COMPLETE FIX LIST FOR YOUR AGENT

```
File                                        Fix
─────────────────────────────────────────────────────────────────
hooks/useWebSocket.js                       Remove auto-connect useEffect
WebcamDetector/WebcamDetector.jsx           Use offscreen canvas for capture
pages/Settings/Settings.jsx  line 51        status.is_running → status.status === "running"
pages/Detection/Detection.jsx               Full rewrite: dark theme + real stats
components/dashboard/ActivityFeed.jsx       Full rewrite: dark theme + fetch from getAlerts()
components/dashboard/RealTimeMonitor.jsx    Replace fake devices with getDevices() API call
pages/Reports/Reports.jsx                   Add useEffect + getDashboardTimeline() for real data
```

---

## ✅ WHAT IS CORRECT — DO NOT TOUCH

```
src/index.css              ✅ CSS variables correct (#0B1120 etc.)
src/services/api.js        ✅ All 17 endpoints correct
src/context/AppContext.jsx ✅ State management correct
src/main.jsx               ✅ AppProvider wrapping correct
AlertsPanel.jsx            ✅ Fetches real data, resolve works
AlertItem.jsx              ✅ Severity colors correct
Dashboard.jsx              ✅ Fetches stats + timeline, correct layout
UsageMetrics.jsx           ✅ Real data when timelineData passed, chart colors correct
Devices.jsx                ✅ Fetches from getDevices() correctly
Settings.jsx               ✅ Settings load/save correct (only training status field wrong)
WebcamDetector.jsx         ✅ bbox logic correct, colors correct — only canvas bug
```