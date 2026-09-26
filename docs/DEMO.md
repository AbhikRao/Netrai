# Demonstrating NetrAI

A 60–90 second screen recording can show MATLAB image processing, the generated
report and the SimEvents capacity model. A recording has not yet been published.

## Prepare the session

1. Open the repository's `matlab` folder in MATLAB and run `verifyMatlabPackage`.
2. Choose a local fundus image whose terms permit showing it in a shared recording. Use a neutral filename and remove personal identifiers.
3. Run the demo once before recording so model import is complete and the report is ready:

```matlab
result = demo_netrai("/absolute/path/to/fundus.png");
```

Quality enforcement is enabled. A rejected image returns recapture feedback and
no grading report. An accepted image produces a new report directory; open
`result.reportPath`. Grad-CAM and candidate overlays retain their warnings.
Do not present a candidate detection as a confirmed diagnosis.

Prepare the capacity model in another MATLAB tab:

```matlab
open_system('models/NetrAI_Telemedicine_Model.slx')
```

Use the [recorded scenario table](../results/implementation_2026-09-24/m5_simevents/simulink_scenario_sweep.csv)
instead of running an annual sweep during the recording.

## Recording on Ubuntu GNOME

On the tested desktop, **Ctrl+Alt+Shift+R** opens the screen-recording controls.
Select the MATLAB/report area or the screen and start recording. Stop using the
recording indicator in the top bar. Videos are saved in **Videos/Screencasts**.
GNOME's built-in recorder does not record microphone narration; use captions,
a separate voiceover, or a recorder with microphone support if narration is needed.
[GNOME recording instructions](https://help.gnome.org/users/gnome-help/stable/screen-shot-record.html.en)

Close notifications and unrelated windows. Increase MATLAB text size before
recording and keep the report readable. Make a short test recording first and
check that all relevant text is legible.

## Suggested sequence

| Time | Show | Explain |
| --- | --- | --- |
| 0–10 s | MATLAB and input image | NetrAI is a MATLAB/SimEvents screening prototype |
| 10–30 s | Quality, grade, referral output and routing | Acquisition feedback and shared-backbone disagreement routing |
| 30–65 s | Generated report | Model attribution, candidate overlays and any QC warning; validation remains in progress |
| 65–85 s | SimEvents model and one bandwidth comparison | Scenario results describe capacity under assumptions |
| 85–90 s | Repository address | Source and validation evidence: github.com/AbhikRao/Netrai |

It is acceptable to show precomputed outputs while saying so. Do not imply the
recording proves model accuracy or clinician review time. Before sharing the video,
check its image permissions, visible paths and report identifiers. Add a demo link
or QR only after the recording exists and opens without authentication.
