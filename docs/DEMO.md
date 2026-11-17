# Demonstrating NetrAI

A 60–90 second screen recording can show MATLAB image processing, the generated
report and the SimEvents capacity model. A recording has not yet been published.

## The screening console (recommended for a live demonstration)

`netraiApp` puts the whole demonstration in one window. It runs the same `main_pipeline` as the command line and shows only what
that call returned.

```matlab
netraiApp                                  % or: netraiApp(Image="/absolute/path/to/fundus.png")
```

From a Linux terminal: `./scripts/launch_demo.sh --app /absolute/path/to/fundus.png`.

1. **Choose image…**, then **Run screening**. The first run in a MATLAB session loads the model (up to two minutes for E_768);
   run one image before the audience arrives. Later images take seconds.
2. The right-hand panel gives the routing decision in one word (REFER, NO REFERRAL, HUMAN REVIEW or RECAPTURE), the screening
   level, the most probable grade, the referral output, the image-quality score, and the warnings: an uncalibrated cut-off, an
   input that does not look like a fundus photograph, or an attention map flagged by its quality check.
3. **Photograph / Model attention / Lesion candidates** switch the picture. The attention map is Grad-CAM of the referral output;
   the lesion marks are unverified candidates.
4. **Site calibration file…** loads a policy written by `calibrateSite`; the cut-off line then reads "Calibrated for this site".
5. The **District capacity (SimEvents)** tab shows the recorded annual scenarios for one or two reviewers and three recapture
   rates, and opens the SimEvents model. These are recorded runs under stated assumptions, not field data.
6. **Open the PDF report** opens the report of the current image.
7. Timing on the tested laptop: about 19 s per image with E_768 and the PDF report, about 11 s without the PDF. Untick
   **Also write the PDF report** when speed matters. The learned lesion picture (offered only when the research lesion model
   is installed locally; it is not distributed) adds about 40 s.

A failed step shows its error message; the console never substitutes a result. Say what the screen says: "no referral" is not a
clearance, and the prototype is not a diagnosis.

## Prepare the session (command-line demonstration)

From a **Linux terminal**, run the launcher at the repository root:

```bash
./scripts/launch_demo.sh /absolute/path/to/fundus.png
```

If your local checkout has an image in the documented dataset layout, omitting
the image argument uses that local sample. A fresh clone requires your own image.
The launcher opens MATLAB, produces the report and opens the capacity model.
Allow the first model import to finish before recording.

For manual control, enter the following commands in the **MATLAB Command Window**.

1. Open the repository's `matlab` folder in MATLAB and run `verifyMatlabPackage`.
2. Choose a local fundus image whose terms permit showing it in a shared recording. Use a neutral filename and remove personal identifiers.
3. Run the demo once before recording so model import is complete and the report is ready:

```matlab
result = demo_netrai("/absolute/path/to/fundus.png");
```

Quality enforcement is enabled. A rejected image returns recapture feedback and
no grading report. An accepted image produces a new report directory; open
`result.reportPath`. Grad-CAM and candidate overlays retain their warnings. An
image that does not look like a fundus photograph is still graded but carries a
red warning in the console and the report (`result.inputWarning`).
Do not present a candidate detection as a confirmed diagnosis.

Prepare the capacity model in another MATLAB tab:

```matlab
open_system('models/NetrAI_Telemedicine_Model.slx')
```

Use the [recorded scenario table](../results/benchmarks/capacity_simulation/simulink_scenario_sweep.csv)
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
