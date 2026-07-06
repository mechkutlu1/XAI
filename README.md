# XAI Sandbox — a client-side Shapley laboratory

Train a small neural network on tabular data in the browser and audit its
predictions with **Kernel SHAP**, **Integrated Gradients**, dependence plots
and **faithfulness diagnostics** — with no server, no build step and no data
leaving the device. Installable as a Progressive Web App.

Built as a teaching aid and as a companion artefact to attach to explainable-AI
manuscripts: reviewers and readers can open the live tool, load their own CSV,
and interrogate the attributions themselves.

## What it does

- **Data** — generate a benchmark dataset drawn from a *known* structural
  equation (so attributions can be validated against ground truth), or upload
  any numeric CSV with a header row and pick the target column. Binary and
  regression targets are detected automatically.
- **Train** — a configurable MLP is fitted on-device with TensorFlow.js
  (WebGL backend where available). Standardised features, 80/20 hold-out,
  live loss trace, and held-out AUC / F1 (or R² / RMSE).
- **Explain** — Kernel SHAP (exact coalition enumeration for ≤ 11 features,
  paired kernel sampling above that) under the interventional value function,
  plus Integrated Gradients via autodiff. SHAP summary beeswarm, per-instance
  waterfall with an additivity check, dependence plots with automatic
  interaction detection, and a global importance comparison
  (mean |SHAP| vs mean |IG| vs permutation). Export φ as CSV or the summary as PNG.
- **Diagnose** — deletion / insertion curves against a random-order null,
  the Bhatt et al. faithfulness correlation, the IG completeness gap, and
  SHAP–IG rank agreement.

## Validation

The Kernel SHAP solver reproduces the analytic Shapley values of a linear model
to ~1e-8 and satisfies the efficiency (local-accuracy) constraint exactly. On
the built-in benchmark the pipeline correctly ranks the two strong effects
first and drives the pure-noise feature to the smallest attribution.

## Deploy to GitHub Pages

1. Create a repository and copy every file in this folder to its root
   (keep the `icons/` folder and the `.nojekyll` file — the latter stops
   GitHub from mangling the asset paths).
2. Push, then in **Settings → Pages** set the source to the `main` branch,
   root folder.
3. Open `https://<user>.github.io/<repo>/`. On mobile Chrome/Edge use
   *Add to Home Screen* (or the in-app **Install** button); on iOS Safari use
   *Share → Add to Home Screen*. The app then works offline.

Because the app is entirely static, it also runs from any static host or from
`file://` after the first load.

## Files

| file | purpose |
|------|---------|
| `index.html` | app shell and tab structure |
| `style.css` | instrument-panel design system |
| `app.js` | data handling, tf.js training, SHAP / IG, plotting, diagnostics |
| `manifest.webmanifest` | PWA metadata |
| `sw.js` | service worker (offline cache) |
| `icons/` | app icons, incl. maskable variants |

## Notes on interpretation

Baseline-replacement attribution and its removal-based diagnostics evaluate the
model partly off the data manifold; the deletion/insertion AUCs are therefore
comparative (read against the random null), not absolute. The benchmark
generator exists so the whole pipeline can be checked where ground truth is
known before it is trusted on real data.

## References

Lundberg & Lee (2017); Sundararajan, Taly & Yan (2017); Petsiuk, Das & Saenko
(2018); Bhatt, Weller & Moura (2020); Covert, Lundberg & Lee (2021).

## Licence

MIT.
