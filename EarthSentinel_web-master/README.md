# EarthSentinel — Frontend

EarthSentinel is an interactive frontend dashboard for monitoring landslide and terrain risk.
This repository contains the React-based UI used to visualize risk zones, alerts, and analytics for monitored regions.

The project was bootstrapped with Create React App and uses Tailwind CSS for styling and `lucide-react` for icons.

## Quick start (development)

Recommended Node: 16+ (or the version used by your team). From the project root, run:

```powershell
npm install
npm start
```

Open http://localhost:3000 in your browser. The dev server supports hot reloading.

## Available scripts

- `npm start` — start the dev server
- `npm test` — run tests (Create React App test runner)
- `npm run build` — create an optimized production build in `build/`
- `npm run eject` — eject CRA config (one-way operation)

If your project has additional scripts (linting, format), they will appear in `package.json`.

## Tech stack

- React (Create React App)
- Tailwind CSS (utility-first styling used in `src/App.js`)
- lucide-react (icons)
- SVG for charts/visuals (small inline visualizations in `src/App.js`)

## Project structure (important files)

- `src/` — source code
	- `App.js` — main dashboard component (UI + visualizations)
	- `index.js` — app entry
	- `App.css` / `index.css` — base styles (Tailwind utilities expected)
	- `App.test.js`, `setupTests.js`, `reportWebVitals.js` — default CRA support files
- `public/` — static assets and `index.html`

The UI uses Tailwind utility classes extensively. Make sure your build includes the Tailwind setup if you adjusted the project after bootstrapping.

## Features (from current UI)

- Interactive header with tabs: Overview, Analytics, Alerts
- Mountain silhouette risk visualization with color-graded risk zones
- Multiple metric cards, area charts, and small inline SVG charts
- Responsive grid layout using Tailwind's grid utilities

## Contributing

1. Fork the repo and create a branch for your feature: `git checkout -b feat/your-feature`
2. Make small, focused commits with clear messages
3. Open a pull request describing the changes and any setup steps

If you add dependencies, update `package.json` and include a short rationale in your PR.

## Tests

Run the default CRA tests with:

```powershell
npm test
```

Add unit/UI tests alongside components when possible. The repo currently contains the standard CRA test harness.

## Notes & next steps

- If Tailwind is not yet configured in `postcss`/`craco`/`tailwind.config.js`, add it so utility classes are processed for production.
- Consider adding a `NETLIFY`/`Vercel` deploy section if you want one-click deployment instructions.
- Add a LICENSE file if you want to specify a license (MIT is common for open-source projects).

---

If you'd like, I can:

- add a small CONTRIBUTING.md with a PR checklist
- add a short Tailwind setup guide if your local environment lacks it
- include a preview screenshot and badges (build / license)

Tell me which of the above you'd like next and I will implement it.
