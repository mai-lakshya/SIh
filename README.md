What The Project Does

In India, big infrastructure projects (highways, railways, expressways) almost always need to acquire private land first. There's a specific law — the RFCTLARR Act, 2013 — that governs how that acquisition has to happen, and it comes with a hard legal clock: once the first notice is published, the government has 365 days to finish the formal land declaration. Miss that deadline, and the entire legal process lapses — years of paperwork and surveys have to restart from zero. According to the project's own documentation, land acquisition delays like this are responsible for over 70% of the time and cost overruns on India's infrastructure projects, worth well over ₹100 lakh crore in total.

The project is essentially an early-warning and treatment-planning system for these projects. You feed it a project's details (cost, location, land needed, families affected, clearance statuses, etc.), and it:

Diagnoses — how likely is this project to be delayed, and by how much?
Explains — exactly which factors are driving that risk?
Prescribes — what specific actions would reduce the risk, and what would that be worth in time and money saved?

It's built for the people who'd actually use this day-to-day: project directors (e.g., at the National Highways Authority), district land acquisition officers, and financial auditors who need to justify where intervention money goes.

Key Features
Risk & Delay Prediction — Like a weather forecast, but for bureaucracy. You get a delay probability, a risk category (Low/Medium/High), and an estimated number of delay days — with an honest "give or take" range attached (e.g., "280–342 days, 90% confident") rather than a falsely precise single number.
Legal Timeline Forecast — Instead of one number, this shows a full curve of the project's odds of still being on-track at each legal checkpoint: 90 days, 180 days, and the big one — 365 days. It borrows the same math doctors use to estimate patient survival over time, just repurposed here to estimate a project's "survival" against legal deadlines.
Explainable "Why" — Rather than a black-box verdict, it shows exactly which factors are pushing a given project's risk up or down — e.g., "pending forest clearance: +18% risk" or "slow fund disbursement: +12% risk" — as both a plain breakdown and a radar-style chart.
Prescriptive Recommendations — For each risk driver it finds, it suggests a specific, legally-grounded action (e.g., invoking a particular section of the Act), how many delay-days that action would likely save, and a rough financial return on investment for taking it.
What-If Simulator — Lets an official test a hypothetical ("what if the forest clearance gets resolved next month?") and instantly see the revised risk and delay — without touching the real saved record.
Interactive National Map — Plots every project on an accurate map of India (down to district level), color-codes states by aggregate risk, and overlays major infrastructure corridors (like Delhi–Mumbai).
Auto-Reading the Official Land Form — Indian land acquisition uses a standard government form called "Form LA-7." Instead of typing everything by hand, you can upload a filled copy of that form and the system reads and extracts the fields automatically.
AI Advisor — A natural-language assistant that gives advisory commentary on a project, built with explicit safeguards against people trying to manipulate it into ignoring its own rules.
"Remoteness" Scoring — Separately calculates how physically hard-to-reach a project site is, using road networks, terrain, and satellite data from India's space agency (ISRO) — since remote sites tend to run later.
Self-Retraining AI — The model isn't frozen. As new project outcomes get saved, the system checks daily whether real-world data has started drifting away from what it was trained on, and retrains itself in the background if so — but a retrained model only goes live if it passes three separate quality checks first. Otherwise, the trusted older version keeps running.
Save, Track & Report — Analyses can be saved, revisited, and exported as a clean, shareable "risk summary memo."
Model Health Dashboard — A more technical view (for admins/auditors) showing how accurate and current the underlying AI actually is right now — useful for proving the system is trustworthy, not just impressive-looking.
How It Works

The big picture: Picture two parallel diagnostic tracks that feed into one shared "explain and advise" stage:

Track 1 answers: will this be delayed, and by how many days?
Track 2 answers: is this project still on-track at each legal milestone?

The backend (the engine room) is written in Python:

Track 1 combines four different AI models that specialize in spreadsheet-style data (XGBoost, LightGBM, CatBoost, and ExtraTrees), plus a neural-network model for some tasks, and blends their opinions using a technique called stacking — the same idea as a hospital getting four specialists' opinions and having a lead doctor weigh them into one diagnosis.
Track 2 uses "survival analysis" — literally the branch of statistics built for medical survival curves — applied here to legal deadlines instead of patients.
The "why" comes from a well-established technique called SHAP, which mathematically assigns credit or blame to each input factor for a given prediction.
All of it is served over the web using FastAPI (a Python tool for building web APIs), with around 30 distinct endpoints handling everything from predictions to map data to saving records.
The code itself is organized like a factory floor with numbered stations: 01 intake (reading forms/data in) → 02 preprocessing (cleaning it) → 03 models (the AI itself) → 04 explainability & recommendations → 05 orchestration (ties it together) → 06 automatic retraining & monitoring → 07 the API the frontend actually talks to.

The frontend (what you see): A set of web pages — a landing/overview page, the main dashboard ("command center"), an interactive map page, a methodology page explaining the approach, and an explainability page with the risk radar chart — built with HTML/CSS/JavaScript plus a couple of chart components in React. There's also a separate Streamlit app (a Python tool for quick data dashboards) for deeper, more exploratory analysis.

The data: A dataset of roughly 13,600 Indian infrastructure project records — covering cost, land area, number of affected families, dispute rates, clearance statuses, and more — is what the models are trained on.

Storage & deployment: Saved analyses live in a lightweight database (SQLite). The whole system is packaged with Docker (separate containers for the API and the dashboard) with a docker-compose file to run both together consistently anywhere.

User Flow

Here's what actually using it looks like, start to finish, for someone like a project officer:

Arrive & sign in — Land on an overview page showing the national picture (map, headline stats), then log in.
Enter a project — On the main dashboard, either fill in a form (state, district, cost, land area, families affected, clearance statuses, etc.) or upload a scanned "Form LA-7" and let the system auto-fill the fields.
Run the prediction — One click gets you: delay probability, risk tier, estimated delay days with a confidence range, and the full legal-milestone timeline curve.
Understand why — Switch to the explainability view to see exactly which factors are driving the risk, in plain language plus a radar chart.
Explore "what if" — Tweak one input in the simulator to see whether fixing that specific issue would meaningfully change the outlook.
Get a game plan — Check the prescriptive panel for concrete next actions, each with an estimated days-saved and rough cost payoff.
See it on the map — View this project in context alongside every other project nationally, colored by risk and clustered geographically.
Save & share — Save the analysis (which also quietly feeds back into the system's training data) and export a one-page risk summary memo to share with colleagues.
(For admins) Check the AI's health — Someone responsible for the system can open the model-governance view to confirm accuracy hasn't drifted, or manually trigger a retrain.
