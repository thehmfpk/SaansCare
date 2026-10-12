# SaansCare
### An Intelligent Environmental Health Monitoring and Public Health Risk Prediction System

**Team SaansCare** — Hafiz Muhammad Faizan · Huma Aslam
Department of Software Engineering, Lahore Garrison University, Lahore, Pakistan

---

## Project Overview & Problem Statement

Lahore is consistently ranked among the most polluted cities in the world — it recorded the highest AQI globally on a single day in November 2024. During the 2024–2025 winter smog season, 98% of outpatients at a Lahore hospital reported at least one pollution-related symptom (shortness of breath 74%, eye irritation 56%, coughing 53%).

Punjab EPA already operates 59 monitoring stations and publishes live AQI readings — the data infrastructure exists. What's missing is a way to turn population-level numbers into individual decisions. Three gaps define the current situation:

1. **No personalization** — a 7-year-old with asthma and a healthy adult in the same neighborhood receive the same AQI figure and the same generic advice.
2. **No pollution-aware navigation** — no app in Pakistan accounts for air quality when suggesting a commute route; people take the shortest or fastest route with no regard for exposure.
3. **Reactive vehicle emission enforcement** — owners act only once smoke is visible or an inspection is due, with no way to check emission health proactively.

The public health response today is reactive: advisories are issued only after conditions become hazardous, schools close only once smog is already heavy, and no tool tells an individual — especially children, asthma or heart patients — whether it is personally safe to go outside.

## Proposed Solution

SaansCare is a role-based web platform powered by **four independently trained machine learning models**, turning real-world air quality and vehicle data into personalized, predictive health and routing decisions:

- **AQI Forecasting Model** — predicts next-day and weekly district-level air quality.
- **Personalized Health Risk Classifier** — combines current AQI with a user's age group and medical condition into a personal risk tier (Low/Moderate/High/Very High) with tailored precautions.
- **Vehicle Emission Classifier & Green Score** — classifies a registered vehicle's emission tier and produces a composite 0–100 Green Score with a downloadable certificate.
- **Route Health Risk Classifier** (primary research contribution) — paired with a pollution-aware route optimizer, classifies a commuter's full journey as Safe, Caution, or Avoid instead of a single raw AQI number.

Training data is drawn from multiple public sources rather than a single feed — Punjab EPA/aqicn.org historical AQI readings, Kaggle and other publicly published air-quality and vehicle-emission datasets, US EPA and UK DVSA vehicle certification data, and OpenStreetMap's Lahore road network. Where no ready-made dataset exists — most notably the route-level exposure data behind the Route Health Classifier — the team builds it directly: real AQI and weather readings are combined with the real Lahore road network and a documented exposure-scoring rule to produce a labelled dataset purpose-built for this project. No arbitrary or fabricated figures are used anywhere in the pipeline.

## Objectives & Expected Outcomes

- Give Lahore residents a personalized, actionable answer to "is it safe for *me* to go outside today?" instead of a generic city-wide number.
- Enable pollution-aware commute planning so residents can choose the lowest-exposure route between two points in the city.
- Give vehicle owners a structured, data-driven view of their vehicle's emission health before a PUC test or inspection.
- Give Punjab EPA and government officials city-wide, district-level analytics (AQI trends, Green Score aggregates, emission-tier breakdowns) to plan smog-season interventions proactively rather than reactively.
- Demonstrate that genuinely trained ML models — built on a mix of public datasets and a purpose-built route dataset, not formula-only heuristics — can deliver real-time, personalized environmental health guidance at city scale.

## Target Users

| User Group | Need |
|---|---|
| **Residents** (~14M in Lahore, 2–3M in high-risk health categories) | Personalized health guidance matched to their own age/condition, not city-wide averages |
| **Vehicle Owners** | An accessible way to assess emission status and maintenance needs before inspection |
| **Punjab EPA & Government Officials** | Predictive, route-level air quality intelligence to design smog-season interventions |
| **System Administrators** | Manage the platform, user records, and model outputs across all portals |

## Key Features / Deliverables

- **Resident Portal** — personal AQI dashboard, health risk score with precautionary guidance, pollution-aware route planner with route health classification, vehicle Green Score registration and certificate download, rule-based maintenance tips, automated alerts.
- **Government / EPA Portal** — city-wide AQI monitoring map, vehicle Green Score aggregates by district and category, emission-tier breakdown, district-level trend analytics.
- **Admin Portal** — user and role management, model output monitoring, system health dashboard.
- **AQI Health Impact Information Module** — documented health effects, at-risk groups, and protective actions per AQI level, sourced from US EPA technical guidance.
- **Multi-trigger Alert System** — district-level AQI threshold alerts, individualized alerts by age/condition, and vehicle alerts (smoke test expiry, overdue service, Green Score drop).

## Implementation Plan & Timeline

Developed using an **Agile methodology** across five phases, with the ML pipeline and application layer proceeding in parallel sprints and integration milestones at the end of each phase.

| Phase | Focus | Timeline |
|---|---|---|
| **1. Planning & Setup** | System architecture, database schema, dataset collection (EPA/aqicn, Kaggle, EPA/DVSA), labelled route dataset groundwork | Month 1 |
| **2. ML Model Development** | Train & evaluate all four models; document evaluation metrics and model comparisons | Months 1–4 |
| **3. Application Development** | Build all three portals; connect the Python model-serving layer to the Node.js backend | Months 3–5 |
| **4. Route Optimizer Integration** | Integrate OSMnx road network + spatial AQI interpolation + modified Dijkstra/A\* routing with the route classifier | Months 4–5 |
| **5. Testing & Finalization** | End-to-end testing, user acceptance testing, performance testing, final documentation | Month 5 |

## Resources or Requirements

**Technology Stack:**

| Layer | Technology |
|---|---|
| Frontend | React.js |
| Backend | Node.js, Express.js |
| Database | MySQL (structured data) + MongoDB (AQI time-series) |
| ML Layer | Python, Scikit-learn, XGBoost, Prophet, TensorFlow — served via FastAPI |
| Routing & Maps | OSMnx, GeoPandas, PostGIS |

**Data Sources:**
- Punjab EPA / aqicn.org — historical and current AQI readings across Lahore's monitoring network
- Kaggle and other public dataset repositories — supplementary air-quality and vehicle-emission datasets
- Open-Meteo API — historical and forecast weather variables (temperature, wind, humidity)
- US EPA — AQI health breakpoint guidelines and vehicle emission certification data
- UK DVSA — vehicle MOT/emission test records
- OpenStreetMap (via OSMnx) — Lahore's road network
- **Purpose-built route dataset** — constructed by the team by combining the above AQI/weather data with the Lahore road graph and a documented exposure-scoring rule, since no ready-made route-level pollution dataset exists for any Pakistani city

## Potential Challenges or Risks

- **No single authoritative data source** — Punjab's own AQI dashboard has no public bulk-export API, so the team draws on multiple complementary sources (aqicn.org, Kaggle, Open-Meteo) and will clearly document which dataset backs each model to avoid any ambiguity about provenance.
- **Class imbalance** in risk/route categories (most readings fall in "safe" ranges) — addressed by evaluating with per-class F1-score and confusion matrices instead of raw accuracy, reading the matrix asymmetrically (under-warning is worse than over-warning).
- **Geographic transfer of vehicle emission data** — US EPA/UK DVSA/Kaggle datasets are used since emission physics is not geography-dependent, but local validation against Pakistan's vehicle fleet remains a known limitation.
- **Building the route dataset from scratch** — since no existing dataset labels Lahore routes by pollution exposure, the team must construct and validate this dataset itself; mitigated by using a transparent, documented exposure-scoring rule rather than arbitrary labels.
- **Real-time performance at scale** — route computation and spatial interpolation must stay under a 10-second response target even under concurrent load; mitigated by pre-computing interpolated AQI surfaces rather than recalculating per request.
- **Model reproducibility** — every prediction is logged with the exact model version used, so results remain traceable as models are retrained.

## Additional Details

SaansCare aligns with three UN Sustainable Development Goals:
- **SDG 3 (Good Health & Well-Being)** — personalized risk guidance reduces pollution-driven health harm for vulnerable groups.
- **SDG 11 (Sustainable Cities & Communities)** — pollution-aware routing and vehicle emission scoring support cleaner urban mobility.
- **SDG 13 (Climate Action)** — emissions and exposure data generated by the platform can inform smog-season mitigation policy.

This project is also being developed as the team's Final Year Project at Lahore Garrison University, which means the models and datasets described above are being built to genuine academic rigor — independently trained, evaluated, and documented — rather than as a one-off hackathon demo.
