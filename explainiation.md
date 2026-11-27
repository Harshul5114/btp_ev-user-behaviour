# Explaining the EV charging project after trip chains

This walkthrough starts after the NHTS trip chains have been built. It explains the charging simulation, the machine-learning experiments, what the results mean, and how to present them.

## 1. How the work fits together

The project asks two related questions.

- **Charging simulation.** Given a vehicle’s travel and parking schedule, how much power might it draw if it were an EV?
- **Behaviour prediction.** Given a driver’s past charging sessions, can we predict how long they will stay plugged in, how much energy they will take, or when they will next plug in?

The behaviour prediction experiments are the main research result. The simulation shows how travel and charging assumptions can become an electricity demand curve.

```text
NHTS travel reports → Trip chains → Assumed EV and charging rules → Simulated demand

Residential charging records → User history → Behaviour predictions → Evaluate errors
                                                       ↓
                                      Future test of aggregate demand forecasts
```

## 2. From a trip chain to charging demand

A trip chain shows when a vehicle drove, how far it went and how long it was parked. The simulator assigns each selected vehicle a hypothetical battery and charger.

1. **Driving uses energy.** A 30 km trip at an assumed 0.18 kWh/km uses **5.4 kWh**.
2. **Parking at home gives the vehicle an opportunity to charge.** A 7.4 kW charger running at full power for an hour draws **7.4 kWh** from the grid. At 90% efficiency, about **6.66 kWh** reaches the battery.
3. **The battery level changes through the day.** State of charge, or **SOC**, is how full the battery is. The prototype assumes a 60 kWh battery that starts at 80% SOC and charges until it reaches 90% or the vehicle departs.
4. **We add the vehicles’ power.** For each 15-minute interval, calculate each vehicle’s average grid power and sum it across the sample.

NHTS covers one travel day. If a vehicle finishes at home, the overnight scenario assumes it leaves the following morning at the same clock time as its first departure on the survey day. The original prototype passed energy, SOC and charging checks for **6,136 feasible vehicle-days**.

> **What to say with the SOC graph:** “The battery level falls when the vehicle drives and rises when it charges during a home parking window. The trips come from the survey. The battery, charger and charging rules are modelling assumptions.”

## 3. What data we used

| Dataset | What it contains | Role in the project |
|---|---|---|
| 2022 US National Household Travel Survey (NHTS) | Reported trips, distances, destinations and vehicles | Reconstruct travel schedules and simulate charging demand |
| Norway residential charging | 35,377 recorded sessions from 267 user IDs at 12 sites | Main benchmark for predicting charging timing and energy |
| Electric Nation, UK | 65,126 usable CrowdCharge sessions from 299 participants after filtering | Repeat the connection-duration experiment on an independent residential cohort |
| Calibrated synthetic data | 432,000 generated sessions across nine datasets | Compare prediction methods under controlled patterns of driver behaviour |

The US survey and the European charging datasets describe separate populations. The audited models learn behaviour from charging records. NHTS supplies the trip chains for the engineering simulation. The synthetic generator uses the **training portions** of the Norway and UK data to set plausible behaviour patterns.

## 4. What machine learning means in this project

Imagine a driver plugs in at 19:00 on Tuesday. We know the time, weekday, charging site and the driver’s completed sessions. If they eventually unplug at 07:00, the answer is **12 hours connected**. The model sees many historical examples like this during training. For a test session, it gets the facts available at the forecast time and makes a prediction. We then compare that prediction with what happened.

| Question | When the forecast is made | What the model predicts |
|---|---|---|
| How long will the vehicle stay plugged in? | At plug-in | Hours until unplugging |
| How much energy will this session deliver? | At plug-in | Delivered energy in kWh |
| When will this user next plug in? | At the current session’s unplug time | Hours until the next recorded plug-in |

**Connection time** includes any idle time while the vehicle is still plugged in. **Next recorded plug-in** means the next session visible in the dataset. A driver may charge somewhere else in between.

## 5. How we evaluated the models fairly

### Forecast with information available at the time

If a session starts at 18:00 but finishes the next morning, its final energy and duration can enter the history only after it finishes. The gap between two sessions becomes known when the second session starts.

The earlier code sometimes used those outcomes too soon. This is called **data leakage**. It gives a model information it would lack in real use. The audited benchmark fixes when those outcomes become available and removes training labels still unknown at the cutoff date.

### Ask two different questions

- **Can we forecast later sessions?** Train on earlier sessions and evaluate later ones. Update user histories as the later outcomes become available.
- **Can we forecast for a new driver?** Hold entire users out of training. In the zero-history Norway check, the model gets no personal history. In the similar-user experiments, we reveal exactly **1, 3, 5 or 10 completed sessions** and predict the first eligible session at each checkpoint.

At a given checkpoint, every method predicts the **same cases**. Different drivers can qualify for different checkpoints, so compare methods **within a row**. An improvement from one row to another can also reflect a change in the people being evaluated.

### Understand the score

**MAE** means **mean absolute error**. A 9-hour forecast for a 12-hour connection is off by 3 hours. A 15-hour forecast is also off by 3 hours. MAE averages those errors, so lower is better.

To estimate uncertainty, we repeatedly sampled **whole users**, keeping a user’s sessions together. For each sample we compared both methods on the same records. The connection-duration improvement was about **35 minutes on average**, with a paired 95% interval of about **25–45 minutes**. This is uncertainty around the **average improvement**. An individual prediction can still be wrong by several hours.

## 6. The first comparison: simple rules versus boosting

A **baseline** is a simple prediction rule we use to judge whether the more complex model adds value.

| Method | How it makes a prediction |
|---|---|
| Training median | Use the middle value from the training data for everyone. |
| Calendar median | Use the typical value for the same weekday and hour. |
| Own-history median | Use the middle value from the driver’s completed sessions, with a fallback when history is unavailable. |
| Last outcome | Use the driver’s most recent completed result. |
| Gradient boosting | Combine small decision trees that learn from time, weekday, site and summaries of the driver’s past sessions. |

Gradient boosting is easier to picture as a series of small trees that ask questions such as *“Is the plug-in in the evening?”* Each successive tree helps correct errors left by earlier trees. The final prediction combines their contributions. The audited model uses the count, mean and median of past available outcomes alongside calendar and site information. It is trained to reduce absolute error.

### What happened on later Norway sessions

| Target | Personal-history median MAE | Boosting MAE | Interpretation |
|---|---:|---:|---|
| Connection duration | 8.36 h | **7.78 h** | Clearest gain, about 35 minutes less average error. |
| Next recorded plug-in | 46.03 h | **44.61 h** | The error remains large and the uncertainty interval for the gain includes zero. |
| Delivered energy | 5.64 kWh | **5.52 kWh** | The personal median already explains much of the predictable variation. |

- **Connection duration has a visible routine.** An evening plug-in often lasts into the morning. Clock time and the driver’s usual duration can work together as clues.
- **Long gaps make the next plug-in harder.** A driver’s schedule can change or they can charge elsewhere. On observed gaps longer than seven days, boosting performed worse than the history median. The training median also put a larger share of forecasts within 24 hours overall.
- **Energy varies for reasons we cannot see.** How far the car drove and how full its battery was on arrival could explain some of the remaining variation. The driver’s past median captures much of their typical demand.

These are plausible explanations for the measured performance. The separate zero-history Norway test favoured simple baselines by MAE on all three targets. The 35-minute improvement refers to the later-session evaluation.

## 7. The similar-user experiments

After one session, the driver’s personal median is just that one result. We asked whether people with similar charging habits could supply a better starting estimate.

1. **Profile the driver.** Summarize completed sessions using plug-in and unplug times, typical duration, and the shares of overnight and weekend sessions.
2. **Find five similar training drivers.** Compare profiles built from the same number of completed sessions. Treat clock time as circular, so 23:55 and 00:05 are close.
3. **Estimate duration from the matches.** Each match supplies its typical connection duration for sessions near the current plug-in hour. Closer matches carry more weight.
4. **Combine the information.** The fixed blend gives the driver’s own median a weight of `n / (n + 5)`, where `n` is their completed-session count. After one session, that is one-sixth of the weight. The learned hybrid instead supplies the neighbour estimate, distance and disagreement to boosting as additional inputs.

### Results after one completed session

| Real dataset | Own-history median | Fixed similar-user blend | Original boosting |
|---|---:|---:|---:|
| Norway | 11.34 h | 8.92 h | **8.64 h** |
| Electric Nation | 8.01 h | 6.30 h | **5.23 h** |

The fixed blend helped relative to using one past session alone in both countries. Original boosting showed lower MAE in both one-session comparisons and beat the blend at all four UK checkpoints. The Norway blend narrowly led after three sessions, but its uncertainty interval versus boosting included zero. The calendar median slightly beat boosting at the UK three-session checkpoint.

The learned hybrid showed no consistent gain over boosting with richer personal history alone. Matches from a very short history can be noisy. They can also repeat patterns the model already sees in personal features. The enriched models used 682 matched training examples, while original boosting used 21,168 sessions. Comparing the two enriched models on equal training data isolates the effect of adding similar-user features.

> **What to say with the UK chart:** “Electric Nation repeats the test on an independent residential cohort. The neighbour blend improved on one prior session alone, but boosting beat the blend at all four checkpoints. The UK split also covers a change in trial phase, so the later sessions were collected under somewhat different conditions.”

## 8. The synthetic benchmark

**Synthetic data means simulated charging sessions.** The generator uses patterns from genuine Norway and UK training sessions to produce plausible schedules for simulated drivers. The separate real held-out sessions remain the basis for checking real-world performance.

The question was whether the ranking changes when simulated drivers share routines, follow individual habits, or behave irregularly.

| Scenario | How sessions are sampled |
|---|---|
| Shared routines | 85% from the driver’s assigned group of similar reference drivers, 10% from one reference driver and 5% from the wider source. |
| Individual routines | 10% from the group, 85% from one driver and 5% from the wider source. |
| Irregular routines | 10% from the group, 10% from one driver and 80% from the wider source, with more clock-time variation. |

The generator samples plug-in time, duration and energy from the calibrated patterns, adds random variation and prevents overlapping sessions. A **random seed** lets us reproduce a simulation and repeat it with a different set of random choices.

- Three scenarios × three seeds × 1,000 drivers × 48 sessions = **432,000 sessions**.
- Each dataset holds out **200 drivers** and scores four history checkpoints per driver.
- We compared **eight methods** on **7,200 held-out forecasts** in total.

Original boosting had the lowest average MAE in **8 of 12** scenario and history combinations. The fixed neighbour blend reduced error by **2.35–3.09 hours** versus the own-history median after one session. Original boosting still beat the blend in those comparisons. The calendar median narrowly led in the irregular scenario at one and three prior sessions. Adding neighbour features to boosting showed no consistent gain.

> **What to say with the synthetic graph:** “The simulation lets us control how much drivers share a routine. It tests whether the model ranking changes with those assumptions. Norway and Electric Nation provide the separate real-data checks.”

## 9. What comes next

1. **Predict connected vehicles over time.** Use predicted unplug times to estimate how many vehicles remain connected at a given time, and compare against actual connection intervals.
2. **Estimate power from behaviour.** Combine timing and energy with measured power or an explicit assumption about when and how fast a connected vehicle draws electricity.
3. **Evaluate feeder impacts.** Add a network model, baseline household demand and a charger-to-bus mapping. Then evaluate voltage and losses against a no-EV baseline.

Plugged-in vehicle counts, charging power and feeder impact are three different outputs. One does not automatically give you the next.

## Short presentation script

> “After reconstructing the trip chains, we built a simulator that turns trip distance into battery energy and uses home parking windows to estimate charging opportunities. To learn actual charging behaviour, we used repeated residential charging records. We predicted connection duration, delivered energy, and time until the next recorded plug-in.
>
> “We compared gradient boosting with calendar and user-history rules, using only information available when each forecast was made. Connection duration gave the clearest result. Boosting reduced average error by about 35 minutes against the user-history median in Norway.
>
> “Matching similar drivers helped when a new driver had only one completed session, compared with using that session alone. Original boosting remained the strongest overall benchmark across the real UK comparison and the synthetic scenarios. Our next step is to see whether better unplug forecasts also improve predictions of how many vehicles remain connected over time.”

### Three numbers to remember

- **35 minutes** less average connection-duration error in the Norway chronological benchmark.
- **65,126 usable sessions** in the independent UK replication.
- **432,000 generated sessions** in the controlled synthetic benchmark.

### Project references

- [Trip-chain schema](docs/methodology/vehicle_trip_chain_schema.md)
- [Charging prototype validation](docs/methodology/prototype_validation.md)
- [ML audit and main results](docs/ML_AUDIT_RESULTS.md)
- [Similar-user comparison](docs/ML_SIMILAR_USERS.md)
- [Learned hybrid comparison](docs/ML_SIMILAR_USERS_HYBRID.md)
- [Electric Nation intake and results](docs/methodology/electric_nation_intake.md)
- [Synthetic methods and rankings](docs/ML_SYNTHETIC_BENCHMARK.md)
