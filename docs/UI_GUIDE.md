# QuorumWatch demo app: what everything on screen means

Start the app with `python app/server.py` and open http://127.0.0.1:8050. The examples below use the **2022 monsoon season** demo, week of **4 July 2022**.

## The idea behind the screen

The two maps are the two stages of the system:

| | Left map | Right map |
|---|---|---|
| Stage | 1, the Isolation Forest | 2, the bacterial layer |
| Question | Is this district unusual this week, by itself? | Is there a regional outbreak? |
| Uses | Only that district's own data | The district's score plus its neighbours |

The gap between the two maps is our contribution. A district can be dark orange on the left (unusual) but stay pale on the right with no alert, because its neighbours don't agree. That's a single noisy district being ignored. A red outline only appears where several neighbouring districts rise together.

Read the screen as: **left map = evidence, right map = decision, timeline = the story over time.**

## Top bar

- **Explore / Results:** switch between the live pipeline on real data and the simulation results.
- **Theme button:** light or dark mode.
- **Data menu:** the five Karnataka example periods.
- **Upload CSV:** your own file with columns `date`, `facility_id`, `cases`. Weekly district files need 14 warm-up weeks; leave cases blank for a missing report. You can also drag a file onto the page.
- **First alert (top right):** the first week any district raised a regional alert in this file, and which districts. For 2022 that's 4 Jul 2022: Chikkamagaluru and Chitradurga.
- **"2 later alerts":** two more separate alert episodes later in the file. Alerts that start within 2 weeks of each other count as one episode.

## Left map: Isolation Forest score

**What it is:** how unusual this district's dengue count is this week, compared with its own normal weeks.

**How it's calculated:**

1. The district's count is turned into a z-score. This compares it with the count expected for that district at that time of year, so a small district and Bengaluru sit on the same scale.
2. From the z-score, 9 features are built: this week, recent 7-week average, spread, trend, a longer-window test, season and weekday.
3. The Isolation Forest scores these features. It was trained only on earlier normal (non-epidemic) years.
4. The score is calibrated against that district's own calibration year: **P = ln(0.5 / p)**, where p is the share of normal weeks that scored at least this high.

**Scale:** from 0 up to about 3.3.

| P | Meaning | Colour |
|---|---|---|
| 0 | Normal (the lower half of normal weeks) | Pale |
| 1.6 | Top 10% | |
| 2.3 | Top 5%. This is the "unusual" cut-off. | |
| 3.3 | Higher than every week in the calibration year | Darkest |

The scale stops at 3.3 because each district has only 52 calibration weeks.

**Colours:**

- **Pale to dark orange:** the score.
- **Grey:** no report from that district this week.
- **Red outline:** the district is in regional alert. This is copied from the right map so the two can be compared.

**Only rises count.** An unusually *low* week scores 0.

## Right map: bacterial-layer signal

**What it is:** each district's signal level A, a running "evidence level". Each week:

- **It goes up** by the district's forest score multiplied by its reliability.
- **It goes up more** through positive feedback once it's already high.
- **It moves along the grey lines** between neighbouring districts, from higher to lower.
- **It goes down** by about half each week (decay).

**The formula:** dA/dt = r·P + β·H(A) + D·Σ w(Aⱼ − A) − δ·A, with β = 1, D = 0.8, δ = 0.5. Each grey line links a district to one of its 4 nearest districts.

**Scale:** 0 up to the number at the right of the colour bar (6.6 in the 2022 demo). That top value is set per file: the highest signal in that file, but at least 1.6 times the activation level.

**Activation level:** the black tick on the colour bar, at A ≈ **4.1**. It comes from the Hill function H(A) = Aⁿ/(Kⁿ + Aⁿ) with K = 5 and n = 2.2. The district switches on when H(A) ≥ 0.393, which happens at A ≈ 4.1.

**Outlines:**

- **Dashed outline (Activated):** the signal has crossed 4.1 and the district is "on". It stays on until the signal drops below about **2.9** (hysteresis), so it doesn't flicker on and off.
- **Red outline (Alert, quorum):** the district is activated, and at least 25% of its neighbourhood (itself plus its neighbours, and at least 2 districts) is also activated.
- **Thick dark outline:** the district you've selected.

## The states

| State | Meaning |
|---|---|
| **Quiet** | Reported this week; signal below the activation level |
| **No report** | No data from this district this week. Its signal can still rise or fall through its neighbours. |
| **Activated** | Signal has crossed about 4.1 and hasn't yet dropped below about 2.9, but not enough neighbours agree yet |
| **Alert** | Activated, and the quorum is met: a regional outbreak signal |

An unusual forest score is not a state. It's evidence that feeds the signal, and the system can ignore it.

## Timeline (the whole file, one bar per week)

Click any week, or use the ← and → keys, to update both maps and the side panels. The vertical line marks the selected week.

1. **Reported cases (all districts):** total dengue cases reported across Karnataka each week. The number at the top left (1,601) is the scale maximum. A gap in the line means no district reported that week. This row is context: the real epidemic curve.
2. **Unusual score, districts (top 5%):** how many of the 30 districts had a forest score of at least 2.3 that week. In a normal year you'd expect about 5% of districts, so 1–2 a week. A value of **11** means 11 of 30 districts were unusual at once, far above chance.
3. **Activated · alert, districts:**
    - **Blue bar:** number of districts activated.
    - **Red bar:** number in regional alert, drawn on top, so alerting districts are always a subset of activated ones.
    - Visible blue above the red means districts that are activated but don't yet have a quorum.

## Week panel (top right)

- **Reported dengue cases (30/30 reporting):** the statewide total that week (453 on 4 Jul 2022), and how many districts sent a report.
- **Districts with an unusual forest score:** the count from timeline row 2 (11). It is a count of districts out of 30, not a score.
- **Districts activated:** districts whose signal is above the activation level (2).
- **Districts in regional alert:** activated districts with a quorum (2).
- **Chips:** the districts in alert, then the districts that are only activated. Click one to select it.

## District panel (bottom right)

Example: Chikkamagaluru.

- **State:** one of the four states above.
- **Dengue cases this week:** 13. The district's raw reported count.
- **Typical:** 1.8. The district's average weekly cases over its training years, for comparison.
- **Forest score:** 3.18 (top 2%). The P value shown on the left map. "Top 2%" means only about 2% of this district's normal weeks scored this high, calculated as 0.5 × e^(−P).
- **Reliability:** 0.99. How much the district's score is trusted, on a scale of 0.05 to 1.
    - It is timeliness (share of reports on time, late reports getting half credit) × (1 − 10 × the rate of implausible reports).
    - An implausible report is a huge spike (z > 4) that neither the neighbours nor the district's own previous weeks back up.
    - Unreliable districts send weaker signal. This is the "quorum quenching" part of the model.
- **Signal:** 4.32 / 4.1 to activate. Its current signal against the activation level.
- **Three small charts** (the whole period, same weeks as the timeline):
    - **Dengue cases (bars):** "max" is the highest week.
    - **Isolation Forest score:** the left-map value over time.
    - **Bacterial-layer signal:** the dashed line is the activation level (4.1).
    - **Pink shading:** weeks this district was in alert.
    - **Vertical line:** the selected week.
    - Click inside a chart to jump to that week.

## How to show it works, using the screen

1. **Noise gets filtered.** Click weeks in Feb–May 2022. A few districts flash orange on the left, but the right map stays pale and nothing is outlined. One district looking odd isn't enough, which is exactly what a single noisy clinic or a reporting dump looks like.
2. **The trigger is agreement.** On 4 Jul, 11 districts are unusual but only 2 alert. Chikkamagaluru and Chitradurga, which are neighbours, both score high. Their signal adds up through diffusion and feedback, crosses 4.1, and the quorum is met. Point at the grey links between them.
3. **It grows like an outbreak.** Step forward week by week. The red alert bars grow from 2 to about 14 districts by Sep–Oct, following the rise in reported cases. The alert spreads to neighbouring districts rather than jumping randomly.
4. **The district story.** Click Chikkamagaluru. Its forest score jumps in June, its signal climbs and crosses the dashed line, and the pink alert band starts.
5. **The control case.** Switch to "2018, a quiet year". The timeline has no red at all.
6. **Then the evidence.** Open Results: in 300 simulated test runs where the true outbreaks are known, the same mechanism detected 0.91 of outbreaks vs 0.75 without the bacterial layer.

**The point to make:** the bacterial layer turns many local, noisy anomaly scores into few, spatially coherent alerts. It alerts when neighbouring districts rise together and ignores isolated spikes.

**What not to claim from the maps:** the Karnataka examples were chosen as illustrations, so they show *how* the system behaves, not *how accurate* it is. The accuracy claim comes only from the Results tab.

## Results tab (simulation)

- **Tiles:**
    - Detection: 0.91 vs 0.75 at an equal 2 false alerts a year.
    - Mean delay: 11.4 vs 16.0 days.
    - Paired gain: +0.15 detection (95% CI 0.11–0.20) and 4.5 days sooner.
    - Message cost: 439 vs 135 messages a day.
- **Detection vs false alerts:** each dot is one alert threshold. Higher and further left is better. Blue (with the layer) is above orange (without) everywhere; the dashed line marks the 2-per-year operating point.
- **Robustness to unreliable reporting:** dots are estimates and whiskers are 95% confidence intervals. The gap widens as reporting gets worse: 0.91 vs 0.69 when half the sites are unreliable.
- **By outbreak type:** each row is one kind of outbreak; orange is without the layer, blue is with it. The longer the connecting line, the bigger the gain. Gains are largest for weak, small, short or slow outbreaks.
- **Replication on a second geography:** the same study on a second, independently generated network reaches the same conclusion (0.94 vs 0.81).
- **The bacterial layer:** the signal equation and the selected parameter values.
