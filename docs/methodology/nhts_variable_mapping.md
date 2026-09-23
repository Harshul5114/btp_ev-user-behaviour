# 2022 NHTS variable mapping for mobility exploration

## Sources and scope

This mapping is based on the official **2022 NextGen NHTS User Guide** (pp. 22, 29, 32) and **Public Use Codebook v2.0.1** (December 2024; household pp. 1–7, person pp. 8–24, vehicle pp. 25–33, trip pp. 34–47), both stored in `data/raw/nhts_2022/docs/`.

It covers the four day-travel tables requested for the current phase. `ldtv2pub.csv` is the separate long-distance file and is not used here.

| Project concept | NHTS variable | Source table | Units/codes | Notes |
|---|---|---|---|---|
| Household identifier | `HOUSEID` | Household | Character, length 10; official label: unique household identifier | Primary key of Household; shared by Person, Vehicle, and Trip. |
| Person identifier | `HOUSEID` + `PERSONID` | Person | `PERSONID` is character length 2 | Composite person key. The User Guide directs joins between Person and Trip on these two fields. |
| Vehicle identifier | `VEHCASEID` | Vehicle | Character, length 12; official label: unique vehicle identifier | Vehicle primary key. It is also present on applicable Trip records. |
| Vehicle roster position | `HOUSEID` + `VEHID` | Vehicle | `VEHID` is vehicle ID within household | Composite vehicle roster key; `VEHCASEID` is preferred where available. |
| Trip-record identifier | `TDCASEID` | Trip | Character, length 14; unique identifier for every trip record | Unique **person-trip record**, not a vehicle-movement identifier. |
| Person trip sequence | `HOUSEID` + `PERSONID` + `SEQ_TRIPID` | Trip | `SEQ_TRIPID`: renumbered sequential trip ID | Appropriate ordering/key candidate within a person’s reported travel day; User Guide warns ID variables are not always sequential before this renumbering. |
| Original trip ID | `TRIPID` | Trip | Character length 2 | Retain for traceability; do not assume its numeric-looking code is a reliable chronology. |
| Household vehicle used | `TRPHHVEH` | Trip | `01` yes, used household vehicle; `02` no; `-1` valid skip | Initial scope flag for vehicle travel. |
| Vehicle used on trip | `VEHCASEID`, `VEHID` | Trip | `-1` appropriate/valid skip when not available | Join to Vehicle with `VEHCASEID`; in inspected data, populated IDs matched the vehicle roster. |
| Driver/passenger status | `DRVR_FLG`, `PSGR_FLG` | Trip | `DRVR_FLG`: `01` driver, `02` not driver, `-1` valid skip; `PSGR_FLG`: `01` passenger, `02` not passenger | Use the derived `DRVR_FLG` for a candidate driver-operated vehicle trajectory. `WHODROVE` / `WHODROVE_IMP` identify the reported household-person driver or non-household driver. |
| Person’s driver status | `DRIVER` | Person | `01` yes, `02` no, `-1` valid skip | Official codebook label: “Driver status, derived.” A person characteristic; not evidence that the person drove a specific trip. |
| Trip start time | `STRTTIME` | Trip | Local 24-hour time, `0000`–`2359`; `-9` not ascertained | Sort candidate chain records by this field after handling `-9`. |
| Trip end time | `ENDTIME` | Trip | Local 24-hour time, `0000`–`2359`; `-9` not ascertained | End time; retain records where end precedes start for review rather than dropping them. |
| Travel time | `TRVLCMIN` | Trip | Minutes, 0–1425; `-9` not ascertained | Official codebook label: trip duration in minutes. |
| Destination dwell time | `DWELTIME` | Trip | Minutes, 0–1050; `-9` not ascertained | Time at destination; useful later for parking analysis, but missing for many records. |
| Trip distance | `TRPMILES` | Trip | Calculated miles; `-9` not ascertained | Keep decimal values; no conversion is performed in this phase. |
| Vehicle miles traveled field | `VMT_MILE` | Trip | Miles; `-1` appropriate skip, `-9` not ascertained | Only applicable to a subset; distinguish from `TRPMILES`. |
| Origin activity/purpose | `WHYFROM` | Trip | Activity code; `-9` not ascertained | Official label: origin activity. For all but the first recorded trip it is derived from preceding `WHYTO` (User Guide p. 25). |
| Destination activity/purpose | `WHYTO` | Trip | Activity code | Official label: destination activity. Use for home/work/other definitions only after retaining the official code labels. |
| Summary purpose | `WHYTRP1S`, `TRIPPURP`, `WHYTRP90` | Trip | Categorical derived purpose summaries | `WHYTRP1S` includes Home, Work, School/Daycare/Religious, Shopping/Errands, etc. Keep original `WHYTO` alongside any summary. |
| Travel mode | `TRPTRANS` | Trip | `01` car, `02` van, `03` SUV/crossover, `04` pickup, …, `20` walked, `21` other | Official codebook calls this “Trip mode, derived.” Use together with `TRPHHVEH`, `VEHCASEID`, and driver flag. |
| Undocumented raw mode field | `TRIPMODE` | Trip | Values exist in CSV | The supplied Codebook v2.0.1 does **not** contain a `TRIPMODE` entry. Do not assign meanings to its values until an official clarification is obtained; use documented `TRPTRANS` meanwhile. |
| Day of week | `TRAVDAY` | Household, Person, Vehicle, Trip | `01` Sunday through `07` Saturday | Travel-day day of week. |
| Travel-day date field | `TDAYDATE` | Household, Person, Vehicle, Trip | Character length 6, values such as `202202` | Official label: “Date of travel day (YYYYMM)”; public file supplies month, not a full calendar day. |
| Weekend flag | `TDWKND` | Trip | `01` weekend, `02` weekday | Trip-level convenience field. |
| Household vehicle count | `HHVEHCNT` | Household | Count | Useful household context, not a vehicle identifier. |
| Vehicle fuel | `VEHFUEL` | Vehicle | `04` plug-in hybrid; `05` electric only; other documented fuel codes | Observed vehicle characteristic. No EV model is implied by identifying it. |
| Vehicle characteristics | `VEHTYPE`, `VEHYEAR`, `MAKE`, `HYBRID`, `ANNMILES`, `VEHOWNED`, `WHOMAIN` | Vehicle | Vehicle type / model year / codes per codebook | Context for later descriptive mobility analysis; `WHOMAIN` is main driver code, not a trip-level driver flag. |
| Household weight | `WTHHFIN`, `WTHHFIN2D`, `WTHHFIN5D` | Household | 7-, 2-, and 5-day national household weights | Choose a weight only when making weighted estimates; do not sum across table levels. |
| Person weight | `WTPERFIN`, `WTPERFIN2D`, `WTPERFIN5D` | Person | 7-, 2-, and 5-day national person weights | Appropriate for person-level estimates. |
| Trip weight | `WTTRDFIN`, `WTTRDFIN2D`, `WTTRDFIN5D` | Trip | 7-, 2-, and 5-day national trip weights | User Guide §7.2 explicitly uses trip weights for trip/mileage estimates, including driver-reported privately operated vehicle trips. Use 7-day for all-day annualized results, 5-day for weekday results, and 2-day for weekend results. |

## Coding and data-use notes

- The codebook uses negative numeric codes as **valid coded responses**, not conventional missing values. Common meanings are `-1` appropriate/valid skip, `-7` refusal, `-8` don’t know, and `-9` not ascertained; applicability is variable-specific and must be checked in the codebook before recoding.
- Load ID and categorical fields as strings to preserve leading zeroes and special codes.
- The User Guide (p. 29) states: every TRIP-file record is one **person trip**. A family travelling together yields multiple person trips. Therefore, do not sum all `TRPMILES` in the Trip table as vehicle miles.
- The User Guide (p. 26) defines a vehicle trip using `DRVR_FLG == "01"` and `TRPTRANS` in `{01, 02, 03, 04, 06, 07}`. For an identified household-vehicle trajectory, also require `TRPHHVEH == "01"` and a valid `VEHCASEID`. The survey travel day runs 04:00 to 04:00 the following calendar day (User Guide p. 4); shift 00:00–03:59 clock times to the end of that survey day when ordering trips. Same-vehicle simultaneous driver records need explicit review before deduplication.
- `DWELTIME` is labelled "Time at Destination (minutes)" in the codebook (p. 35). It describes a person-trip destination interval; it is not by itself a measured household-vehicle parking interval.
- There is one reported travel day per sampled household context, but no exact calendar day in `TDAYDATE`; the public file offers travel month and day of week.
