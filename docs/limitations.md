# MotionGuard AI - Scientific & Practical Limitations

Academic and deployment integrity requires honest reporting of what the dataset and model can and cannot achieve.

## 1. No Medical or Clinical Safety Certification
- **Absence of Real Fall Incidents:** The KU-HAR benchmark was gathered from 90 volunteers executing prescribed daily living and athletic protocols in controlled environments. It does NOT contain genuine medical emergencies, syncopal falls, stroke events, or elderly frailty falls.
- **Do Not Claim Fall Detection:** MotionGuard AI must NOT be deployed as a certified medical fall alarm or emergency response trigger without independent clinical trials on target patient populations.

## 2. Sensor Placement and Orientation Sensitivity
- Sensors in KU-HAR were placed inside trousers pockets. Sensor orientation inversions (e.g. phone upside down or sideways) can alter accelerometer polarity ($a_y$ and $a_z$).
- While magnitude features ($\sqrt{a_x^2 + a_y^2 + a_z^2}$) are orientation-invariant, individual axis projections will shift.

## 3. Generalization to Unseen Demographics
- 75 of 90 participants were male (83.3% male bias), and all subjects were young university participants.
- The dataset does not capture gait abnormalities, pediatric movement patterns, or assistive device use (wheelchairs, crutches, canes).

## 4. Sampling Jitter and Device Heterogeneity
- KU-HAR was resampled/interpolated to 100 Hz. In real-world IoT and smartwatches, variable Bluetooth latencies and hardware clock drift may produce irregular timestamps.

## 5. Statistical Anomaly vs Dangerous Event
- An unfamiliar pattern flag indicates that the motion signature deviates from the 18 training activities. It does NOT imply that the user is injured, in danger, or experiencing an emergency.
