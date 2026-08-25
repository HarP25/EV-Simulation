# EV Physical AI Control System

An RL-based EV powertrain controller that 
generalizes across real-world conditions.

## What It Does
- Manages motor torque across 0-12° slopes,
- Maintains battery temperature in optimal 20-30°C window
- Predictive regenerative braking before obstacles
- Full ADAS suite: AEB, TC, ABS, ESC, ACC, TPMS
- Weather adaptation (rain/dry)
- 19-dimensional observation space

## Results
| Condition | Speed | Temp | Status |
|-----------|-------|------|--------|
| Flat road | 108 km/h | 52°C | ✅ |
| 9° + 194kg | 58 km/h | 82°C | ✅ |
| 12° + 500kg | 30 km/h | 80°C | ✅ |
| Rain + slope | adapts | stable | ✅ |

## The Key Innovation
Reward engineering that eliminates local minima
while maintaining physical realism.

## Run It
pip install gymnasium stable-baselines3
python ev_env.py
