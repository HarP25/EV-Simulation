import streamlit as st
from stable_baselines3 import PPO
from ev_env import EVStartupEnv
import pandas as pd
import numpy as np

st.set_page_config(page_title="EV Physical AI Demo", layout="wide")

st.title("🚗 EV Physical AI Control System")
st.markdown("Reinforcement Learning based EV powertrain controller — Phase 3")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Vehicle Conditions")

slope = st.sidebar.slider("Slope (°)", -12.0, 12.0, 4.0, 0.5)
payload = st.sidebar.slider("Payload (kg)", 0, 500, 200, 10)  
weather = st.sidebar.selectbox("Weather", ["Dry", "Rain"])
speed_limit = st.sidebar.number_input(
    "Speed Limit (km/h)", value=90, min_value=60, max_value=400, step=10
)
steps = st.sidebar.number_input(
    "Simulation Steps", min_value=100, max_value=10000, step=10, value=1000
)

st.sidebar.divider()
st.sidebar.header("🏎️ Drive Mode")
drive_mode = st.sidebar.selectbox(
    "Performance Mode",
    ["eco", "normal", "sport", "track", "custom"],
    index=1,
    help=(
        "🍃 ECO — Max range, aggressive regen, coast enabled\n"
        "🚙 NORMAL — Balanced daily driving\n"
        "🏁 SPORT — Sharp throttle, lower regen\n"
        "🏎️ TRACK — No speed limit, max boost, ESC off\n"
        "⚙️ CUSTOM — Set your own regen/throttle below"
    )
)

custom_regen    = 1.0
custom_throttle = 1.0
custom_coast    = False

if drive_mode == "custom":
    st.sidebar.markdown("**Custom Settings**")
    custom_regen    = st.sidebar.slider("Regen Aggression", 0.5, 2.0, 1.0, 0.1)
    custom_throttle = st.sidebar.slider("Throttle Sensitivity", 0.5, 1.5, 1.0, 0.1)
    custom_coast    = st.sidebar.checkbox("Enable Lift & Coast", value=False)

st.sidebar.divider()
st.sidebar.header("🌡️ Starting Conditions")
start_battery_temp = st.sidebar.slider("Battery Temp (°C)", 20.0, 45.0, 32.0, 1.0)
start_soc          = st.sidebar.slider("Starting SoC (%)", 20.0, 100.0, 100.0, 1.0)
start_tire_psi     = st.sidebar.slider("Tire Pressure (PSI)", 26.0, 35.0, 33.0, 0.5)

# ── HEADER METRICS ────────────────────────────────────────────────────────────
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Slope",       f"{slope}°")
col2.metric("Payload",     f"{payload} kg")
col3.metric("Weather",     weather.upper())
col4.metric("Drive Mode",  drive_mode.upper())
col5.metric("Speed Limit", f"{speed_limit} km/h")

MODE_DESCRIPTIONS = {
    "eco":    "🍃 ECO — Maximum range. Regen ×2.0. Coast enabled. Fan 30%.",
    "normal": "🚙 NORMAL — Balanced. Standard regen. Comfort suspension.",
    "sport":  "🏁 SPORT — Sharp throttle. Low regen (glide feel). Rear-biased torque.",
    "track":  "🏎️ TRACK — No speed limit. ESC off. Max boost. 100% fan cooling.",
    "custom": f"⚙️ CUSTOM — Regen ×{custom_regen:.1f}. Throttle ×{custom_throttle:.1f}. Coast: {custom_coast}.",
}
st.info(MODE_DESCRIPTIONS[drive_mode])

# ── RUN ───────────────────────────────────────────────────────────────────────
if st.button("▶️ Run Simulation", type="primary"):
    with st.spinner("Running AI simulation..."):
        env = EVStartupEnv()
        model = PPO.load("ev_startup_phase3")

        obs, _ = env.reset()

        # Apply user conditions
        env.slope_angle   = slope
        env.slope_rad     = np.radians(slope)
        env.extra_payload = float(payload)
        env.mass          = 2100.0 + float(payload)
        env.weather       = weather.lower()
        env.mu            = 0.9 if weather.lower() == "dry" else 0.45
        env.speed_limit   = float(speed_limit)
        env.battery_temp  = start_battery_temp
        env.soc           = start_soc
        env.tire_pressure = start_tire_psi

        # Apply drive mode
        env.drive_mode_performance = drive_mode
        if drive_mode == "custom":
            env.driver_regen_level    = custom_regen
            env.driver_throttle_sens  = custom_throttle
            env.driver_coast_pref     = custom_coast
        env.apply_mode_settings()

        # Freeze traffic light for clean demo
        env.traffic_light_state    = "green"
        env.traffic_light_distance = 999.0
        env.traffic_light_timer    = 0.0
        env.traffic_light_cycle    = {"green": 9999.0, "amber": 3.0, "red": 30.0}
        env.animal_active = False
        env.obstacle_distance = 400.0
        env.obstacle_velocity = 0.0                          
        env.lead_vehicle_speed = float(speed_limit) * 0.88

        obs = env.get_obs()

        data = []

        for i in range(int(steps)):
            action, _ = model.predict(obs, deterministic=True)
            raw_action = float(action[0])

            v = env.speed / 3.6
            if v > 2.0:
                # Approaching obstacle
                if env.obstacle_distance < 150.0:
                    close = (env.speed - env.lead_vehicle_speed) / 3.6
                    if close > 0:
                        req = (close**2) / (2.0 * max(1.0, env.obstacle_distance))
                        raw_action = min(raw_action, -min(0.8, req/5.0))

                # Downhill — always regen
                if env.slope_angle < -0.5:
                    raw_action = min(raw_action, -0.3)

                # Coasting to stop
                if env.speed > 5.0 and raw_action < 0.05:
                    raw_action = -0.15  # Light regen when coasting

                if env.drive_mode_performance == "eco" and env.speed > 20.0:
                    target = env.speed_limit * 0.80
                    if env.speed > target:
                        # Above eco target — always regen lightly
                        raw_action = min(raw_action, -0.15)

                # General coasting regen — any time agent is not accelerating
                if raw_action < 0.05 and env.speed > 10.0:
                    raw_action = -0.10

            action = np.array([np.clip(raw_action, -1.0, 1.0)], dtype=np.float32)

            # Force regen when approaching obstacle or slowing
            if env.obstacle_distance < 120.0 and env.speed > 15.0:
                closing = (env.speed - env.lead_vehicle_speed) / 3.6
                if closing > 0:
                    req_decel = (closing ** 2) / (2.0 * max(1.0, env.obstacle_distance - 10.0))
                    regen_torque = -min(0.8, req_decel / 5.0)
                    raw_action = min(raw_action, regen_torque)

            # Downhill regen — physics demands it
            if env.slope_angle < -0.5 and env.speed > 3.0:
                gravity_decel = abs(env.gravity * np.sin(abs(env.slope_rad)))
                regen_torque = -min(0.9, gravity_decel / 3.0)
                raw_action = min(raw_action, regen_torque)

            # Hill assist for steep uphill
            raw_action = float(action[0])
            v = env.speed / 3.6

            if env.slope_angle > 3.0 and env.speed < 20.0:
                # UPHILL — hill assist, no regen
                gravity_comp = env.mass * env.gravity * np.sin(abs(env.slope_rad))
                speed_factor = max(0.3, 1.0 - (env.speed / 25.0))
                min_needed   = np.clip((gravity_comp * speed_factor * 1.6) / 14000.0, 0.0, 0.95)
                raw_action   = max(raw_action, min_needed)

            elif env.slope_angle < -0.5 and v > 2.0:
                # DOWNHILL — regen proportional to slope, cap speed
                gravity_decel = abs(env.gravity * np.sin(abs(env.slope_rad)))
                # Don't let speed exceed limit on downhill
                if env.speed > env.speed_limit * 0.85:
                    raw_action = min(raw_action, -0.4)   # Strong regen to hold speed
                else:
                    raw_action = min(raw_action, -min(0.6, gravity_decel / 4.0))

            elif v > 2.0:
                # FLAT — smart coasting regen
                if env.obstacle_distance < 150.0:
                    close = (env.speed - env.lead_vehicle_speed) / 3.6
                    if close > 0:
                        req = (close**2) / (2.0 * max(1.0, env.obstacle_distance))
                        raw_action = min(raw_action, -min(0.8, req / 5.0))
                elif raw_action < 0.05 and env.speed > 10.0:
                    raw_action = -0.10
                if env.drive_mode_performance == "eco" and env.speed > env.speed_limit * 0.80:
                    raw_action = min(raw_action, -0.15)

            action = np.array([np.clip(raw_action, -1.0, 1.0)], dtype=np.float32)
            
            obs, reward, done, truncated, _ = env.step(action)
            
            data.append({
                'Step':              i,
                'Speed (km/h)':      round(env.speed, 1),
                'Motor Temp (°C)':   round(env.temp, 1),
                'Battery Temp (°C)': round(env.battery_temp, 1),
                'SoC (%)':           round(env.soc, 2),
                'Cap SoC (kWh)':     round(env.cap_soc_kwh, 3),
                'Regen Ratio':       round(
                    env.total_energy_recovered_kwh /
                    max(0.001, env.total_energy_spent_kwh), 3
                ),
                'Attention':         round(env.attention_level, 2),
                'Lateral G':         round(env.lateral_g, 3),
                'Rollover Risk':     round(env.rollover_risk, 3),
                'Drive Mode':        env.drive_mode_performance,
                'Swerve':            int(env.swerve_active),
                'Override':          env.override_reason or "—",
                'Reward':            round(reward, 2),
            })
            if done or truncated:
                break

        df = pd.DataFrame(data)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.success(f"Simulation Complete — {len(df)} steps survived")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Avg Speed",     f"{df['Speed (km/h)'].mean():.1f} km/h")
    c2.metric("Max Motor Temp",f"{df['Motor Temp (°C)'].max():.1f} °C")
    c3.metric("Max Bat Temp",  f"{df['Battery Temp (°C)'].max():.1f} °C")
    c4.metric("Final SoC",     f"{df['SoC (%)'].iloc[-1]:.1f} %")
    c5.metric("Regen Ratio",   f"{df['Regen Ratio'].iloc[-1]*100:.1f} %")
    c6.metric("Steps Survived",len(df))

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🚀 Speed")
        st.line_chart(df.set_index('Step')['Speed (km/h)'])

        st.subheader("🔋 Battery SoC + Supercap")
        st.line_chart(df.set_index('Step')[['SoC (%)', 'Cap SoC (kWh)']])

        st.subheader("⚡ Regen Recovery Ratio")
        st.line_chart(df.set_index('Step')['Regen Ratio'])

    with col_b:
        st.subheader("🌡️ Thermal Management")
        st.line_chart(df.set_index('Step')[['Motor Temp (°C)', 'Battery Temp (°C)']])

        st.subheader("🎯 Driver Attention")
        st.line_chart(df.set_index('Step')['Attention'])

        st.subheader("⚠️ Rollover Risk")
        st.line_chart(df.set_index('Step')['Rollover Risk'])

    st.divider()
    st.subheader("📋 Raw Data")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Results CSV",
        csv,
        "ev_simulation_results.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("📊 Session Summary")
    st.markdown(f"""
    | Metric | Value |
    |---|---|
    | Drive Mode | **{drive_mode.upper()}** |
    | Slope | **{slope}°** ({np.tan(np.radians(np.clip(slope, -89, 89)))*100:.1f}% grade) |
    | Payload | **{payload} kg** |
    | Weather | **{weather}** |
    | Energy Spent | **{env.total_energy_spent_kwh*1000:.2f} Wh** |
    | Energy Recovered | **{env.total_energy_recovered_kwh*1000:.2f} Wh** |
    | Recovery Rate | **{env.total_energy_recovered_kwh/max(0.001,env.total_energy_spent_kwh)*100:.1f}%** |
    | Peak Motor Temp | **{df['Motor Temp (°C)'].max():.1f} °C** |
    | Peak Battery Temp | **{df['Battery Temp (°C)'].max():.1f} °C** |
    | Min Attention Level | **{df['Attention'].min():.2f}** |
    """)

st.markdown("---")
st.caption("EV Physical AI — Phase 3 Simulation | Built with PPO + Gymnasium | Kerala, India 🇮🇳")
