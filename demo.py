import streamlit as st
from stable_baselines3 import PPO
from ev_env import EVStartupEnv
import pandas as pd
import numpy as np

st.set_page_config(page_title= "EV Physical AI Demo",
                   layout= "wide")

st.title("🚗 EV Physical AI Control System")
st.markdown("Reinforcement Learning based EV powertrain controller")

st.sidebar.header("Vehicle Conditions")

slope = st.sidebar.slider("Slope (°)", 0.0, 12.0, 4.0, 0.5)
payload = st.sidebar.slider("Payload (kg)", 0, 500, 200, 10)
weather = st.sidebar.selectbox("Weather", ["Dry", "Rain"])
speed_limit = st.sidebar.number_input(
    "Speed Limit (km/h)", value= 90, min_value= 60, max_value= 120, step= 10
)

steps = st.sidebar.number_input("Simulation Steps",
                                min_value= 100, max_value= 1000, step= 10)

st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Slope", f"{slope}°")
with col2:
    st.metric("Payload", f"{payload} kg")
with col3:
    st.metric("Weather", weather.upper())

if st.button("Run Simulation", type= "primary"):
    with st.spinner("Running AI simulation..."):
        env = EVStartupEnv()
        model = PPO.load("ev_startup_v1", env=env)

        obs, _ = env.reset()
        env.slope_angle = slope
        env.slope_rad = np.radians(slope)
        env.extra_payload = payload
        env.mass = 2100.0 + payload
        env.weather = weather
        env.mu = 0.9 if weather == "dry" else 0.45
        env.speed_limit = float(speed_limit)
        obs = env.get_obs()

        data = []
        for i in range(steps):
            action, _ = model.predict(obs, deterministic= True)
            obs, reward, done, truncated, info = env.step(action)
            data.append({
                'Step': i,
                'Speed (km/h)': round(env.speed, 1),
                'Motor Temp (°C)': round(env.temp, 1),
                'Battery Temp (°C)': round(env.battery_temp, 1),
                'SoC (%)': round(env.soc, 2),
                'Attention': round(env.attention_level, 2),
                'Reward': round(reward, 2)
            })
            if done or truncated:
                break

        df = pd.DataFrame(data)
    
    st.success(f"Simulation Complete — {len(df)} steps")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Speed", f"{df['Speed (km/h)'].mean():.1f} km/h")
    col2.metric("Max Temp", f"{df['Motor Temp (°C)'].max():.1f} °C")
    col3.metric("Final SoC", f"{df['SoC (%)'].iloc[-1]:.1f} %")
    col4.metric("Steps Survived", len(df))

    st.divider()
    st.subheader("Speed")
    st.line_chart(df.set_index('Step')['Speed (km/h)'])

    st.divider()
    st.subheader("Thermal Management")
    st.line_chart(df.set_index('Step')[
        ['Motor Temp (°C)', 'Battery Temp (°C)']
    ])

    st.divider()
    st.subheader("Battery SOC")
    st.line_chart(df.set_index('Step')['SoC (%)'])

    st.divider()
    st.subheader("Raw Data")
    st.dataframe(df)

    csv = df.to_csv(index= False)
    st.download_button(
        "📥 Download Results CSV",
        csv,
        "ev_simulation_results.csv"
    )

    st.markdown("---")