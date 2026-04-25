"""
EQUATIONS

R = (v * w_speed) - (ΔT * w_heat) - (E_loss * w_efficiency)
If the Agent goes fast (v), it gains points. If the temperature (T) spikes or the battery (E) drains too fast, it loses points.
Using Proximal Policy Optimization (PPO)

Calculates Net Force:
F_net = [(τ * G * η) / r] - (F_roll + F_drag + F_gravity)
WHERE 
τ = AI requested torque
G = Fixed gear ratio 
η = 0.9 (Efficiency)
r = 0.33m (Wheel radius)

Jerk:

Jerk = da/dt
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import math
import os

class EVStartupEnv(gym.Env):
    def __init__(self):
        super(EVStartupEnv, self).__init__()
        
        self.extra_payload = np.random.uniform(0.0, 500.0)
        self.slope_angle = np.random.uniform(0.0, 12.0)

        self.mass = 2073.0 + self.extra_payload    
        self.gravity = 9.81
        self.slope_rad = np.radians(self.slope_angle)
        self.wheel_radius = 0.33
        self.cd = 0.22

        self.mu = 0.9   # Grip coefficient for rolling resistance
        self.tire_pressure = np.random.uniform(28.0, 35.0) # PSI, affects rolling resistance
        self.airbag_deployed = False
        self.obstacle_distance = 100.0  

        self.battery_capacity_kwh = 100.0
        self.cap_max_kwh = 3.0      # Supercapacitor max kWh

        self.max_temp = 100.0
        self.ambient_temp = 25.0

        self.acc_enabled = True
        self.acc_target_speed = np.random.choice([60.0, 80.0, 100.0, 120.0])
        self.following_target_distance = self.obstacle_distance

        # ------------------- INITIALIZING VARIABLES -------------------
        self.gear_ratios = {1: 15.0}
        self.current_gear = 1

        self.lateral_g = 0
        self.speed_limit = 90.0
        self.weather = "dry"
        self.pitch_angle = 0.0  # Can affect aero or drag
        self.cell_temp_variance = 0.0    # Variable to simulate uneven cell heating, which can affect BMS behavior

        self.esc_active = 0
        self.adjacent_vehicle_left = np.random.uniform(50.0, 200.0)
        self.adjacent_vehicle_right = np.random.uniform(50.0, 200.0)
        self.blind_spot_warning = False

        self.attention_level = 1.0
        self.attenion_decay_rate = np.random.uniform(0.0001, 0.0005)

        # Action space: Torque request from 0.0 (idle, 0%) to 1.0 (full throttle, 100%)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: [Values to be observed]
        self.observation_space = spaces.Box(low=-1, high= 1, shape= (19,), dtype= np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):   # Seed - required for reproducibility, Options - for future extensions
        super().reset(seed=seed)

        self.tire_pressure = np.random.uniform(28.0, 35.0) # PSI
        self.speed = 0.0
        self.ambient_temp = 35.0  # Optimal starting temp

        self.cap_soc_kwh = 0.0  # Start with empty supercapacitor
        self.soc = 100.0  # Full battery

        self.slope_angle = np.random.uniform(0.0, 12.0)
        self.slope_rad = np.radians(self.slope_angle)

        self.extra_payload = np.random.uniform(0.0, 500.0)
        self.mass = 2100.0 + self.extra_payload

        self.current_gear = 1
        self.last_acceleration = 0.0
        self.airbag_deployed = False
        self.obstacle_distance = np.random.uniform(100.0, 200.0)
        self.stall_steps = 0

        self.temp = self.ambient_temp
        self.battery_temp = np.random.uniform(15.0, 45.0)
        self.attention_level = np.random.uniform(0.6, 1.0)

        self.lateral_g = 0.0
        self.speed_limit = np.random.choice([60.0, 70.0, 80.0, 90.0, 100.0, 120.0])
        self.weather = np.random.choice(["dry", "rain"], p= [0.75, 0.25])
        self.mu = 0.9 if self.weather == "dry" else 0.45
        self.pitch_angle = 0.0
        self.cell_temp_variance = np.random.uniform(0.0, 5.0)
        
        self.history = {
                    'soc': [], 'temp': [], 'speed': [], 'gear': [], 
                    'tc': [], 'abs': [], 'aeb': [], 'tpms_loss': [],
                    'esc': []
                }
        
        return self.get_obs(), {}
    
    def get_obs(self):
        # Noormalization for better learning, we will scale speed to 0-1 
        return np.array([
            self.speed / 210.0,
            self.temp / self.max_temp,  
            self.soc / 100.0,
            self.cap_soc_kwh / self.cap_max_kwh,
            1.0,
            self.get_motor_rpm() / 18000.0,
            min(1.0, self.obstacle_distance / 150.0),
            self.slope_angle / 12.0,
            self.extra_payload / 500.0,
            np.clip(self.lateral_g / 1.0, 0, 1),
            self.speed_limit / 120.0,
            1.0 if self.weather == "dry" else 0.0,
            np.clip(self.cell_temp_variance / 10.0, 0, 1),
            np.clip(abs(self.pitch_angle) / 15.0, 0, 1),
            np.clip(self.last_acceleration / 10.0, -1.0, 1.0),
            (self.battery_temp - 25.0) / 25.0,
            min(1.0, self.adjacent_vehicle_left / 200.0),
            min(1.0, self.adjacent_vehicle_right / 200.0),
            self.attention_level
        ],
        dtype= np.float32)
    
    def get_motor_rpm(self):
        speed_mps = self.speed / 3.6 
        return (speed_mps / (2 * np.pi * self.wheel_radius)) * self.gear_ratios[self.current_gear] * 60     # RPM = (Vehicle Speed / Wheel Circumference ) * Gear Ratio * 60 

    def step(self, action):
        torque_request = action[0]
        reward = 0.0
        dt = 0.1  # Time step in seconds
        tc_active, abs_active, aeb_active = 0, 0, 0
        self.last_speed = getattr(self, 'last_speed', 0.0)
        done = False

        # ------------- SENSORS -------------
        self.obstacle_distance -= (self.speed / 3.6) * dt   
        if self.obstacle_distance < 15.0:
            self.obstacle_distance = np.random.uniform(300.0, 500.0)  # Reset obstacle distance after "passing" it

        # ------------- SINGLE GEAR -------------
        rpm = self.get_motor_rpm()

        # ------------- SAFETY FEATURES -------------

        # AEB - Automatic brakes
        ttc = self.obstacle_distance / (self.speed / 3.6 + 0.00001)
        if ttc < 0.5:
            torque_request = -1.0
            aeb_active = 1
            reward -= 15.0 

        elif ttc < 1.2: 
            torque_request = min(torque_request, -0.8)  # Limit throttle
            aeb_active = 0.8
            reward -= 5.0
        
        elif ttc < 3.0:
            torque_request = min(torque_request, 0.0)  # No throttle, but allow coasting
            reward -= (2.5 - ttc) * 0.1 

        if torque_request >= 0:
            adjusted_request = 0.05 + (torque_request * 0.95)  # Lower minimum and smoother ramp
        
        else:
            adjusted_request = torque_request   

        # BMS - thermal and SoC protection
        bms_limit = 1.0

        if self.temp > 88 or self.soc < 20:  # Gradual derating before hard limit
            bms_limit = 0.3
        if self.temp > 85.0:
            bms_limit *= (1.0 - (self.temp - 85.0) / 15.0)

        if bms_limit < 0.7:
            reward -= 1.0

        if self.soc < 30:
            bms_limit *= (self.soc / 30.0) ** 0.5  

        cell_imbalance_derate = 1.0 - (self.cell_temp_variance / 50.0)     # Cell imbalance derate is 
        bms_limit *= max(0.5, cell_imbalance_derate)    
        
        applied_torque = adjusted_request * bms_limit

        if bms_limit < 0.5:
            reward -= 0.5

        motor_efficiency = 0.96 * np.exp(-0.5 * ((rpm / 7000.0) - 1.0) ** 2)
        motor_efficiency = max(0.60, motor_efficiency)  

        # Boost - reduced for stability
        boost = 1.1 if self.cap_soc_kwh > 1.0 and torque_request > 0.8 else 1.0

        def get_smooth_mult(request):
            base_torque = 250.0 + (100.0 * math.tanh(5 * request))  
            if self.speed < 10:
                rain_factor = 0.6 if self.weather == "rain" else 1.8
                return base_torque * rain_factor
            return base_torque
        
        motor_torque_mult = get_smooth_mult(applied_torque)
        motor_force = (applied_torque * motor_torque_mult * boost * self.gear_ratios[self.current_gear] * motor_efficiency) / self.wheel_radius

        # TC and ABS
        traction_limit = self.mu * self.mass * self.gravity * np.cos(self.slope_rad)  # Traction limit = Grip Coefficient * Mass * Gravity * cos(slope)
        if motor_force > traction_limit:
            motor_force = traction_limit
            tc_active = 1
            reward -= 0.5
        
        elif motor_force < -traction_limit:  # If we brake too hard and exceed traction limit in reverse, activate ABS to prevent lockup
            motor_force = -traction_limit
            abs_active = 1
            reward -= 0.5

        # TPMS
        optimal_pressure = 35.0
        current_pressure = self.tire_pressure

        pressure_loss_pct = max(0, (optimal_pressure - current_pressure) / optimal_pressure) 

        tpms_mult = 1.0 + (pressure_loss_pct * 0.5)  # If pressure is low, we get a multiplier that increases rolling resistance and reduces performance
        rolling_resistance = self.mass * self.gravity * tpms_mult * 0.015 * np.cos(self.slope_rad)

        if pressure_loss_pct > 0:
            efficency_penalty = pressure_loss_pct * (self.speed / 100.0)
            reward -= efficency_penalty

        drag = 0.5 * self.cd * 1.225 * 2.2 * (self.speed / 3.6) ** 2
        gravity_pull = self.mass * self.gravity * np.sin(self.slope_rad)

        acceleration = (motor_force - drag - gravity_pull - rolling_resistance) / self.mass
        self.speed = max(0.0, self.speed + (acceleration * 3.6 * dt))

        # ---------------- LATERAL G AND RIDE COMFORT ----------------
        self.lateral_g = np.random.uniform(0.0, (self.speed / 100.0) * 0.6)
        
        longitudinal_g = abs(acceleration) / self.gravity 
        lateral_comfort_limit = 0.4

        if self.lateral_g > lateral_comfort_limit:
            reward -= (self.lateral_g - lateral_comfort_limit) * 3.0
        
        if longitudinal_g > 0.6:
            reward -= (longitudinal_g - 0.6) 
        
        self.pitch_angle = np.clip(acceleration * 0.3, -15.0, 15.0)     # Pitch angle is the amount the vehicle would tilt forward or backward based on acceleration, which can affect aerodynamics and stability. Capping it at 15 degrees for realism.

        if acceleration < -34.3:
            self.airbag_deployed = True
            reward -= 5000.0

        safe_distance = (self.speed / 3.6) * 2.0  

        if self.obstacle_distance < 10.0:
            reward -= 100.0
            done = True
        
        elif self.obstacle_distance < (safe_distance * 0.5):
            torque_request = min(torque_request, -0.8)  
        
        elif self.obstacle_distance < safe_distance:
            torque_request = min(torque_request, 0.0)
            reward -= 2.0

        # ---------------- ESC - Electronic Stability Control ----------------
        self.esc_active = 0
        stability_threshold = 0.6

        if self.lateral_g > stability_threshold:
            esc_correction = (self.lateral_g - stability_threshold) / 0.4   # ESC Correction = (Lateral G - threshold) / (Max Lateral G - threshold)
            torque_request = torque_request * (1.0 - esc_correction * 0.5)  # Torque request = Torque request * (1 - ESC Correction * 0.5) - reduces torque by up to 50% based on how much we exceed the stability threshold
            self.esc_active = 1
            reward -= esc_correction * 1.5

        # ---------------- HILL START ASSIST ----------------
        if self.slope_angle > 3.0 and self.speed < 0.5 and torque_request > 0:
            min_hold_torque = (self.mass * self.gravity * np.sin(self.slope_rad)) / 5000.0
            adjusted_request = max(adjusted_request, min_hold_torque)
            reward += 0.2

        # ------------- ENERGY AND THERMAL PARTS AND REGEN OPTIMIZATION -------------
        self.battery_temp = self.ambient_temp

        if self.obstacle_distance < 100 and self.speed > 20:
            regen_intensity = np.clip(
                (100 - self.obstacle_distance) / 100.0, -0, 1
                )

            if torque_request < 0 and abs(torque_request) <= regen_intensity:
                reward += regen_intensity * 0.3

        power_watts = motor_force * (self.speed / 3.6)

        if applied_torque < 0:
            regen_efficiency = 0.7 * np.exp(-0.5 * ((self.temp - 30.0) / 40.0) ** 2)
            regen_efficiency = max(0.3, regen_efficiency)

            if self.battery_temp < 20.0:
                regen_efficiency *= 0.5     # 50% regen
            elif self.battery_temp > 35.0:
                regen_efficiency *= 0.8     # 80% regen
            elif 20.0 <= self.battery_temp <= 30.0:
                regen_efficiency *= 1.0     # Full regen

            energy_exchange = (power_watts * dt * regen_efficiency) / 3600000

            space_in_cap = self.cap_max_kwh - self.cap_soc_kwh
            to_cap = min(abs(energy_exchange), space_in_cap)
            self.cap_soc_kwh += to_cap
            energy_exchange += to_cap
        
        else:
            energy_exchange = (power_watts * dt) / (motor_efficiency * 3600000)     # Energy (kWh) = Power (W) * Time (s) / (Efficiency * 3600000) 
            if self.cap_soc_kwh > 0:
                draw = min(energy_exchange, self.cap_soc_kwh)
                self.cap_soc_kwh -= draw
                energy_exchange -= draw

        if applied_torque > 0 and self.cap_soc_kwh > 0:
            draw = min(energy_exchange, self.cap_soc_kwh)
            self.cap_soc_kwh -= draw
            energy_exchange -= draw

        self.soc -= (energy_exchange / self.battery_capacity_kwh) * 100

        heat_gen = (abs(applied_torque) ** 2 * 0.02) + (abs(power_watts) / 15000.0)
        cooling = (self.temp - self.ambient_temp) * (0.05 + 0.0007 * self.speed)
        self.temp += (heat_gen - cooling) * dt

        # -------------- BATTERY THERMAL MANAGEMENT --------------
        optimal_battery_temp_low = 20.0
        optimal_battery_temp_high = 30.0
        self.battery_temp = self.temp * 0.7 + self.ambient_temp * 0.3

        if self.battery_temp < optimal_battery_temp_low:
            battery_efficiency_loss = (optimal_battery_temp_low - self.battery_temp) / 20.0
            reward -= battery_efficiency_loss * 2.0

        elif self.battery_temp > optimal_battery_temp_high:
            battery_efficiency_loss = (optimal_battery_temp_high - self.battery_temp) / 30.0
            reward -= battery_efficiency_loss * 2.0

        else:
            reward += 0.3

        # ---------------- LANE CHANGE MECHANICS ----------------
        self.adjacent_vehicle_left -= (self.speed / 3.6) * dt * np.random.uniform(0.8, 1.2)
        self.adjacent_vehicle_right -= (self.speed / 3.6) * dt * np.random.uniform(0.8, 1.2)

        if self.adjacent_vehicle_left < 20.0:
            self.adjacent_vehicle_left = np.random.uniform(150.0, 300.0)
        if self.adjacent_vehicle_right < 20.00:
            self.adjacent_vehicle_right = np.random.uniform(150.0, 300.0)    

        blind_spot_zone = 15.0

        if self.adjacent_vehicle_left < blind_spot_zone or self.adjacent_vehicle_right < blind_spot_zone:
            self.blind_spot_warning = True
            speed_factor = self.speed / 50.0
            reward -= 1.0 * speed_factor    # Putting this value bcz reward -= 100% of speed / 50.0 [normalized speed]

            if self.adjacent_vehicle_left < 5.0 or self.adjacent_vehicle_right < 5.0:
                reward -= 10.0
                done = True
        
        else:
            self.blind_spot_warning = False
        
         # ------------- REWARD CALCULATION -------------
        target_speed = 80.0 - (self.slope_angle * 3.5) - (self.extra_payload / 100.0)    # Realistic target for speed optimization
        target_speed = max(30.0, target_speed) 
        target_speed = min(target_speed, self.speed_limit * 0.90)

        min_acceptable = target_speed * 0.7

        if self.speed < min_acceptable and self.slope_angle < 8.0:
            reward -= (min_acceptable - self.speed) * 0.1

        if self.speed > 2.0:
            speed_reward = np.exp(-0.5 * ((self.speed - target_speed) / 15.0) ** 2)
        
        else:
            speed_reward = 0.0
            reward -= 0.5
        
        if hasattr(self, 'last_speed'):
            speed_delta = self.speed - self.last_speed

            if self.speed < target_speed and self.speed < self.speed_limit:
                if speed_delta > 0:
                    reward += speed_delta * 0.3
                else:
                    reward -= abs(speed_delta) * 0.2
            
            elif self.speed > target_speed:
                if speed_delta < 0:
                    reward += abs(speed_delta) * 0.2

        if self.acc_enabled and self.speed > 5.0:
            speed_error = abs(self.speed - self.acc_target_speed)
            speed_acc_reward = np.exp(-0.5 * (speed_error / 20.0) ** 2) * 0.8

            distance_error = abs(self.obstacle_distance - self.following_target_distance)
            distance_acc_reward = np.exp(-0.5 * (distance_error / 30.0) ** 2) * 0.5

            speed_change = abs(self.speed - self.last_speed) if hasattr(self, 'last_speed') else 0.0
            comfort_reward = -speed_change * 0.1

            reward += speed_acc_reward + distance_acc_reward + comfort_reward
        
        self.last_speed = self.speed
        
        slope_difficulty = 1.0 + (self.slope_angle / 12.0)
        adj_speed_reward = speed_reward * slope_difficulty
        reward += adj_speed_reward * 2.0

        if self.speed > self.speed_limit:
            excess = self.speed - self.speed_limit
            reward -= excess * 10.0
            reward -= (excess ** 1.5) * 2.0
            if excess > 5.0:
                reward -= 30.0
            if excess > 15.0:
                reward -= 100.0
                done = True 
        
        if self.speed > (self.speed_limit + 20):
            done = True
        
        ideal_ttc = 4.0
        ttc_current = 0.0
        
        if self.speed > 5.0:
            ttc_current = self.obstacle_distance / (self.speed / 3.6 + 0.0001)
            ttc_error = abs(ttc_current - ideal_ttc)
            acc_reward = np.exp(-0.5 * (ttc_error * 2.0) ** 2) * 0.5
            reward += acc_reward
        
        if ttc_current < 2.0 and self.speed > 30.0:
            reward -= (2.0 - ttc_current) * 10.0
        
        if self.weather == "rain" and torque_request > 0.7:
            reward -= (torque_request - 0.7) * 2.0

        if self.speed < 1.0 and abs(torque_request) < 0.15 and acceleration <= 0.0:
            reward -= 1.0

        if self.speed < 2.0 and abs(torque_request) > 0.1 and acceleration < 0:
            if self.slope_angle < 8.0:
                self.stall_steps += 1
                reward -= 1.5

        else:
            self.stall_steps = max(0, self.stall_steps - 2)

        stall_limit = 20 if self.slope_angle < 8.0 else 60
        if self.stall_steps > stall_limit:
            reward -= 20.0
            done = True

        if self.slope_angle > 7.0:
            if torque_request > 0.5:
                reward += 1.5
            elif torque_request < 0.1:
                reward -= 2.0

        jerk = abs(acceleration - self.last_acceleration)
        if jerk > 2.0:
            reward -= (jerk - 2.0) * 0.05

        motor_redline = 14000   # Ideal RPM
        if rpm > motor_redline:
            reward -= (rpm - motor_redline) / 1000.0

            redline_derate = max(0.5, 1.0 - (rpm - motor_redline) / 5000.0)
            motor_force *= redline_derate

        energy_efficiency = 1.0 - (abs(self.soc - 100.0) / 100.0)
        speed_ratio = min(1.0, self.speed / target_speed)
        edge_score = speed_ratio * energy_efficiency

        if edge_score > 0.85:
            reward += edge_score 

        elif edge_score > 0.70:
            reward += edge_score * 0.5

        thermal_efficiency = 1.0
        if self.temp < 60.0:
            thermal_efficiency = 1.2

        elif self.temp < 70.0:
            thermal_efficiency = 1.0
        
        elif self.temp < 80.0:
            thermal_efficiency = 0.8

        else:
            thermal_efficiency = 0.5

        reward += adj_speed_reward * (thermal_efficiency - 1.0)
        
        # ---------------- ATTENTION AND DISTRACTION MECHANICS ----------------
        self.attention_level = max(0.3, self.attention_level - self.attenion_decay_rate) 

        if jerk > 1.5:
            self.attention_level = min(1.0, self.attention_level + 0.1)

        if self.attention_level < 0.5:
            attention_noise = np.random.normal(0, (1.0 - self.attention_level) * 0.2)
            torque_request = np.clip(torque_request + attention_noise, -1, 1)
            reward -= (0.5 - self.attention_level) 

        if self.attention_level > 0.9:
            reward += 0.2

        self.last_acceleration = acceleration

        self.log_history(tc_active, abs_active, aeb_active, tpms_mult)

        done = done or (self.temp > self.max_temp or self.soc < 0 or self.airbag_deployed or self.speed > 170.0)
        return self.get_obs(), reward, done, False, {}
    
    def log_history(self, tc, abs, aeb, tpms):
        self.history['soc'].append(self.soc)
        self.history['temp'].append(self.temp)
        self.history['speed'].append(self.speed)
        self.history['gear'].append(self.current_gear)
        self.history['tc'].append(tc)
        self.history['abs'].append(abs)
        self.history['aeb'].append(aeb)
        self.history['tpms_loss'].append(tpms)
        self.history['esc'].append(self.esc_active)
    
    def plot_performance(self):

        fig, axes = plt.subplots(7, 1, figsize=(12, 18))

        axes[0].plot(self.history['speed'], color='g', label='Speed (km/h)')
        axes[0].axhline(y=self.speed_limit, color='purple', linestyle='--', label='Speed Limit')
        axes[0].set_title("Vehicle Speed")

        axes[1].plot(self.history['temp'], color='r', label='Motor Temp (°C)')
        axes[1].axhline(y=85, color='orange', linestyle='--', label='BMS Derating')
        axes[1].set_title("Thermal")

        axes[2].plot(self.history['soc'], color='blue', label='Battery SoC (%)')
        axes[2].set_title("State of Charge")

        axes[3].plot(self.history['gear'], color='brown', label='Gear')
        axes[3].set_title("Gearbox")

        axes[4].step(range(len(self.history['tc'])), self.history['tc'], label='TC')
        axes[4].step(range(len(self.history['abs'])), self.history['abs'], label='ABS')
        axes[4].step(range(len(self.history['aeb'])), self.history['aeb'], label='AEB', color='red')
        axes[4].set_title("Safety Systems")

        axes[5].plot(self.history['tpms_loss'], label='TPMS Loss Multiplier', color='orange')
        axes[5].set_title("Tire Pressure Loss Impact")

        axes[6].step(range(len(self.history['esc'])), self.history['esc'], label='ESC Active', color='purple')
        axes[6].set_title("Electronic Stability Control Activation")

        for ax in axes:
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

# --- Quick Test ---
if __name__ == "__main__":

    env = EVStartupEnv()

    if os.path.exists("env_startup_v1.zip"):
        os.remove("env_startup_v1.zip")

    model = PPO(
        'MlpPolicy',
        env,
        verbose= 1,
        learning_rate= 0.0001,
        batch_size= 256,
        n_steps= 4096,
        ent_coef= 0.003,
        clip_range= 0.15,
        clip_range_vf= 0.15,
        gae_lambda= 0.95,
        n_epochs= 10,
        seed= 42,
        normalize_advantage= True,
        policy_kwargs= dict(net_arch = [256, 256, 128]),     # net_arch means the number of neurons in each layer of the policy network. Using 3 hidden layers with 256, 256, and 128 neurons respectively
        device= "cpu"
    )

    print("Training Agent for 2 million steps...")
    model.learn(total_timesteps=2000000)
    model.save("ev_startup_v1")

    print("\nTraining Complete! Running Test Drive...")
    obs, info = env.reset()
    for i in range(1000):
        action, states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        if i % 100 == 0:
            current_rpm = env.get_motor_rpm()
            ttc = env.obstacle_distance / (env.speed / 3.6 + 0.00001)
            payload_kg = f"{env.extra_payload:.1f}"
            print(f"Step {i} | Speed: {env.speed:.1f} km/h | Temp: {env.temp:.2f}°C | SoC: {env.soc:.1f}% | Distance: {env.obstacle_distance:.1f}m | Weather: {env.weather} | Slope degree: {env.slope_angle:.2f} | Tire pressure: {env.tire_pressure:.2f}PSI | Payload: {payload_kg}kg | RPM: {current_rpm:.0f} | TTC: {ttc:.2f}s | Attention: {env.attention_level:.2f} | Distance left: {env.adjacent_vehicle_left:.2f} | Distance right: {env.adjacent_vehicle_right:.2f}")
        if done or truncated:
            break
    
    env.plot_performance()
