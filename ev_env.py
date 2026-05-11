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
from stable_baselines3.common.callbacks import CheckpointCallback

class SafetyGuard:
    @staticmethod
    def apply(env, action):
        if env.acceleration_prev < -30.0:
            action = max(action, -0.5)

        if env.speed > 150.0:
            action -= 1.0

        if env.traffic_light_state == "red" and env.speed > 2.0:
            if env.traffic_light_distance < 15.0:
                action = -1.0   # Full emergency brake

            elif env.traffic_light_distance < 30.0:
                vel = env.speed / 3.6
                dist = max(1.0, env.traffic_light_distance - 1.0)
                req = (vel ** 2) / (2.0 * dist * 6.0)
                action = min(action, -min(1.0, req))

        if env.speed > 0.1:
            ttc = env.obstacle_distance / (env.speed / 3.6 + 0.001)
            if ttc < 0.5:
                action -= 1.0

        if env.animal_active and env.animal_distance < 5.0 and env.speed > 2.0:
            action -= 1.0

        if env.rollover_risk > 0.95:
            action = 0.0

        if env.temp > 95.0 or env.soc < 3.0:
            action = min(action, 0.1)

        if env.attention_level < 0.35 and env.speed > 30.0:
            action = min(action, -0.2)

        return float(np.clip(action, -1.0, 1.0))

# HYBRID CONTROL — Driver + AI
class HybridController:
    @staticmethod

    def blend(env, agent_action, driver_input):
        """
        'auto' — AI full control, driver watches
        'hybrid' — AI suggests, driver mainly drives
        'manual' — Driver controls, AI safety net only
        """

        mode = env.drive_mode

        if mode == "auto":
            final = agent_action
            env.override_active = False
            env.override_reason = None

        elif mode == "manual":
            final = driver_input
            env.override_active = False
            env.override_reason = None

            if env.obstacle_distance < 20.0:
                final = min(final, -0.5)
                env.override_active = True
                env.override_reason = "AEB"
            
            if env.speed > env.speed_limit * 1.1:
                final = min(final, 0.0)
                env.override_active = True
                env.override_reason = "SpeedLimit"
            
            if (env.traffic_light_state == "red"
                and env.traffic_light_distance < 30.0
                and env.speed > 2.0):

                final = min(final, -0.5)
                env.override_active = True
                env.override_reason = "RedLight"

        else:
            if abs(driver_input) > 0.1:
                blend = 0.3
                final = agent_action * (1 - blend) + driver_input * blend

            else:
                final = agent_action

            env.override_active = False
            env.override_reaon = None

            if env.obstacle_distance < 30.0:
                final = min(final, -0.5)
                env.override_active = True
                env.override_reason = "Obstacle"
            
            if env.speed > env.speed_limit * 1.05:
                final = min(final, 0.0)
                env.override_active = True
                env.override_reason = "SpeedLimit"
            
            if (env.traffic_light_state == "red"
                and env.traffic_light_distance < 30.0
                ):

                final = min(final, -0.3)
                env.override_active = True
                env.override_reason = "RedLight"

            if env.animal_active and env.animal_distance < 40.0:
                final = min(final, 0.0)
                env.override_active = True
                env.override_reason = "Animal"

        return float(np.clip(final, -1.0, 1.0))

class EVStartupEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.gravity      = 9.81
        self.wheel_radius = 0.33
        self.cd_base      = 0.22
        self.gear_ratios  = {1: 15.0}

        self.battery_capacity_kwh = 100.0
        self.cap_max_kwh          = 3.0
        self.max_temp             = 100.0

        self.action_space      = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(47,), dtype=np.float32)

        self.traffic_light_state    = "green"
        self.traffic_light_timer    = 0.0
        self.traffic_light_cycle    = {"green": 30.0, "amber": 3.0, "red": 30.0}
        self.ran_red_light          = False

        self.driver_regen_level = 1.0 
        self.animal_active   = False
        self.animal_distance = 999.0
        self.animal_velocity = 0.0
        self.animal_type     = "cow"
        self.animal_crossed  = False

        self.intersection_distance = 500.0
        self.intersection_active   = False
        self.cross_traffic_speed   = 40.0
        self.cross_traffic_dist    = 50.0
        self.intersection_type     = "signal"

        self.swerve_active   = False
        self.swerve_steps    = 0

        self.rear_vehicle_left  = 60.0
        self.rear_vehicle_right = 60.0

        self.drive_mode        = "auto"
        self.driver_input      = 0.0
        self.override_active   = False
        self.override_reason   = None

        self.indicator_left         = False
        self.indicator_right        = False
        self.indicator_steps        = 0
        self.lane_change_intent     = None
        self.indicator_reward_given = False
        self.driver_coast_pref = False

        self.reset()

        
    def reset(self, seed=None, options=None):   # Seed - required for reproducibility, Options - for future extensions
        super().reset(seed=seed)

        self.slope_angle  = np.random.uniform(0.0, 12.0)
        self.slope_rad = np.radians(self.slope_angle)
        self.extra_payload= np.random.uniform(0.0, 500.0)
        self.mass = 2100.0 + self.extra_payload
        self.tire_pressure = np.random.uniform(28.0, 35.0)
        self.speed  = 0.0
        self.last_speed = 0.0
        self.last_acceleration = 0.0
        self.acceleration_prev = 0.0
        self.stall_steps = 0

        self.soc = 100.0
        self.cap_soc_kwh = 0.0
        self.total_energy_spent_kwh  = 0.0
        self.total_energy_recovered_kwh= 0.0
        self.decel_rate  = 0.0
        self.range_anxiety = False

        self.ambient_temp = 35.0
        self.temp = self.ambient_temp
        self.battery_temp = np.random.uniform(28.0, 42.0)
        self.chiller_active = False
        self.chiller_power = 0.0
        self.cell_temp_variance = np.random.uniform(0.0, 5.0)

        self.weather = np.random.choice(["dry", "rain"], p=[0.75, 0.25])
        self.mu = 0.9 if self.weather == "dry" else 0.45
        self.speed_limit= float(np.random.choice([60, 70, 80, 90, 100, 120]))
        self.road_surface = np.random.choice(
            ["asphalt", "wet", "gravel", "pothole"],
            p=[0.60, 0.20, 0.15, 0.05]
        )

        self.obstacle_distance = np.random.uniform(100.0, 200.0)
        self.obstacle_velocity = np.random.uniform(-10.0, 10.0)   # m/s relative
        self.airbag_deployed   = False
        self.lateral_g = 0.0
        self.pitch_angle = 0.0
        self.rollover_risk = 0.0

        self.driver_throttle_sens = 1.0 

        self.adjacent_vehicle_left = np.random.uniform(150.0, 300.0)
        self.adjacent_vehicle_right = np.random.uniform(150.0, 300.0)
        self.rear_vehicle_left = np.random.uniform(30.0, 100.0)
        self.rear_vehicle_right  = np.random.uniform(30.0, 100.0)
        self.blind_spot_warning = False
        self.oncoming_distance = np.random.uniform(200.0, 500.0)
        self.oncoming_speed = np.random.uniform(40.0, 100.0)
        self.head_on_risk = 0.0

        self.traffic_light_distance = np.random.uniform(150.0, 400.0)
        self.traffic_light_state    = np.random.choice(
            ["green", "amber", "red"], p=[0.60, 0.15, 0.25]
        )
        self.traffic_light_timer = 0.0
        self.traffic_light_cycle = {
            "green": np.random.uniform(20.0, 40.0),
            "amber": 3.0,
            "red":   np.random.uniform(20.0, 40.0)
        }
        self.ran_red_light = False

        self.animal_active = np.random.random() < 0.12   # 12 % chance
        self.animal_distance = np.random.uniform(80.0, 250.0)
        self.animal_velocity = np.random.uniform(-3.0, 3.0) # m/s lateral
        self.animal_type = np.random.choice(
            ["cow", "dog", "deer"], p=[0.50, 0.30, 0.20]
        )
        self.animal_crossed  = False

        self.intersection_distance = np.random.uniform(200.0, 500.0)
        self.intersection_active = np.random.random() < 0.30
        self.cross_traffic_speed = np.random.uniform(20.0, 60.0)  # km/h
        self.cross_traffic_dist= np.random.uniform(20.0, 80.0)  # m to intersection
        self.intersection_type = np.random.choice(
            ["signal", "yield", "stop"], p=[0.50, 0.30, 0.20]
        )

        self.swerve_active = False
        self.swerve_steps = 0

        self.cd_active = self.cd_base
        self.suspension_lowered  = False

        self.slope_history = [self.slope_angle] * 10
        self.slope_change_rate = 0.0
        self.grade_percent = np.tan(self.slope_rad) * 100

        self.drive_mode = "auto"
        self.driver_input = 0.0
        self.override_active = False
        self.override_reason = None

        self.acc_enabled = True
        self.acc_target_speed = float(np.random.choice([60, 80, 100, 120]))
        self.current_gear = 1
        self.esc_active = 0

        self.attention_level = np.random.uniform(0.6, 1.0)
        self.attention_decay_rate = np.random.uniform(0.0001, 0.0005)

        self.indicator_left = False
        self.indicator_right = False
        self.indicator_steps = 0      
        self.lane_change_intent = None   
        self.indicator_reward_given = False

        self.steering_angle = 0.0   
        self.steering_rate = 0.0    

        self.lead_vehicle_speed = np.random.uniform(40.0, 100.0)  
        self.lead_vehicle_accel = 0.0    

        self.brake_pressure = 0.0   
        self.driver_throttle = 0.0    

        self.dt = 0.1
        self.drive_mode_performance = np.random.choice(
        ["eco", "normal", "sport", "track", "custom"],
        p=[0.25, 0.35, 0.20, 0.10, 0.10]
    )
        
        self.apply_mode_settings()

        self.driver_regen_level = 1.0   #  driver adjusts regen strength
        self.driver_throttle_sens = 1.0   #  throttle sensitivity
        self.driver_coast_pref = False  # Driver manually enables coast
        self.custom_mode_active = False  # True when driver overrides mode

        # Cooling
        self.cooling_fan_speed      = 0.5   # 0.0 to 1.0
        self.battery_cooling_active = False

        # Torque vectoring (track mode)
        self.torque_bias_rear       = 0.5   # 0.0=front, 0.5=balanced, 1.0=rear
        self.stability_control_on   = True

        self.history = {
            'soc': [], 'temp': [], 'speed': [], 'gear': [],
            'tc': [], 'abs': [], 'aeb': [], 'tpms_loss': [],
            'esc': [], 'tl_state': [], 'swerve': [], 'indicator': []
        }

        return self.get_obs(), {}
    
    def apply_mode_settings(self):
        """Apply performance mode settings."""
        mode = self.drive_mode_performance

        if mode == "eco":
            self.mode_speed_limit_factor  = 0.80
            self.mode_regen_aggression    = 2.0    # Maximum regen
            self.mode_throttle_smoothing  = 0.5    # Very smooth/dull throttle
            self.mode_coast_enabled       = True
            self.mode_speed_limit_enforce = True
            self.mode_boost_allowed       = False
            self.cooling_fan_speed        = 0.3    # Reduced — saves power
            self.battery_cooling_active   = False
            self.stability_control_on     = True
            self.torque_bias_rear         = 0.5

        elif mode == "normal":
            self.mode_speed_limit_factor  = 0.92
            self.mode_regen_aggression    = 1.0
            self.mode_throttle_smoothing  = 1.0
            self.mode_coast_enabled       = False
            self.mode_speed_limit_enforce = True
            self.mode_boost_allowed       = False
            self.cooling_fan_speed        = 0.5
            self.battery_cooling_active   = False
            self.stability_control_on     = True
            self.torque_bias_rear         = 0.5

        elif mode == "sport":
            self.mode_speed_limit_factor  = 0.98
            self.mode_regen_aggression    = 0.6    # Low — glide feel
            self.mode_throttle_smoothing  = 1.4    # Sharp
            self.mode_coast_enabled       = False
            self.mode_speed_limit_enforce = True
            self.mode_boost_allowed       = True
            self.cooling_fan_speed        = 0.8
            self.battery_cooling_active   = True
            self.stability_control_on     = True
            self.torque_bias_rear         = 0.6    # Slight rear bias

        elif mode == "track":
            self.mode_speed_limit_factor  = 1.30   # Well above limit
            self.mode_regen_aggression    = 0.3    # Minimal — pure speed
            self.mode_throttle_smoothing  = 1.8    # Extreme response
            self.mode_coast_enabled       = False
            self.mode_speed_limit_enforce = False
            self.mode_boost_allowed       = True
            self.cooling_fan_speed        = 1.0    # Max cooling
            self.battery_cooling_active   = True
            self.stability_control_on     = False  # Driver has full control
            self.torque_bias_rear         = 0.75   # Rear-biased for rotation

        elif mode == "custom":
            # Custom — use driver settings directly
            # Driver sets everything manually
            self.mode_speed_limit_factor = 0.95   # Default
            self.mode_regen_aggression = self.driver_regen_level
            self.mode_throttle_smoothing = self.driver_throttle_sens
            self.mode_coast_enabled = self.driver_coast_pref
            self.mode_speed_limit_enforce = True
            self.mode_boost_allowed = False
            self.cooling_fan_speed = 0.5
            self.battery_cooling_active = False
            self.stability_control_on = True
            self.torque_bias_rear = 0.5
    
    def get_obs(self):
        
        # Encoding
        tl_enc  = {"green": 1.0, "amber": 0.5, "red": 0.0}
        rd_enc  = {"asphalt": 1.0, "wet": 0.67, "gravel": 0.33, "pothole": 0.0}
        dm_enc  = {"auto": 1.0, "hybrid": 0.5, "manual": 0.0}
        mode_perf_enc = {
            "eco": 0.0, "normal": 0.25, "sport": 0.50,
            "track": 0.75, "custom": 1.0
        }

        # Normalization for better learning 
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
        np.clip(self.lateral_g, 0, 1),
        self.speed_limit / 120.0,
        1.0 if self.weather == "dry" else 0.0,
        np.clip(self.cell_temp_variance / 10.0, 0, 1),
        np.clip(abs(self.pitch_angle) / 15.0, 0, 1),
        np.clip(self.last_acceleration / 10.0, -1.0, 1.0),
        np.clip((self.battery_temp - 35.0) / 35.0, -1.0, 1.0),
        min(1.0, self.adjacent_vehicle_left / 200.0),
        min(1.0, self.adjacent_vehicle_right / 200.0),
        self.attention_level,
        1.0 if self.chiller_active else 0.0,
        np.clip(self.oncoming_distance / 300.0, 0, 1),
        np.clip(self.head_on_risk, 0, 1),
        np.clip(self.rollover_risk, 0, 1),
        np.clip(self.slope_change_rate / 5.0, -1, 1),
        np.clip(self.cd_active / 0.30, 0, 1),
        1.0 if self.range_anxiety else 0.0,
        np.clip(self.total_energy_recovered_kwh / 5.0, 0, 1),
        tl_enc.get(self.traffic_light_state, 1.0),
        min(1.0, self.traffic_light_distance / 200.0),
        1.0 if self.animal_active else 0.0,
        min(1.0, self.animal_distance / 200.0),
        rd_enc.get(self.road_surface, 1.0),
        dm_enc.get(self.drive_mode, 1.0),
        1.0 if self.override_active else 0.0,
        1.0 if self.swerve_active else 0.0,
        1.0 if self.indicator_left  else 0.0,   
        1.0 if self.indicator_right else 0.0,  
        np.clip(self.steering_angle / 30.0, -1, 1),              
        np.clip(self.lead_vehicle_speed / 120.0, 0, 1),          
        self.brake_pressure,                                     
        np.clip((self.speed - self.lead_vehicle_speed) /         
                50.0, -1, 1),  
        mode_perf_enc.get(self.drive_mode_performance, 0.33),
        np.clip(self.driver_regen_level / 2.0, 0, 1),     
        np.clip(self.driver_throttle_sens / 1.5, 0, 1),   
        1.0 if self.stability_control_on else 0.0,          
        np.clip(self.torque_bias_rear, 0, 1),               
        self.cooling_fan_speed
        ],
        dtype= np.float32)
    
    def get_motor_rpm(self):
        speed_mps = self.speed / 3.6 
        return (speed_mps / (2 * np.pi * self.wheel_radius)) * self.gear_ratios[1] * 60     # RPM = (Vehicle Speed / Wheel Circumference ) * Gear Ratio * 60 

    def step(self, action):
        torque_request = action[0]
        reward = 0.0
        dt = 0.1  # Time step in seconds
        tc_active, abs_active, aeb_active = 0, 0, 0
        self.last_speed = getattr(self, 'last_speed', 0.0)
        under_limit = self.speed <= self.speed_limit
        done = False
        agent_torque = float(action[0])
        torque_request = HybridController.blend(self, agent_torque, self.driver_input)
        target_speed = 0.0

        # ------------- SENSORS -------------
        closing_speed = (self.speed - self.lead_vehicle_speed) / 3.6
        self.obstacle_distance -= closing_speed * dt
        self.obstacle_distance -= self.obstacle_velocity * dt
        
        lead_accel_noise = np.random.normal(0, 0.1)
        target_lead = self.speed * 0.92 + lead_accel_noise
        self.lead_vehicle_speed = np.clip(
            self.lead_vehicle_speed + (target_lead - self.lead_vehicle_speed) * 0.05,
            10.0, 150.0
        )
        if self.obstacle_distance < 15.0:
            self.obstacle_distance = np.random.uniform(300.0, 500.0)
            self.obstacle_velocity = np.random.uniform(0.0, 5.0)
        
        relative_speed_mps = (self.speed + self.oncoming_speed) / 3.6
        self.oncoming_distance -= relative_speed_mps * dt

        if self.oncoming_distance < 20.0:
            self.oncoming_distance = np.random.uniform(300.0, 600.0)
            self.oncoming_speed = np.random.uniform(40.0, 100.0)

        oncoming_ttc = self.oncoming_distance / (relative_speed_mps + 0.001)
        self.head_on_risk = np.clip(1.0 - (oncoming_ttc / 5.0), 0, 1)

        if oncoming_ttc < 3.0 and self.speed > 20.0:
            torque_request = min(torque_request, 0.0)
            reward -= (3.0 - oncoming_ttc) * 2.0

        if oncoming_ttc < 1.0:
            torque_request = -1.0
            reward -= 20.0

        # Rear cam
        self.rear_vehicle_left = max(0.0, self.rear_vehicle_left - 0.5 * dt)
        self.rear_vehicle_right = max(0.0, self.rear_vehicle_right - 0.5 * dt)

        if self.rear_vehicle_left < 5.0:
            self.rear_vehicle_left  = np.random.uniform(40.0, 100.0)

        if self.rear_vehicle_right < 5.0:
            self.rear_vehicle_right = np.random.uniform(40.0, 100.0)

        # Dynamic slope
        self.slope_history.append(self.slope_angle)
        self.slope_history.pop(0)
        self.slope_change_rate = self.slope_history[-1] - self.slope_history[0]
        self.grade_percent = np.tan(self.slope_rad) * 100

        if np.random.random() < 0.002:
            self.slope_angle = np.clip(
                self.slope_angle + np.random.uniform(-1.5, 1.5),
                0.0,
                12.0
            )

            self.slope_rad = np.radians(self.slope_angle)

        # ------------- TRAFFIC LIGHTS -------------
        self.traffic_light_timer += dt
        phase_duration = self.traffic_light_cycle[self.traffic_light_state]

        if self.traffic_light_timer >= phase_duration:
            self.traffic_light_timer = 0.0

            if self.traffic_light_state == "green":
                self.traffic_light_state = "amber"

            elif self.traffic_light_state == "amber":
                self.traffic_light_state = "red"

            elif self.traffic_light_state == "red":
                self.traffic_light_state = "green"
                self.traffic_light_distance = np.random.uniform(150.0, 400.0)

        self.traffic_light_distance -= (self.speed / 3.6) * dt

        if self.traffic_light_distance > 0:
            traffic_ttc = self.traffic_light_distance / (self.speed / 3.6 + 0.001)
        
            if self.traffic_light_state == "red":
                if self.traffic_light_distance < 3.0 and self.speed > 3.0:
                    self.ran_red_light = True
                    reward -= 200.0
                    done    = True
                else:
                    v_mps = self.speed / 3.6
                    achievable_decel = self.mu * 5.0
                    min_stop_dist = (v_mps ** 2) / (2.0 * max(0.1, achievable_decel)) + 5.0

                    if v_mps > 0.1:
                        # Calculate exact torque needed to stop in time
                        stop_dist = max(1.0, self.traffic_light_distance - 3.0)
                        req_decel = (v_mps ** 2) / (2.0 * stop_dist)
                        tl_brake  = -min(1.0, req_decel / achievable_decel)
                        torque_request = min(torque_request, tl_brake)

                        # Reward smooth calculated braking
                        if tl_brake < -0.1 and under_limit:
                            reward += 0.2

                        # Penalty scales with proximity
                        if self.traffic_light_distance < min_stop_dist:
                            reward -= (min_stop_dist - self.traffic_light_distance) * 0.5

            elif self.traffic_light_state == "amber":
                if traffic_ttc < 5.0:
                    torque_request = min(torque_request, 0.0)
                    reward -= 0.3

            elif self.traffic_light_state == "green":
                if self.traffic_light_distance < 2.0:
                    self.traffic_light_distance = np.random.uniform(150.0, 400.0)
                    self.traffic_light_state    = "green"
                    self.traffic_light_timer    = 0.0
                    if under_limit:
                        reward += 1.0

        # ------------- ANIMAL CROSSING? -------------
        if self.animal_active:
            # Animal closes in relative to our speed
            self.animal_distance -= (self.speed / 3.6) * dt

            # Velocity-based prediction — where will animal be in 2.5s?
            predicted_animal_pos = self.animal_distance - (self.speed / 3.6) * 2.5
            animal_will_cross    = predicted_animal_pos < 8.0  

            # TTC based on current closing speed
            animal_ttc = self.animal_distance / (self.speed / 3.6 + 0.001)

            if self.animal_distance < 3.0 and self.speed > 3.0:
                reward -= 300.0
                done    = True

            # -- ZONE 1: Far warning (8-15s TTC) — begin deceleration --
            elif animal_ttc < 15.0 and animal_ttc >= 8.0:
                # Start braking gently — prepare for swerve or stop
                torque_request = min(torque_request, 0.3)
                reward -= 0.1

            # -- ZONE 2: Medium warning (4-8s TTC) — reduce to swerve speed --
            elif animal_ttc < 8.0 and animal_ttc >= 4.0:
                # Must reduce speed to safe swerve speed (< 40 km/h)
                if self.speed > 40.0:
                    # Calculate torque needed to reach 40 km/h before animal
                    # Deceleration target: reach 40 km/h in remaining distance
                    speed_excess  = (self.speed - 40.0) / 3.6  # m/s over target
                    dist_to_brake = self.animal_distance * 0.6  # use 60% of distance
                    req_decel     = (speed_excess ** 2) / (2.0 * dist_to_brake + 0.001)
                    brake_torque  = -min(1.0, req_decel / 8.0)  # normalize to -1, 0
                    torque_request = min(torque_request, brake_torque)
                    reward -= 0.3
                else:
                    # Already at safe speed — coast
                    torque_request = min(torque_request, 0.0)

            # -- ZONE 3: Critical (< 4s TTC) — swerve or emergency brake --
            elif animal_ttc < 4.0 or animal_will_cross:

                # Safe swerve speed — must be under this to swerve without rollover
                # Physics: lateral force = m*v²/r — higher speed = higher rollover risk
                # Under 35 km/h swerve is generally safe for most vehicles
                swerve_speed_limit = 35.0  # km/h

                # Dynamic rollover threshold based on speed
                # At 20 km/h: allow rollover_risk < 0.3
                # At 35 km/h: allow rollover_risk < 0.15
                # Above 35 km/h: don't swerve at all
                if self.speed <= swerve_speed_limit:
                    max_rollover = np.clip(
                        0.3 - (self.speed / swerve_speed_limit) * 0.2,
                        0.05, 0.30
                    )
                else:
                    max_rollover = 0.0  # Above swerve limit — never swerve

                lane_clear  = (self.adjacent_vehicle_left  > 40.0 or
                                self.adjacent_vehicle_right > 40.0)
                rear_clear = (self.rear_vehicle_left  > 15.0 and
                                self.rear_vehicle_right > 15.0)
                oncoming_clear = self.oncoming_distance > 80.0
                v_swerve = self.speed / 3.6
                steer_rad_est = np.radians(15.0)  # swerve steering angle
                wheelbase = 2.7
                turn_r_est = wheelbase / np.tan(steer_rad_est + 0.001)
                lat_g_est  = np.clip((v_swerve ** 2) / (turn_r_est * self.gravity), 0, 1)
                rollover_est = np.clip(
                    lat_g_est * (self.speed / 150.0) * (1.0 + self.extra_payload / 500.0),
                    0, 1
                )
                rollover_safe = rollover_est < max_rollover
                speed_ok = self.speed <= swerve_speed_limit

                if (lane_clear and rear_clear and oncoming_clear
                        and rollover_safe and speed_ok and self.speed > 3.0):

                    # -- EXECUTE SWERVE--------------------------------
                    # Maintain current speed during swerve (no throttle, no brake)
                    torque_request  = min(torque_request, 0.05)
                    self.swerve_active = True
                    self.swerve_steps  = 12

                    # Indicator fires here (your indicator block handles reward)
                    if under_limit:
                        reward += 3.0

                elif not speed_ok and self.speed > swerve_speed_limit:
                    v_mps = self.speed / 3.6
                    v_swerve = swerve_speed_limit / 3.6

                    # Distance needed to brake to swerve speed:
                    dist_to_swerve_speed = (v_mps**2 - v_swerve**2) / (2.0 * 6.0)

                    # Distance needed after that to swerve (approx 5m):
                    dist_to_complete_swerve = dist_to_swerve_speed + 5.0

                    if self.animal_distance > dist_to_complete_swerve + 3.0:
                        # Check if we can ACTUALLY complete two-phase given current TTC
                        if animal_ttc > 3.0:
                            # Enough time — brake to swerve speed
                            speed_diff = v_mps - v_swerve
                            dist_left = max(1.0, self.animal_distance - 5.0)
                            req_decel = (speed_diff ** 2) / (2.0 * dist_left)
                            brake_torque = -min(1.0, req_decel / 9.8)
                            torque_request = min(torque_request, brake_torque)
                            self.swerve_active = False
                            reward -= 0.3
                        else:
                            # TTC too short for two-phase — full emergency stop
                            stop_dist = max(1.0, self.animal_distance - 2.0)
                            req_decel = (v_mps ** 2) / (2.0 * stop_dist)
                            brake_torque = -min(1.0, req_decel / 9.8)
                            torque_request = min(torque_request, brake_torque)
                            self.swerve_active = False
                            reward -= (4.0 - min(animal_ttc, 4.0)) * 2.0
                    else:
                        # Not enough room for two-phase — full emergency stop
                        stop_dist = max(1.0, self.animal_distance - 2.0)
                        req_decel = (v_mps ** 2) / (2.0 * stop_dist)
                        brake_torque = -min(1.0, req_decel / 9.8)
                        torque_request = min(torque_request, brake_torque)
                        self.swerve_active = False
                        reward -= (4.0 - min(animal_ttc, 4.0)) * 2.0

                else:
                    # Cannot swerve at all — emergency brake to stop before animal
                    # Calculate torque to stop within animal_distance - 2m safety margin
                    stop_dist = max(1.0, self.animal_distance - 2.0)
                    speed_mps = self.speed / 3.6
                    req_decel = (speed_mps ** 2) / (2.0 * stop_dist)
                    brake_torque = -min(1.0, req_decel / 9.8)

                    torque_request = min(torque_request, brake_torque)
                    self.swerve_active = False
                    reward -= (4.0 - min(animal_ttc, 4.0)) * 3.0

            # Animal passed / cleared road
            if self.animal_distance < 0:
                self.animal_active  = False
                self.animal_crossed = True
                self.swerve_active  = False
                if under_limit:
                    reward += 2.0

        # Swerve countdown (runs regardless of animal_active)
        if self.swerve_active:
            self.swerve_steps -= 1
            if self.swerve_steps <= 0:
                self.swerve_active = False

        # ---------------- VIRTUAL INTERSECTION ----------------
        if self.intersection_active:
            self.intersection_distance -= (self.speed / 3.6) * dt

            if 0 < self.intersection_distance < 80.0:
                cross_ttc = self.cross_traffic_dist / (self.cross_traffic_speed / 3.6 + 0.001)
                int_ttc   = self.intersection_distance / (self.speed / 3.6 + 0.001)

                if self.intersection_type in ("signal", "stop"):
                    if cross_ttc < 4.0:
                        torque_request = min(torque_request, 0.0)
                        reward -= 1.0

                        if int_ttc < 2.0 and self.speed > 10.0:
                            reward -= 10.0

                elif self.intersection_type == "yield":
                    if cross_ttc < 3.0:
                        torque_request = min(torque_request, 0.0)

            if self.intersection_distance < 0:
                self.intersection_distance = np.random.uniform(200.0, 600.0)
                self.cross_traffic_speed = np.random.uniform(20.0, 60.0)
                self.cross_traffic_dist = np.random.uniform(20.0, 80.0)
                self.intersection_type = np.random.choice(
                    ["signal", "yield", "stop"], p=[0.50, 0.30, 0.20]
                )

                if under_limit:
                    reward += 1.0  

        # ------------- SAFETY FEATURES -------------
        rpm = self.get_motor_rpm()

        # AEB - Automatic brakes
        ttc = self.obstacle_distance / (self.speed / 3.6 + 0.00001)

        if ttc < 0.8:
            torque_request = -1.0
            aeb_active = 1
            reward -= 15.0 

        elif ttc < 1.5: 
            torque_request = min(torque_request, -0.7)  # Limit throttle
            aeb_active = 0.8
            reward -= 5.0
        
        elif ttc < 3.0:
            torque_request = min(torque_request, 0.0)  # No throttle, but allow coasting
            reward -= (3.0 - ttc) * 0.5 

        elif ttc < 5.0:
            torque_request = min(torque_request, 0.5)
            reward -= 0.3

        elif ttc < 8.0 and self.speed > 60.0:
            reward -= 0.1

        torque_request = SafetyGuard.apply(self, torque_request)

        if torque_request >= 0:
            adjusted_request = 0.05 + (torque_request * 0.95)  # Lower minimum and smoother ramp
        
        else:
            adjusted_request = torque_request   

        # BMS - thermal and SoC protection
        bms_limit = 1.0

        if self.temp > 88 or self.soc < 20:
            weather_derate = 0.7 if self.weather == "rain" else 1.0
            slope_hold_minimum = np.clip(
                self.mass * self.gravity * np.sin(self.slope_rad) /
                (14000.0 * weather_derate),
                0.0, 0.6
            )
            bms_limit = max(0.3, slope_hold_minimum)
            
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
        if self.mode_boost_allowed and self.cap_soc_kwh > 1.0 and torque_request > 0.8:
            if self.drive_mode_performance == "track":
                boost = 1.25   # Maximum boost in speed mode
            elif self.drive_mode_performance == "sport":
                boost = 1.15
            else:
                boost = 1.1
        else:
            boost = 1.0

        def get_smooth_mult(request):
            base_torque = 250.0 + (100.0 * math.tanh(5 * request))
            if self.speed < 15:
                if self.slope_angle > 3.0:
                    # Scale boost with slope severity
                    slope_boost = 1.0 + (self.slope_angle / 12.0) * 0.8
                    return base_torque * slope_boost
                rain_factor = 0.7 if self.weather == "rain" else 1.8
                return base_torque * rain_factor
            return base_torque

        motor_torque_mult = get_smooth_mult(applied_torque)
        motor_force = (applied_torque * motor_torque_mult * boost * self.gear_ratios[1] * motor_efficiency) / self.wheel_radius

        # Torque vectoring — rear bias increases cornering in track/sport
        if self.torque_bias_rear > 0.5 and self.lateral_g > 0.2:
            # More torque to rear during cornering = better rotation
            rear_bias_effect = (self.torque_bias_rear - 0.5) * 2.0  # 0 to 1
            # Slight motor force increase when rear biased (rear-wheel drive benefit)
            motor_force *= (1.0 + rear_bias_effect * 0.1)
            if self.drive_mode_performance == "track" and under_limit is False:
                reward += rear_bias_effect * self.lateral_g * 0.3

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

        pressure_loss_pct = max(0, (optimal_pressure - self.tire_pressure) / optimal_pressure) 

        tpms_mult = 1.0 + (pressure_loss_pct * 0.5)  # If pressure is low, we get a multiplier that increases rolling resistance and reduces performance
        rolling_resistance = self.mass * self.gravity * tpms_mult * 0.015 * np.cos(self.slope_rad)

        if pressure_loss_pct > 0:
            efficency_penalty = pressure_loss_pct * (self.speed / 100.0)
            reward -= efficency_penalty

        if self.speed > 70.0:
            self.suspension_lowered = True
            cd_reduction = min(0.03, (self.speed - 70.0) * 0.001)
            self.cd_active = self.cd_base - cd_reduction

        else:
            self.suspension_lowered = False
            self.cd_active = self.cd_base

        if self.temp > 50.0:
            self.cd_active += 0.01

        surface_cd = {"asphalt": 0.0, "wet": 0.005, "gravel": 0.01, "pothole": 0.015}
        self.cd_active += surface_cd.get(self.road_surface, 0.0)
        self.cd_active = np.clip(self.cd_active, 0.18, 0.28)

        drag = 0.5 * self.cd_active * 1.225 * 2.2 * (self.speed / 3.6) ** 2
        gravity_pull = self.mass * self.gravity * np.sin(self.slope_rad)

        if self.speed > 70.0 and self.cd_active < 0.21 and under_limit:
            reward += (0.22 - self.cd_active) * 5.0
              
        acceleration = (motor_force - drag - gravity_pull - rolling_resistance) / self.mass
        self.speed = max(0.0, self.speed + (acceleration * 3.6 * dt))
        self.acceleration_prev = acceleration

        # ---------------- LATERAL G AND RIDE COMFORT ----------------
        if self.swerve_active:
            target_steering = 15.0 * self.swerve_direction if hasattr(self, 'swerve_direction') else 12.0
        else:
            target_steering = self.driver_throttle * 10.0  # driver steering

        # Steering rate limited to 30 deg/s (realistic)
        max_steer_rate = 30.0 * self.dt
        self.steering_angle += np.clip(
            target_steering - self.steering_angle,
            -max_steer_rate,
            max_steer_rate
        )
        self.steering_angle = np.clip(self.steering_angle, -30.0, 30.0)

        # a_lateral = v² * tan(steering) / wheelbase
        wheelbase = 2.7  # metres (typical sedan car)
        steer_rad = np.radians(abs(self.steering_angle))
        if steer_rad > 0.001 and self.speed > 1.0:
            turn_radius = wheelbase / np.tan(steer_rad + 0.001)
            v_mps       = self.speed / 3.6
            self.lateral_g = np.clip((v_mps ** 2) / (turn_radius * self.gravity), 0, 1)
        else:
            self.lateral_g = 0.0
                
        longitudinal_g = abs(acceleration) / self.gravity 
        lateral_comfort_limit = 0.4
        self.pitch_angle  = np.clip(acceleration * 0.3, -15.0, 15.0)

        if self.lateral_g > lateral_comfort_limit:
            reward -= (self.lateral_g - lateral_comfort_limit) * 3.0
        
        if longitudinal_g > 0.6:
            reward -= (longitudinal_g - 0.6) 
        
        self.rollover_risk = np.clip(
            (self.lateral_g) *
            (self.speed / 150.0) *
            (1.0 + self.extra_payload / 500.0),
            0, 1
        )

        if self.rollover_risk > 0.7:
            torque_request *= (1.0 - self.rollover_risk * 0.5)
            reward -= self.rollover_risk * 5.0

        if self.rollover_risk > 0.9:
            reward -= 50.0
            done = True

        # Aquaplaning risk
        if self.weather == "rain" and self.speed > 80.0:
            aquaplane_risk = np.clip(
                (self.speed - 80.0) / 50.0 *
                (1.0 + pressure_loss_pct),
                0, 1
                )

            if aquaplane_risk > 0.4:
                effective_mu = self.mu * (1.0- aquaplane_risk * 0.5)
                traction_limit = effective_mu * self.mass * self.gravity * np.cos(self.slope_rad)
                reward -= aquaplane_risk * 3.0

        if self.road_surface == "pothole" and np.random.random() < 0.01:
            jerk_spike = np.random.uniform(3.0, 8.0)
            reward -= jerk_spike * 0.1
            self.attention_level = min(1.0, self.attention_level + 0.2)

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

        if self.stability_control_on:
            stability_threshold = 0.6
            if self.lateral_g > stability_threshold:
                esc_correction  = (self.lateral_g - stability_threshold) / 0.4
                torque_request  = torque_request * (1.0 - esc_correction * 0.5)
                self.esc_active = 1
                reward -= esc_correction * 1.5
        else:
            # Track mode — ESC off, but still warn on extreme lateral G
            if self.lateral_g > 0.85:
                reward -= (self.lateral_g - 0.85) * 3.0  # Physics limit penalty on

        # ---------------- HILL START ASSIST ----------------
        if self.slope_angle > 3.0 and self.speed < 20.0 and torque_request > 0:
            gravity_component = self.mass * self.gravity * np.sin(self.slope_rad)
            weather_factor = 0.7 if self.weather == "rain" else 1.0
            # Scale minimum proportionally — less boost needed at higher speeds
            speed_factor = max(0.3, 1.0 - (self.speed / 30.0))
            min_hold_torque = np.clip(
                gravity_component * speed_factor / (12273.0 * weather_factor * 0.9),
                0.0, 0.95
            )
            adjusted_request = max(adjusted_request, min_hold_torque)
            if under_limit:
                reward += 0.2

        # ------------- ENERGY AND THERMAL PARTS AND REGEN OPTIMIZATION -------------
        battery_current_heat = abs(applied_torque) * 0.003  # Heat from current
        battery_heat_bleed = (self.temp - self.battery_temp) * 0.01
        battery_cooling = (self.battery_temp - self.ambient_temp) * 0.02
        self.battery_temp += (battery_heat_bleed + battery_current_heat - battery_cooling) * dt
        self.battery_temp = np.clip(self.battery_temp, self.ambient_temp - 5.0, 85.0)

        self.battery_temp = max(
                                self.ambient_temp - 5.0,
                                min(self.battery_temp, 85.0)
                                )

        if self.obstacle_distance < 100 and self.speed > 20:
            regen_intensity = np.clip(
                (100 - self.obstacle_distance) / 100.0, -0, 1
                )

            if torque_request < 0 and abs(torque_request) <= regen_intensity and under_limit:
                reward += regen_intensity * 0.3

        if 100.0 < self.obstacle_distance < 150.0 and self.speed > 30.0:
            if torque_request < 0.1 and under_limit:
                reward += 0.3

        power_watts = motor_force * (self.speed / 3.6)

        # Lift and Coast
        if self.mode_coast_enabled and self.speed > 20.0:

            # Find nearest decel trigger:
            decel_triggers = []

            if self.obstacle_distance < 200.0:
                decel_triggers.append(self.obstacle_distance)

            if self.traffic_light_state in ("red", "amber"):
                if self.traffic_light_distance < 200.0:
                    decel_triggers.append(self.traffic_light_distance)

            if self.intersection_active and self.intersection_distance < 150.0:
                decel_triggers.append(self.intersection_distance)

            nearest_trigger = min(decel_triggers) if decel_triggers else 999.0

            # s = v²/(2*decel) where decel ≈ 1.5 m/s² for gentle coast

            velocity_mps = self.speed / 3.6
            coast_decel = 1.5   # m/s² natural deceleration when coasting
            target_velocity  = 5.0 / 3.6  # target 5 km/h near stop

            coast_distance_needed = (velocity_mps ** 2 - target_velocity ** 2) / (2 * coast_decel)

            # Lift point: start coasting when distance = coast_distance_needed
            should_coast = (nearest_trigger < coast_distance_needed * 1.2 and
                            nearest_trigger < 150.0)

            if should_coast:
                # Apply lift — reduce throttle to zero or light regen
                if torque_request > 0:
                    torque_request = 0.0    # Lift throttle
                    if under_limit:
                        reward += 0.4       # Reward smart lift

                elif torque_request > -0.3:
                    # Light regen during coast
                    if under_limit:
                        reward += 0.2       # Reward coasting

            # Reward being at right speed approaching trigger
            speed_appropriateness = 1.0 - abs(self.speed - 30.0) / 30.0
            if should_coast and speed_appropriateness > 0.7 and under_limit:
                reward += speed_appropriateness * 0.3


        if applied_torque < 0:

            if self.drive_mode_performance == "track":
                energy_exchange = 0.0
                raw_regen = 0.0

            v_mps = self.speed / 3.6

            regen_eff = 0.7 * np.exp(-0.5 * ((self.temp - 30.0) / 40.0) ** 2)
            regen_eff = max(0.3, regen_eff)

            # Temperature adjustment
            if self.battery_temp < 20.0:
                regen_eff *= 0.5
            elif 20.0 <= self.battery_temp <= 30.0:
                regen_eff *= 1.0
            else:
                regen_eff *= 0.85

            # Mode aggression multiplier
            if self.drive_mode_performance == "track":
                regen_eff = 0.0      # Track: no regen, pure kinetic energy
                energy_exchange = 0.0
            elif self.drive_mode_performance == "sport":
                regen_eff *= 0.4     # Sport: minimal regen, glide feel
            else:
                regen_eff *= self.mode_regen_aggression 

            # Case 1 — Obstacle/lead vehicle approaching
            if self.obstacle_distance < 100.0 and self.speed > 10.0:
                closing_speed_mps = max(0, (self.speed - self.lead_vehicle_speed) / 3.6)
                if closing_speed_mps > 0:
                    ideal_decel = (closing_speed_mps ** 2) / (
                        2.0 * max(1.0, self.obstacle_distance - 10.0)
                    )
                    ideal_regen = np.clip(ideal_decel / 5.0, 0, 1.0)
                    if abs(torque_request) > ideal_regen * 1.5:
                        reward -= 0.3   # Over-braking — wastes energy as heat
                    elif abs(torque_request) >= ideal_regen * 0.8:
                        if under_limit:
                            reward += 0.5   # Precision regen — maximum recovery

            # Case 2 — Red light approaching
            elif (self.traffic_light_state == "red" and
                self.traffic_light_distance < 100.0 and
                self.speed > 10.0):
                ideal_decel = (v_mps ** 2) / (2.0 * max(1.0, self.traffic_light_distance))
                ideal_regen = np.clip(ideal_decel / 5.0, 0, 1.0)
                if abs(torque_request) >= ideal_regen * 0.8 and under_limit:
                    reward += 0.4   # Smart stop regen

            # Case 3 — Downhill regen (engine braking equivalent)
            elif self.slope_angle < -1.0:
                target_decel = abs(self.gravity * np.sin(self.slope_rad)) * 0.8
                ideal_regen  = np.clip(target_decel / 5.0, 0, 0.6)
                if abs(torque_request) >= ideal_regen * 0.7 and under_limit:
                    reward += 0.3

            self.decel_rate = max(
                0, (self.last_speed - self.speed) / (self.dt * 3.6 + 0.001)
            )
            max_smooth_decel = (
                2.0 if self.drive_mode_performance in ("eco", "normal") else 4.0
            )
            if 0.3 <= self.decel_rate <= max_smooth_decel and under_limit:
                reward += 0.25
            elif self.decel_rate > max_smooth_decel * 1.5:
                reward -= 0.2   # Too harsh — kinetic energy wasted as brake heat

            energy_exchange = (power_watts * dt * regen_eff) / 3600000

            raw_regen = abs(energy_exchange)  # ADD THIS LINE

            space_in_cap = self.cap_max_kwh - self.cap_soc_kwh
            if self.soc > 95.0:
                if space_in_cap > 0.1:
                    cap_regen_eff = 0.95
                    regen_energy = (power_watts * dt * cap_regen_eff) / 3600000
                    to_cap = min(abs(regen_energy), space_in_cap)
                    self.cap_soc_kwh += to_cap
                    energy_exchange = 0.0
                    if under_limit:
                        reward += 0.5
                else:
                    regen_eff *= 0.1
                    energy_exchange = (power_watts * dt * regen_eff) / 3600000
                    reward -= 0.2
            else:
                energy_exchange = (power_watts * dt * regen_eff) / 3600000
                to_cap = min(abs(energy_exchange), space_in_cap)
                self.cap_soc_kwh += to_cap
                energy_exchange  += to_cap
                
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

        # -------------------- ENERGY TRACKING --------------------
        if applied_torque > 0:
            self.total_energy_spent_kwh += abs(energy_exchange)
            
        elif applied_torque < 0:
            self.total_energy_recovered_kwh += raw_regen

        if self.total_energy_spent_kwh  > 0.01:
            recovery_ratio = self.total_energy_recovered_kwh / self.total_energy_spent_kwh
            
            if recovery_ratio > 0.05 and under_limit:
                reward += recovery_ratio * 2.0

        # -------------------- RANGE APPROXIMATION --------------------
        if self.speed > 5.0 and abs(power_watts) > 100:
            power_kw = abs(power_watts) / 1000.0
            hours_remaining = (self.soc - 20.0) * self.battery_capacity_kwh / (power_kw * 100 + 0.001)
            km_remaining = hours_remaining * self.speed

            if km_remaining < 20.0:
                self.range_anxiety = True
                reward -= 1.0
            
            else:
                self.range_anxiety = False

        # -------------- BATTERY THERMAL MANAGEMENT --------------
        heat_gen = (abs(applied_torque) ** 2 * 0.02) + (abs(power_watts) / 15000.0)
        cooling = (self.temp - self.ambient_temp) * (0.05 + 0.0007 * self.speed)
        self.temp += (heat_gen - cooling) * dt

        if self.battery_cooling_active or self.cooling_fan_speed > 0.5:
            fan_cooling = self.cooling_fan_speed * 2.0  # °C/s cooling capacity
            self.temp = max(
                self.ambient_temp,
                self.temp - fan_cooling * dt
            )
            # Fan energy cost
            fan_power_watts  = self.cooling_fan_speed * 800  # 0-800W
            fan_energy = (fan_power_watts * dt) / 3600000
            self.soc -= (fan_energy / self.battery_capacity_kwh) * 100

        optimal_battery_temp_low = 15.0
        optimal_battery_temp_high = 50.0

        if self.battery_temp < optimal_battery_temp_low:
            battery_efficiency_loss = (optimal_battery_temp_low - self.battery_temp) / 20.0
            reward -= battery_efficiency_loss * 2.0

        elif self.battery_temp > optimal_battery_temp_high:
            battery_efficiency_loss = (optimal_battery_temp_high - self.battery_temp) / 30.0
            reward -= battery_efficiency_loss * 2.0

            if self.battery_temp > 50.0:
                reward -= (self.battery_temp - 50.0) * 0.5

        else:
            if under_limit:
                reward += 0.3

        if self.battery_temp > 42.0:
            self.chiller_active = True
            chiller_cooling = min(1.0, (self.battery_temp - 40.0) * 0.1)
            
            self.battery_temp -= chiller_cooling * dt

            self.chiller_power = chiller_cooling * 500
            chiller_energy = (self.chiller_power * dt) / 3600000
            self.soc -= (chiller_energy / self.battery_capacity_kwh) * 100

        else:
            self.chiller_active = False
            self.chiller_power = 0.0

        if self.cell_temp_variance < 2.0 and under_limit:
            reward += 0.1

        elif self.cell_temp_variance > 4.0:
            reward -= 0.2

        if abs(applied_torque) > 0.8:
            self.cell_temp_variance = min(5.0, self.cell_temp_variance + 0.001)
        else:
            self.cell_temp_variance = max(0.0, self.cell_temp_variance - 0.0005)
            
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

            if self.rear_vehicle_left < 10.0 or self.rear_vehicle_right < 10.0:
                reward -= 5.0
                
            if self.adjacent_vehicle_left < 5.0 or self.adjacent_vehicle_right < 5.0:
                reward -= 10.0
                done = True
        
        else:
            self.blind_spot_warning = False

        # --------------------------------- LANE CHANGE INDICATOR ---------------------------------
        if self.swerve_active and not self.indicator_left and not self.indicator_right:
            if self.adjacent_vehicle_left > self.adjacent_vehicle_right:
                self.indicator_left = True
                self.indicator_right = False
                self.lane_change_intent = 'left'
            else:
                self.indicator_right = True
                self.indicator_left = False
                self.lane_change_intent = 'right'
            self.indicator_steps = 0
            self.indicator_reward_given = False

        if self.indicator_left or self.indicator_right:
            self.indicator_steps += 1

            if self.indicator_steps >= 3 and not self.indicator_reward_given and under_limit:
                reward += 1.0
                self.indicator_reward_given = True

            if self.swerve_active and self.indicator_steps < 2:
                reward -= 2.0

            if not self.swerve_active and self.indicator_steps > 5:
                self.indicator_left = False
                self.indicator_right = False
                self.indicator_steps = 0
                self.lane_change_intent = None

        if self.indicator_steps > 30:
            reward -= 0.5
            self.indicator_left = False
            self.indicator_right = False
            self.indicator_steps = 0
            self.lane_change_intent = None
                
         # ------------- REWARD CALCULATION -------------
        base_target  = 80.0 - self.slope_angle * 3.5 - self.extra_payload / 100.0
        base_target  = max(30.0, base_target)

        if self.drive_mode_performance == "eco":
            target_speed = min(base_target, self.speed_limit * 0.80)

        elif self.drive_mode_performance == "normal":
            target_speed = min(base_target, self.speed_limit * 0.92)

        elif self.drive_mode_performance == "sport":
            target_speed = min(base_target, self.speed_limit)

        elif self.drive_mode_performance == "track":
            target_speed = min(base_target * 1.3, 250)

        elif self.drive_mode_performance == "custom":
            # Custom target based on driver throttle sensitivity
            factor = 0.85 + (self.driver_throttle_sens - 0.5) * 0.2
            target_speed = min(base_target, self.speed_limit * np.clip(factor, 0.80, 1.10))

        min_acceptable = target_speed * 0.7

        if self.speed < min_acceptable and self.slope_angle < 8.0:
            reward -= (min_acceptable - self.speed) * 0.1

        if self.speed > 2.0:
            if self.speed > self.speed_limit:
                speed_reward = 0.0
                reward -= 2.0
            
            else:
                speed_reward = np.exp(-0.5 * ((self.speed - target_speed) / 15.0) ** 2)
        
        else:
            speed_reward = 0.0
            reward -= 0.5
        
        if hasattr(self, 'last_speed'):
            speed_delta = self.speed - self.last_speed

            if self.speed < target_speed and self.speed < self.speed_limit and self.speed < (self.speed_limit - 5.0):
                if speed_delta > 0:
                    reward += speed_delta * 0.3
                else:
                    reward -= abs(speed_delta) * 0.2
            
            elif self.speed > target_speed:
                if speed_delta < 0:
                    reward += abs(speed_delta) * 0.2

        if self.acc_enabled and self.speed > 5.0 and under_limit:
            effective_acc_target = min(self.acc_target_speed, self.speed_limit * 0.95)
            speed_error = abs(self.speed - effective_acc_target)
            speed_acc_reward = np.exp(-0.5 * (speed_error / 20.0) ** 2) * 0.8
            distance_acc_reward = np.exp(
                -0.5 * (abs(self.speed - getattr(self, 'last_speed', self.speed)) / 30.0) ** 2
            ) * 0.5
            speed_change = abs(self.speed - getattr(self, 'last_speed', self.speed))
            comfort_reward = -speed_change * 0.1
            reward += speed_acc_reward + distance_acc_reward + comfort_reward
        
        self.last_speed = self.speed
        
        slope_difficulty = 1.0 + (self.slope_angle / 12.0)
        adj_speed_reward = speed_reward * slope_difficulty
        reward += adj_speed_reward * 2.0

        effective_limit = self.speed_limit * self.mode_speed_limit_factor

        if self.mode_speed_limit_enforce:
            # Normal enforcement with mode factor:
            if self.speed > effective_limit:
                excess = self.speed - effective_limit
                reward -= excess * 10.0
                reward -= (excess ** 1.5) * 2.0

                if excess > 5.0: 
                    reward -= 30.0

                if excess > 15.0:
                    reward -= 100.0
                    done    = True

            if self.speed > self.speed_limit + 20:
                done = True

        else:
            # Track mode — NO speed limit penalty

            # Reward going fast in speed mode:
            if self.speed > self.speed_limit and under_limit is False:
                speed_bonus = np.clip((self.speed - self.speed_limit) / 50.0, 0, 1)
                reward += speed_bonus * 2.0  # Actively reward going fast 
        
        if self.speed > (self.speed_limit + 20):
            done = True
        
        ideal_ttc = 4.0
        ttc_current = 0.0
        
        if self.speed > 5.0:
            ttc_current = self.obstacle_distance / (self.speed / 3.6 + 0.0001)
            ttc_error = abs(ttc_current - ideal_ttc)
            acc_reward = np.exp(-0.5 * (ttc_error * 2.0) ** 2) * 0.5
            if under_limit:
                reward += acc_reward
        
        if ttc_current < 2.0 and self.speed > 30.0:
            reward -= (2.0 - ttc_current) * 10.0
        
        if self.weather == "rain" and torque_request > 0.7:
            reward -= (torque_request - 0.7) * 2.0

        if self.speed < 1.0 and abs(torque_request) < 0.15 and acceleration <= 0.0:
            reward -= 1.0

        on_steep_slope = self.slope_angle > 5.0 and self.extra_payload > 100.0
        if not on_steep_slope:
            if self.speed < 2.0 and abs(torque_request) > 0.1 and acceleration < 0:
                if self.slope_angle < 8.0:
                    self.stall_steps += 1

        if self.speed < 2.0 and abs(torque_request) > 0.1 and acceleration < 0:

            if self.slope_angle < 5.0:
                if self.last_speed < 5.0:  # Only count if we've been slow for a while
                    self.stall_steps += 1
                    reward -= 1.5
                    
            elif self.slope_angle < 8.0 and self.extra_payload < 200.0:
                self.stall_steps += 1
                reward -= 1.5

        else:
            self.stall_steps = max(0, self.stall_steps - 2)

        if self.slope_angle < 5.0:
            stall_limit = 20
        elif self.slope_angle < 8.0:
            stall_limit = 60
        else:
            stall_limit = 150

        if self.stall_steps > stall_limit:
            reward -= 20.0
            if self.slope_angle >= 5.0:
                self.stall_steps = 0
                reward -= 10.0
            else:
                done = True

        if self.slope_angle > 7.0:
            if torque_request > 0.5:
                if under_limit:
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

        if under_limit:
            energy_efficiency = 1.0 - (abs(self.soc - 100.0) / 100.0)
            speed_ratio = min(1.0, self.speed / (target_speed + 0.001))
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

        if under_limit:
            reward += adj_speed_reward * (thermal_efficiency - 1.0)

        thermal_penalty = 0.0

        if self.temp > 75.0:
            thermal_penalty = (self.temp - 75.0) * 1.5
            reward -= thermal_penalty
        
        if self.temp > 85.0:
            reward -= (self.temp - 85.0) * 3.0
        
        # ---------------- ATTENTION AND DISTRACTION MECHANICS ----------------
        self.attention_level = max(0.3, self.attention_level - self.attention_decay_rate) 

        if jerk > 1.5:
            self.attention_level = min(1.0, self.attention_level + 0.1)

        if self.attention_level < 0.5:
            attention_noise = np.random.normal(0, (1.0 - self.attention_level) * 0.2)
            torque_request = np.clip(torque_request + attention_noise, -1, 1)
            reward -= (0.5 - self.attention_level) 

        if self.attention_level > 0.9 and under_limit: 
            reward += 0.2

        self.last_acceleration = acceleration

        self.log_history(tc_active, abs_active, aeb_active, tpms_mult)

        done = done or (self.temp > self.max_temp or self.soc < 0 or self.airbag_deployed or self.speed > 170.0)
        return self.get_obs(), reward, done, False, {}
    
    def log_history(self, tc, abs_v, aeb, tpms):
        self.history['soc'].append(self.soc)
        self.history['temp'].append(self.temp)
        self.history['speed'].append(self.speed)
        self.history['gear'].append(self.current_gear)
        self.history['tc'].append(tc)
        self.history['abs'].append(abs_v)
        self.history['aeb'].append(aeb)
        self.history['tpms_loss'].append(tpms)
        self.history['esc'].append(self.esc_active)
        tl_enc = {"green": 1.0, "amber": 0.5, "red": 0.0}
        self.history['tl_state'].append(tl_enc.get(self.traffic_light_state, 1.0))
        self.history['swerve'].append(1 if self.swerve_active else 0)
        self.history['indicator'].append(
            1 if self.indicator_left else (-1 if self.indicator_right else 0)
        )
    
    def plot_performance(self):
        fig, axes = plt.subplots(10, 1, figsize=(14, 32))

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

        axes[4].step(range(len(self.history['tc'])),  self.history['tc'],  label='TC')
        axes[4].step(range(len(self.history['abs'])), self.history['abs'], label='ABS')
        axes[4].step(range(len(self.history['aeb'])), self.history['aeb'], label='AEB', color='red')
        axes[4].set_title("Safety Systems")

        axes[5].plot(self.history['tpms_loss'], color='orange', label='TPMS Multiplier')
        axes[5].set_title("Tire Pressure Loss")

        axes[6].step(range(len(self.history['esc'])), self.history['esc'],
                    color='purple', label='ESC')
        axes[6].set_title("ESC Activations")

        axes[7].step(range(len(self.history['tl_state'])), self.history['tl_state'],
                    color='gold', label='TL (1=G 0.5=A 0=R)')
        axes[7].set_title("Traffic Light State")

        axes[8].step(range(len(self.history['swerve'])), self.history['swerve'],
                    color='cyan', label='Swerve Active')
        axes[8].set_title("Swerve Events")

        axes[9].step(range(len(self.history['indicator'])), self.history['indicator'],
                    color='yellow', label='Indicator (1=L -1=R 0=off)')
        axes[9].set_title("Lane Change Indicator")

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

    checkpoint = CheckpointCallback(
    save_freq=300000,
    save_path="./checkpoints/",
    name_prefix="ev_p3",
    verbose=1
    )

    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.00008,
        batch_size=256,
        n_steps=4096,
        ent_coef=0.003,
        clip_range=0.15,
        clip_range_vf=0.15,
        gae_lambda=0.95,
        n_epochs=10,
        seed=42,
        normalize_advantage=True,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        device="cpu"
    )

    print("Training Phase 3 — 3M steps...")
    model.learn(total_timesteps=3_000_000, callback=checkpoint)
    model.save("ev_startup_phase3")
    print("Done — saved ev_startup_phase3.zip")
