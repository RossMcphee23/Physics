import numpy as np
import matplotlib.pyplot as plt

def runge_kutta(f, y0, t, h):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(1, n):
        k1 = h * f(t[i - 1], y[i - 1])
        k2 = h * f(t[i - 1] + 0.5 * h, y[i - 1] + 0.5 * k1)
        k3 = h * f(t[i - 1] + 0.5 * h, y[i - 1] + 0.5 * k2)
        k4 = h * f(t[i - 1] + h, y[i - 1] + k3)
        
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        # this is to cut the graph off at ground level
        if y[i, 0] <= 0:
            break
    
    return y

def falling_ball_with_resistance(t, y):
    g = -9.8  # Acceleration due to gravity (m/s^2)
    b = 0.2  # Damping coefficient for air resistance
    m = 0.5  # Mass of the ball (kg)
    
    x, v = y
    dxdt = v
    dvdt = g - (b / m) * v
    
    # Differential equation representing the motion of a falling ball with air resistance
    # y: [x, v] where x is the position and v is the velocity
    return np.array([dxdt, dvdt])

def falling_ball_without_resistance(t, y):
    g = -9.8  # Acceleration due to gravity (m/s^2)
    
    x, v = y
    dxdt = v
    dvdt = g  # No air resistance
    
    # Differential equation representing the motion of a falling ball without air resistance
    # y: [x, v] where x is the position and v is the velocity
    return np.array([dxdt, dvdt])

# Set up initial conditions
initial_conditions = [0, 0]  # Initial position and velocity
initial_height = 100  
initial_velocity = 0  
initial_conditions = np.array([initial_height, initial_velocity]) 

# Set up time array
total_time = 7
time_step = 0.01  
time_array = np.arange(0, total_time, time_step)

# Run the simulation with air resistance
result_with_resistance = runge_kutta(falling_ball_with_resistance, initial_conditions, time_array, time_step)
positions_with_resistance = result_with_resistance[:, 0]
velocities_with_resistance = result_with_resistance[:, 1]

ground_index_with_resistance = np.argmax(positions_with_resistance <= 0)
velocities_with_resistance[ground_index_with_resistance:] = 0

# Run the simulation without air resistance
result_without_resistance = runge_kutta(falling_ball_without_resistance, initial_conditions, time_array, time_step)
positions_without_resistance = result_without_resistance[:, 0]
velocities_without_resistance = result_without_resistance[:, 1]

ground_index_without_resistance = np.argmax(positions_without_resistance <= 0)

# Analytical solutions
analytical_positions_with_resistance = initial_height + initial_velocity * time_array - 0.5 * 9.8 * time_array**2
analytical_velocities_with_resistance = initial_velocity - 9.8 * time_array

analytical_positions_without_resistance = initial_height + initial_velocity * time_array - 0.5 * 9.8 * time_array**2
analytical_velocities_without_resistance = initial_velocity - 9.8 * time_array

# Plot Position and Velocity vs. Time (with air resistance)
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time_array[:ground_index_with_resistance], positions_with_resistance[:ground_index_with_resistance], label="Numerical Position (Air Resistance)", color='blue')
plt.plot(time_array, analytical_positions_with_resistance, label="Analytical Position (Air Resistance)", linestyle='--', color='blue')
plt.axhline(0, color='black', linestyle='--', label="Ground Level")
plt.scatter(time_array[ground_index_with_resistance], 0, color='red', marker='o', label="Ground Impact (Air Resistance)")
plt.annotate(r'$\frac{d^2x}{dt^2} = g - \frac{b}{m} \frac{dx}{dt}$', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=12, color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Position (meters)")
plt.title("Falling Ball Simulation - Position vs. Time (with Air Resistance)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_array[:ground_index_with_resistance], velocities_with_resistance[:ground_index_with_resistance], label="Numerical Velocity (Air Resistance)", color="orange")
plt.plot(time_array, analytical_velocities_with_resistance, label="Analytical Velocity (Air Resistance)", linestyle='--', color="orange")
plt.axhline(0, color='red', linestyle='--', label="Zero Velocity (Air Resistance)")
plt.annotate(r'$\frac{d^2x}{dt^2} = g - \frac{b}{m} \frac{dx}{dt}$', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=12, color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Falling Ball Simulation - Velocity vs. Time (with Air Resistance)")
plt.legend()

plt.tight_layout()
plt.show()
