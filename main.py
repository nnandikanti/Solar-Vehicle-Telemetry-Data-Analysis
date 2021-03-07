import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from scipy.signal import savgol_filter
from scipy import stats

filepath = "data/telemetry.csv"
dataSet = pd.read_csv(filepath, index_col="time")

# constants

wheelRadius = 0.27
carMass = 815 * constants.lb
carWeight = carMass * constants.g
fluidDensity = 1.255
crossecA = 1
# idk real cross sectional area

# time = dataSet["time"]
Time = dataSet.index
voltage = dataSet["Left_Bus_Voltage"]
current = dataSet["Left_Bus_Current"]
power = voltage * current
rpm =(dataSet["Left_Wavesculptor_RPM"] + dataSet["Right_Wavesculptor_RPM"]) / 2
speed = (2 * np.pi * wheelRadius * rpm) / 60
acceleration = dataSet["Raw_Acceleration_X"] 
# not sure which accel values to use
angleDeg = dataSet["Terrain_Angle"]
angle = angleDeg * np.pi / 180





# fig, axs = plt.subplots(2)  -------> uncomment to see raw speed and power vs time
# fig.suptitle('speed and power plots')
# axs[0].plot(speed)
# axs[0].set_ylabel("speed")
# axs[1].plot(power)
# axs[1].set_ylabel("power")
# axs[1].set_xlabel("time")

speedSmooth = savgol_filter(speed, 51, 3)
speedSmooth = pd.Series(speedSmooth, index=speed.index)
powerSmooth = savgol_filter(power, 51, 3)
powerSmooth = pd.Series(powerSmooth, index=power.index)

fig, axs = plt.subplots(2)
fig.suptitle('smoothed speed and power plots')
axs[0].plot(speedSmooth)
axs[0].set_ylabel("speed")
axs[1].plot(powerSmooth)
axs[1].set_ylabel("power")
axs[1].set_xlabel("time")

force = powerSmooth / speedSmooth
mu = force / carWeight

df = pd.DataFrame({"speedSmooth": speedSmooth, "powerSmooth": powerSmooth, "mu": mu}, speedSmooth.index)
# df.plot(x="speedSmooth", y="mu", style="o") -------> uncomment to see mu vs speed plot (force relationship)

plt.show()

timeStep = 250
# time interval between each data point
setsNum = 10
# number of data points to use for regression
startTime = 1600552550000
# starting point in csv

iterator = 0
low2I = startTime
high2I = startTime + timeStep * (setsNum - 1)
# initial values of high and low based on startTime, number of points to use, and time intervals
lowFinal = 0
highFinal = 0
minStd = 100
# arbitrary, high initial value (to be replaced on first iteration)

while iterator < len(df) - setsNum:
    low2 = low2I + iterator * timeStep
    high2 = high2I + iterator * timeStep
    # moves data set along by one step each time loop iterates
    newDf = df.loc[low2:high2]
    newStd = (newDf["speedSmooth"].std() + newDf["powerSmooth"].std()) / 2
    # calculates standard deviation of current data set
    if newStd < minStd:
        minStd = newStd
        lowFinal = low2
        highFinal = high2
        # if new minimum stddev is found, replaces std value and reassigns range
    iterator = iterator + 1

print(f"Range of flattest region is {lowFinal} to {highFinal}")

filtered = df.loc[lowFinal:highFinal]
# accesses flattest range

slope, intercept, r, _, _, = stats.linregress(filtered.speedSmooth, filtered.mu)
crr1 = intercept
crr2 = slope
# runs linear regression on variables in optimal range to find crr1 and crr2

print(f"crr1 = {intercept}, crr2={slope}, rSquared = {r * r}")

# for cDA subtract power due to friction from total power  = PowerT -Power(crr1,crr2) =  F(drag) * velocity
# run linregress on cDa values to find actual


powerAccel = carMass * acceleration * speed
powerHills = carMass * constants.g * np.sin(angle) * speed
powerFriction = carMass * constants.g * (crr1 + crr2 * speed) * np.cos(angle) * speed

powerDrag = power - (powerAccel + powerHills + powerFriction)
dragForce = powerDrag / speed
cD = (2 * dragForce) / (fluidDensity * speed * speed * crossecA)
CDa = np.mean(cD)
print(f"Calculated coefficient of drag is {CDa}")

