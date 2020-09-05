import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

import r_0
import knox_data

# Get data from Knox County site
day_number, new_cases = knox_data.get_data()

# Prepare data
original, smoothed, day_processed = r_0.prepare_cases(pd.Series(new_cases), day_number, cutoff=1)

# Make plot for news cases per day
plt.figure()
plt.step(day_processed, original, '-', alpha=0.5, label='new cases')
plt.plot(day_processed, smoothed, '-', label='7-day average')
plt.legend()

jan01 = datetime(2020, 1, 1)
_today = (datetime.today()-jan01).days
plt.xlabel("Day [today is day %s]" % _today)
plt.ylabel("New cases")
plt.grid(True, linestyle='--')
plt.savefig('knox_cases.png', dpi=200, facecolor='w', edgecolor='w')

# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = r_0.get_posteriors(smoothed, sigma=.25)

# Note that this takes a while to execute - it's not the most efficient algorithm
hdis = r_0.highest_density_interval(posteriors, p=.67)

most_likely = posteriors.idxmax().rename('ML')

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

# Max number of days reported
n_days=60

fig, ax = plt.subplots()
plt.plot(day_processed[-n_days:],np.asarray(result['ML'])[-n_days:])
ax.fill_between(day_processed[-n_days:], np.asarray(result['Low_67'])[-n_days:],
                np.asarray(result['High_67'])[-n_days:],
                color='k', alpha=0.1, lw=0,
                zorder=3)

jan01 = datetime(2020, 1, 1)
_today = (datetime.today()-jan01).days
plt.xlabel("Day [today is day %s]" % _today)
plt.ylabel("$R_0$")
plt.grid(True, linestyle='--')
plt.title('$R_0$ for the past %s days' % n_days)
plt.savefig('knox_r_0.png', dpi=200, facecolor='w', edgecolor='w')
