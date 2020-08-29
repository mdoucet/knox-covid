import pandas as pd
import numpy as np

import urllib.request
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

%matplotlib notebook


from scipy import stats as sps
from scipy.interpolate import interp1d

%config InlineBackend.figure_format = 'retina'

knox_data_url = 'https://covid.knoxcountytn.gov/js/covid-charts.js'

# Get the Knox county javascript code that has the data (Design hint to Knox County coders: do mix data and code)
with urllib.request.urlopen(knox_data_url) as response:
   js_script = response.read().decode('utf-8')

# Parse the data
# Warning: this way of parsing the data might not work if Knox County changes the way they encore the data.
new_cases = []
day_number = []
for line in js_script.split('\n'):
    if 'var cumulativeNewCases' in line:
        # First replace characters that we don't need to parse the data, the split at comas
        toks = line.replace('];', '').split(',')
        # Skipt first token since it's only the variable declaration
        for t in toks[1:]:
            if t == '':
                new_cases.append(0)
            else:
                new_cases.append(int(t))
        
    elif 'var cumulativeDates' in line:
        # Skip the first tokem since it's only the variable declaration
        date_array = line.replace("'", '').replace("];", '').split(',')[1:]
        jan01 = datetime(2020, 1, 1)
        for d in date_array:
            _date = datetime.strptime(d.strip(), "%m/%d/%Y")
            _day_number = _date - jan01
            day_number.append(_day_number.days)
        
# Sanity check
assert(len(new_cases) == len(day_number))


def prepare_cases(cases, day_number, cutoff=1):
    smoothed = cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    idx_start = np.searchsorted(smoothed, cutoff)
    
    smoothed = smoothed.iloc[idx_start:]
    original = cases.loc[smoothed.index]
    _day_number = day_number[idx_start:]
    
    return original, smoothed, _day_number

original, smoothed, day_processed = prepare_cases(pd.Series(new_cases), day_number, cutoff=1)

plt.figure()
plt.plot(day_processed, original, '-', alpha=0.5, label='new cases')
plt.plot(day_processed, smoothed, '-', label='7-day average')
plt.legend()

jan01 = datetime(2020, 1, 1)
_today = (datetime.today()-jan01).days
plt.xlabel("Day [today is day %s]" % _today)
plt.ylabel("New cases")
plt.grid(True, linestyle='--')

GAMMA = 1/7
# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

def highest_density_interval(pmf, p=.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])


def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood

# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)

# Note that this takes a while to execute - it's not the most efficient algorithm
hdis = highest_density_interval(posteriors, p=.67)

most_likely = posteriors.idxmax().rename('ML')

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

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
