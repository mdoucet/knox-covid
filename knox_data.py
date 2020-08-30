"""
    Get and prepare Knox County data
"""
import urllib.request
from datetime import datetime


KNOX_DATA_URL = 'https://covid.knoxcountytn.gov/js/covid-charts.js'


def get_data():
    """
        Get the Knox county javascript code that has the data
        (Design hint to Knox County coders: do mix data and code)
    """
    with urllib.request.urlopen(KNOX_DATA_URL) as response:
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

    return day_number, new_cases
