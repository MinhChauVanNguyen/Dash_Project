def label_code(row):
    if row['id'] == 'Oregon':
        return 'OR'
    elif row['id'] == 'California':
        return 'CA'
    elif row['id'] == 'Washington':
        return 'WA'
    elif row['id'] == 'Kentucky':
        return 'KY'
    elif row['id'] == 'Texas':
        return 'TX'
    elif row['id'] == 'New York':
        return 'NY'
    elif row['id'] == 'Florida':
        return 'FL'
    elif row['id'] == 'Illinois':
        return 'IL'
    elif row['id'] == 'South Carolina':
        return 'SC'
    elif row['id'] == 'North Carolina':
        return 'NC'
    elif row['id'] == 'Georgia':
        return 'GA'
    elif row['id'] == 'Virginia':
        return 'VA'
    elif row['id'] == 'Ohio':
        return 'OH'
    elif row['id'] == 'Wyoming':
        return 'WY'
    elif row['id'] == 'Missouri':
        return 'MO'
    elif row['id'] == 'Montana':
        return 'MT'
    elif row['id'] == 'Utah':
        return 'UT'
    elif row['id'] == 'Minnesota':
        return 'MN'
    elif row['id'] == 'Mississippi':
        return 'MS'
    elif row['id'] == 'Arizona':
        return 'AZ'
    elif row['id'] == 'Alabama':
        return 'AL'
    else:
        return 'MA'


def label_state(country):
    if country == 'France':
        return "Select Department"
    elif country == 'Australia':
        return "Select Region"
    elif country == 'Canada':
        return "Select Province"
    else:
        return "Select State"