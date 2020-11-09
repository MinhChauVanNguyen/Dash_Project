def label_code(row):
    if row['State'] == 'Oregon':
        return 'OR'
    elif row['State'] == 'California':
        return 'CA'
    elif row['State'] == 'Washington':
        return 'WA'
    elif row['State'] == 'Kentucky':
        return 'KY'
    elif row['State'] == 'Texas':
        return 'TX'
    elif row['State'] == 'New York':
        return 'NY'
    elif row['State'] == 'Florida':
        return 'FL'
    elif row['State'] == 'Illinois':
        return 'IL'
    elif row['State'] == 'South Carolina':
        return 'SC'
    elif row['State'] == 'North Carolina':
        return 'NC'
    elif row['State'] == 'Georgia':
        return 'GA'
    elif row['State'] == 'Virginia':
        return 'VA'
    elif row['State'] == 'Ohio':
        return 'OH'
    elif row['State'] == 'Wyoming':
        return 'WY'
    elif row['State'] == 'Missouri':
        return 'MO'
    elif row['State'] == 'Montana':
        return 'MT'
    elif row['State'] == 'Utah':
        return 'UT'
    elif row['State'] == 'Minnesota':
        return 'MN'
    elif row['State'] == 'Mississippi':
        return 'MS'
    elif row['State'] == 'Arizona':
        return 'AZ'
    elif row['State'] == 'Alabama':
        return 'AL'
    else:
        return 'MA'


