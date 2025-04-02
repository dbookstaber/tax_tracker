# TaxTracker

TaxTracker is a Python module designed to calculate the United States federal tax treatment of trading activity and distributions.  It determines whether capital gains/losses are short- or long-term, and applies the wash loss rule when applicable.  It can determine when dividends become qualified, and whether payments-in-lieu of distributions qualify as investment expenses.

## Features

- **Capital Gains Tracking**: Characterizes capital gains and losses from trading activity, applying wash sale rules.
- **Dividend Qualification**: Handles dividend qualification rules for regular and preferred dividends.
- **Daily P&L and NAV Calculation**: Given price data, it can compute daily P&L and NAV of a portfolio.
- **Detailed Reporting**: Generates both summaries and details of tax lot activity and cashflows.
- **Compliance with IRS Rules**: Implements tax rules as described in [IRS Publication 550 (2024)](https://www.irs.gov/publications/p550).

## Limitations

- This is done to the best of our ability to understand/interpret [IRS Publication 550](https://www.irs.gov/pub/irs-pdf/p550.pdf).
- This does not handle accounting of derivatives or other assets with special tax treatment.
- This does not look for, or account for, "substantially identical" assets.  If you enter trades for asset `XYZ`, and also for `XYZ.B`, it treats them as separate even if the IRS might consider them "substantially identical."

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/dbookstaber/tax_tracker.git
cd tax_tracker
```

Ensure you have Python 3.10 or higher installed. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Documentation

[The code contains detailed in-line documentation](tax_tracker/tax_tracker.py), which has been PDoc'd in [/docs](/docs).

[Examples.ipynb](Examples.ipynb) provides many annotated examples of each of the tracker classes:
* CapGainsTracker
* DistributionTracker
* PNLtracker

## Usage

### Example: Capital Gains

```python
import pandas as pd
from tax_tracker import CapGainsTracker

# Initialize the tracker
tracker = CapGainsTracker()

# Record trades
tracker.trade(pd.Timestamp('2023-01-01'), 'XYZ', 100, 150.0)  # Buy 100 shares of XYZ at $150
tracker.trade(pd.Timestamp('2023-02-01'), 'XYZ', -50, 160.0)  # Sell 50 shares of XYZ at $160

# View open positions
print('Open Lots:\n' + tracker.open_lots_str)

# View closed trades
print('\nClosed Lots:\n' + tracker.closed_lots_str)

# View capital gains
print('\nCapital Gains:\n' + str(tracker.capital_gains_df))
```

### Example: Distributions

```python
import pandas as pd
from tax_tracker import DistributionTracker

# Example distribution data
distribution_data = pd.DataFrame({
    'Ticker': ['ABC', 'ABC', 'ABC'],
    'Date': [pd.Timestamp('2023-02-01'), pd.Timestamp('2023-03-01'), pd.Timestamp('2023-03-15')],
    'Type': ['Dividend', 'Return of Capital', 'Preferred Dividend'],
    'Distribution': [0.22, 0.15, 0.30]
}).set_index(['Ticker', 'Date', 'Type'])

# Initialize the tracker
tracker = DistributionTracker(distribution_data)

# Record trades
tracker.trade(pd.Timestamp('2023-01-01'), 'ABC', 100, 150.0)

# Carry trade through April 1
tracker.run_beginning_of_day(pd.Timestamp('2023-04-01'))

# Show dividend characterization as of April 1
print(tracker.dividends_df)
```

### Example: Daily P&L and NAV

```python
import pandas as pd
from tax_tracker import PNLtracker

# Example price data
price_data = pd.DataFrame({
    'Ticker': ['XYZ'] * 5,
    'Date': pd.date_range(start='2025-03-03', end='2025-03-07', freq='D'),
    'Px': [15.0, 16.0, 17.0, 18.0, 19.0]
}).set_index(['Ticker', 'Date'])

# Initialize the tracker
tracker = PNLtracker(price_data)

# Record trades
tracker.trade(pd.Timestamp('2025-03-03'), 'XYZ', 100, 15.1)
tracker.trade(pd.Timestamp('2025-03-05'), 'XYZ', -50, 16.9)

# Carry trade through March 7
tracker.run_end_of_day(pd.Timestamp('2025-03-07'))

# View daily P&L and NAV
print(tracker.daily_returns_df)
```


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear messages.
4. Submit a pull request.

## License

This project is licensed under the Prosperity Public License. See the [LICENSE](LICENSE.md) file for details.
