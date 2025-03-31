"""Tests for DistributionTracker classes"""
import warnings
from typing import Callable
import pytest
import pandas as pd
from tax_tracker import DistributionTracker, PNLtracker

@pytest.mark.usefixtures("distribution_data", "price_data")
class TestDistributions:
    """Tests for DistributionTracker"""

    def test_distribution_tracker(self, distribution_data):
        """Runs tests on DistributionTracker"""
        self.run_tests_with_data(lambda: DistributionTracker(distribution_data))
        self.run_tests_no_data(lambda: DistributionTracker(None))

    def test_pnl_tracker(self, price_data, distribution_data):
        """Runs tests on PNLtracker"""
        warnings.filterwarnings("ignore", message="No price data for*")
        self.run_tests_with_data(lambda: PNLtracker(price_data, distribution_data))
        self.run_tests_no_data(lambda: PNLtracker(price_data, None))

    def run_tests_with_data(self, tracker_class: Callable[[], DistributionTracker]):
        """Runs tests on a DistributionTracker object"""
        self.unqualified(tracker_class)
        self.qualified(tracker_class)
        self.exempt(tracker_class)
        self.payments_in_lieu(tracker_class)
        self.two_positions(tracker_class)
        self.preferred(tracker_class)
        self.wash_loss_logic(tracker_class)

    def run_tests_no_data(self, tracker_class: Callable[[], DistributionTracker]):
        """Runs tests on a DistributionTracker object without distribution data"""
        self.preferred(tracker_class)
        self.cap_gains_distributions(tracker_class)
        self.example6(tracker_class)

    def unqualified(self, tracker_class: Callable[[], DistributionTracker]):
        """Buy and hold RA for 60 days.  Ref Example 2.1"""
        tracker = tracker_class()
        start_date = pd.Timestamp('2022-01-05')
        tracker.trade(start_date, 'RA', shares=1, price=21)
        end_date = start_date + pd.DateOffset(days=60)
        tracker.trade(end_date, 'RA', -1, price=24)
        # Confirm dividends are Regular
        divs = tracker.dividends_all_df
        assert divs.sum()['Regular'] > 0, 'Regular Dividends'
        assert divs.sum()['Qualified'] == 0, 'Qualified Dividends'
        assert len(tracker.closed_lots_list) == 1, 'Number of Complete Trades'
        # We don't have any holdings after the end date
        assert len(tracker.positions['RA']) == 0, 'No ending position in RA'
        # We should have (end_date - start_date).days days of holdings
        assert tracker.holdings[tracker.holdings != 0]['RA'].count() == (end_date - start_date).days
        # Confirm price return matches ST capital gains
        trade = tracker.closed_lots_list[0]
        assert list(tracker.capital_gains.values())[0]['ShortTerm'] == pytest.approx(
                    trade.tax_gain, rel=1e-6), 'Capital Gains'

    def qualified(self, tracker_class: Callable[[], DistributionTracker]):
        """Buy and hold RA for 61 days.  Ref Example 2.1"""
        tracker = tracker_class()
        start_date = pd.Timestamp('2022-01-05')
        tracker.trade(start_date, 'RA', shares=1, price=21)
        end_date = start_date + pd.DateOffset(days=61)
        tracker.trade(end_date, 'RA', -1, price=24)
        # Confirm dividends are Qualified
        divs = tracker.dividends_all_df
        assert divs.sum()['Regular'] == 0, 'Regular Dividends'
        assert divs.sum()['Qualified'] > 0, 'Qualified Dividends'

    def exempt(self, tracker_class: Callable[[], DistributionTracker]):
        """Buy and hold NEA.  Its dividends are Exempt,
            so holding period has not effect on their tax treatment."""
        tracker = tracker_class()
        start_date = pd.Timestamp('2022-01-05')
        tracker.trade(start_date, 'NEA', shares=2, price=11)
        tracker.trade(start_date + pd.DateOffset(days=60), 'NEA', -1, price=11)
        tracker.trade(start_date + pd.DateOffset(days=366), 'NEA', -1, price=11)
        # Confirm dividends are Exempt
        divs = tracker.dividends_all_df
        assert divs.sum()['Regular'] == 0, 'Regular Dividends'
        assert divs.sum()['Qualified'] == 0, 'Qualified Dividends'
        assert divs.sum()['Exempt'] > 0, 'Exempt Dividends'

    def payments_in_lieu(self, tracker_class: Callable[[], DistributionTracker]):
        """Short NEA; check handling of payments-in-lieu.  Ref Example 2.2"""
        tracker = tracker_class()
        start_date = pd.Timestamp('2022-01-05')
        tracker.trade(start_date, 'NEA', shares=-2, price=11)
        cover_date = start_date + pd.DateOffset(days=45)
        tracker.trade(cover_date, 'NEA', 1, price=11)
        # Confirm PIL up to 45 days of holding has not moved to Expenses
        divs_sum = tracker.dividends_all_df.sum()
        assert divs_sum['Expense'] == 0, 'No PIL Expenses'
        assert divs_sum['PayInLieu'] < 0, 'PILs recorded'
        # Cover second share after more than 45 days so its PIL move to Expenses
        tracker.trade(cover_date + pd.Timedelta(days=1), 'NEA', 1, price=11)
        # We should have equal PIL in PayInLieu and Expenses
        for divs in tracker.daily_dividends.values():
            assert divs['PayInLieu'] == divs['Expense'], 'PIL correct'

    def two_positions(self, tracker_class: Callable[[], DistributionTracker]):
        """2 positions, one held for only 60 days.  Ref Example 2.3"""
        tracker = tracker_class()
        start_date = pd.Timestamp('2022-01-05')
        tracker.BoD(start_date)
        tracker.buy(start_date, 'RA', shares=1, price=21)
        second_buy_date = start_date + pd.Timedelta(days=60)
        tracker.buy(second_buy_date, 'RA', shares=1, price=21)
        tracker.sell(second_buy_date + pd.Timedelta(days=60), 'RA', shares_to_sell=1, price=21)
        # Run the tracker until July 1, 2022
        tracker.BoD(pd.Timestamp('2022-07-01'))
        divs = tracker.dividends_all_df
        assert divs.sum()['Regular'] == pytest.approx(0.398033, rel=1e-6), 'Regular Dividends'
        assert divs.sum()['Qualified'] == pytest.approx(1.194029, rel=1e-6), 'Qualified Dividends'

    def preferred(self, tracker_class: Callable[[], DistributionTracker]):
        """Test preferred dividend qualification logic.  Ref Example 2.5"""
        # Only 2/3 of dividends are qualifiable, and they only qualify after 90 days of holding.
        start_date = pd.Timestamp('2025-01-02')
        qualified_date = start_date + pd.Timedelta(days=91)
        print(f'Preferred qualified_date: {qualified_date}')
        syn_data = {
            'Ticker': ['ABC', 'ABC'],
            'Date': ['2025-02-01', '2025-02-01'],
            'Type': ['P', 'N'],
            'Distribution': [0.10, 0.05]
        }
        test_data = pd.DataFrame(syn_data)
        test_data['Date'] = pd.to_datetime(test_data['Date'])
        test_data.set_index(['Ticker', 'Date', 'Type'], inplace=True)
        tracker = tracker_class()
        tracker.distributions = test_data
        tracker = DistributionTracker(test_data)
        tracker.buy(start_date, 'ABC', shares=1, price=10)
        pre_qualification_date = qualified_date - pd.Timedelta(days=1)
        tracker.BoD(pre_qualification_date)
        assert tracker.dividends_df.iloc[0].Regular == pytest.approx(
                    0.15, rel=1e-6), 'Regular dividends before 90 days'
        tracker.BoD(qualified_date)
        assert tracker.dividends_df.iloc[0].Regular == pytest.approx(
                    0.05, rel=1e-6), 'One third of dividend is Regular'
        assert tracker.dividends_df.iloc[0].Qualified == pytest.approx(
                    0.1, rel=1e-6), 'Two-thirds of dividend is Qualified'

    def cap_gains_distributions(self, tracker_class: Callable[[], DistributionTracker]):
        """Test capital gains distributions.  Ref Example 2.7"""
        start_date = pd.Timestamp('2025-01-02')
        syn_data = {
            'Ticker': ['ABC', 'ABC', 'ABC'],
            'Date': ['2025-02-01', '2025-02-01', '2025-02-01'],
            'Type': ['L', 'S', 'R'],
            'Distribution': [0.10, 0.05, 0.03]
        }
        test_data = pd.DataFrame(syn_data)
        test_data['Date'] = pd.to_datetime(test_data['Date'])
        test_data.set_index(['Ticker', 'Date', 'Type'], inplace=True)
        tracker = tracker_class()
        tracker.distributions = test_data
        tracker = DistributionTracker(test_data)
        tracker.buy(start_date, 'ABC', shares=1, price=10)
        after_div_date = start_date + pd.DateOffset(months=1)
        tracker.BoD(after_div_date)
        summary = tracker.taxable_cashflows_df
        assert len(summary.columns) == 3, 'Cashflow categories'
        assert summary.sum().sum() == pytest.approx(0.18, rel=1e-6), 'Total cashflows'

    def example6(self, tracker_class: Callable[[], DistributionTracker]):
        """Example 2.6"""
        tracker = tracker_class()
        tracker.trade(pd.Timestamp('2025-04-27'), 'XYZ', 10_000, price=1)
        tracker.receive_distribution(pd.Timestamp('2025-05-02'), 'XYZ', 0.09, 'D')
        tracker.receive_distribution(pd.Timestamp('2025-05-02'), 'XYZ', 0.09, 'N')
        tracker.trade(pd.Timestamp('2025-06-15'), 'XYZ', -2_000, price=1)
        tracker.BoD(pd.Timestamp('2025-07-01'))
        assert tracker.dividends_df.sum()['Regular'] == pytest.approx(
                    1080.0, rel=1e-6), 'Regular Dividends'
        assert tracker.dividends_df.sum()['Qualified'] == pytest.approx(
                    720.0, rel=1e-6), 'Qualified Dividends'

    def wash_loss_logic(self, tracker_class: Callable[[], DistributionTracker]):
        """Test wash sale logic"""
        start_date = pd.Timestamp('2022-01-05')
        tracker = tracker_class()

        # Buy 10 shares of RA
        tracker.buy(start_date, 'RA', shares=10, price=21)
        # Sell 5 shares of RA at a loss
        sell_date = start_date + pd.Timedelta(days=29)
        tracker.sell(sell_date, 'RA', shares_to_sell=5, price=18)
        # Buy 5 shares of RA within 30 days to trigger wash sale
        buy_back_date = sell_date + pd.Timedelta(days=25)
        tracker.buy(buy_back_date, 'RA', shares=5, price=20)

        #region Functionality checks
        assert tracker.get_position('RA') == 10, "Ending position in RA should be 10 shares"
        # Check if the wash sale was recorded correctly
        assert len(tracker.wash_trades) == 1, "Should have one wash sale recorded"
        wash_sale = tracker.wash_trades[0]
        assert wash_sale.trade.ticker == 'RA', "Ticker should be RA"
        assert wash_sale.washed_quantity == 5, "Wash sale quantity should be 5"
        # Check if the loss was reversed
        assert tracker.capital_gains[sell_date]['ShortTerm'] == 0, \
            "Short-term capital gains should be adjusted for wash sale"
        #endregion

if __name__ == '__main__':
    pytest.main()
