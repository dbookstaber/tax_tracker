"""Tests for CapGainsTracker classes"""
import warnings
from typing import Callable
import pytest
import pandas as pd
from tax_tracker import Config, TaxLotSelection, CapGainsTracker, DistributionTracker, PNLtracker

@pytest.mark.usefixtures("distribution_data", "price_data")
class TestCapGains:
    """Tests for CapGainsTracker"""

    def test_cap_gains_tracker(self):
        """Runs tests on CapGainsTracker"""
        self.run_tests(CapGainsTracker)

    def test_distribution_tracker(self, distribution_data):
        """Runs tests on DistributionTracker"""
        self.run_tests(lambda: DistributionTracker(distribution_data))

    def test_pnl_tracker(self, price_data, distribution_data):
        """Runs tests on PNLtracker"""
        warnings.filterwarnings("ignore", message="No price data*")
        self.run_tests(lambda: PNLtracker(price_data, distribution_data))

    def run_tests(self, tracker_class: Callable[[], CapGainsTracker]):
        """Runs tests on a CapGainsTracker object"""
        self.wash_loss_sequence(tracker_class)
        self.no_wash_after_30_days(tracker_class)
        self.no_wash_on_gains(tracker_class)
        self.switch_sides(tracker_class)
        self.two_positions(tracker_class)
        self.multiple_repurchases(tracker_class)
        self.multiple_repo_buy_sell(tracker_class)
        self.wash_loss_plus_gain(tracker_class)
        self.zero_trades(tracker_class)
        self.prior_purchase_wash(tracker_class)
        self.prior_purchase_partial_wash(tracker_class)
        self.prior_short_partial_wash(tracker_class)
        self.lot_selection(tracker_class)

    def wash_loss_sequence(self, tracker_class: Callable[[], CapGainsTracker]):
        """Test wash loss logic.  Ref Example 1.1"""
        tracker = tracker_class()

        # Buy 10 shares of ABC at $10
        start_date = pd.Timestamp('2022-01-05')
        tracker.buy(start_date, 'ABC', shares=10, price=10)
        assert len(tracker.open_lots_str) > 0, "Open lots string"
        # Sell 5 shares of ABC at $9, realizing capital loss of $1/share
        sell_date = start_date + pd.Timedelta(days=1)
        tracker.sell(sell_date, 'ABC', shares_to_sell=5, price=9)
        assert tracker.capital_gains[sell_date]['ShortTerm'] == -5.0, "Realized -$5 ST gain"
        # Buy 5 shares of ABC within 30 days to trigger wash sale rule
        buy_back_date = sell_date + pd.Timedelta(days=25)
        tracker.buy(buy_back_date, 'ABC', shares=5, price=10)
        tracker.validate()
        # Verify wash sale recorded correctly
        assert len(tracker.wash_trades) == 1, "Wash sale recorded"
        wash_sale = tracker.wash_trades[0]
        assert wash_sale.washed_quantity == 5, "Wash sale quantity"
        assert len(tracker.washed_lots_str) > 0, "Wash sale string"
        # Verify basis added to replacement shares
        replacement = tracker.positions['ABC'][-1]
        assert replacement.basis_add == -5, "Replacement basis"
        assert replacement.shares == 5, "Replacement shares"
        assert replacement.basis_days == (sell_date - start_date).days, "Replacement basis days"
        assert tracker.capital_gains[sell_date]['ShortTerm'] == 0, "Capital loss washed"

        # Just over a year later, sell all shares
        second_sell_date = start_date + pd.DateOffset(days=367)
        tracker.sell(second_sell_date, 'ABC', shares_to_sell=10, price=8)
        assert not tracker.positions['ABC'], "No open positions"
        assert tracker.capital_gains[second_sell_date]['LongTerm'] == -10, "Realized -$10 LT"
        assert tracker.capital_gains[second_sell_date]['ShortTerm'] == -15, "Realized -$15 ST"
        # Buy 10 shares back within 30 days to trigger wash sale
        second_buy_back_date = second_sell_date + pd.Timedelta(days=25)
        tracker.buy(second_buy_back_date, 'ABC', shares=10, price=7)
        tracker.validate()
        assert tracker.capital_gains[second_sell_date]['LongTerm'] == 0, "LT Loss washed"
        assert tracker.capital_gains[second_sell_date]['ShortTerm'] == 0, "ST Loss washed"
        # Verify this creates two lots (positions),
        #   one with a -15 ST loss and one with a -10 LT loss
        assert len(tracker.positions['ABC']) == 2, "Two positions should exist for ABC"
        for position in tracker.positions['ABC']:
            if position.basis_days == (second_sell_date - start_date).days:
                assert position.basis_add == -10, "One position carries -10 LT basis"
            elif position.basis_days == ((second_sell_date - buy_back_date).days
                                        + (sell_date - start_date).days):
                assert position.basis_add == -15, "One position carries -15 ST basis"
            else:
                pytest.fail("Incorrect basis on second wash")

        # Now sell everything and check final accounting
        date_out = second_buy_back_date + pd.Timedelta(days=1)
        tracker.sell(date_out, 'ABC', shares_to_sell=10, price=8.5)
        assert not tracker.positions['ABC'], "No open positions"
        assert tracker.capital_gains[date_out]['LongTerm'] == -2.5, "Final LT gain"
        assert tracker.capital_gains[date_out]['ShortTerm'] == -7.5, "Final ST gain"

    def no_wash_after_30_days(self, tracker_class: Callable[[], CapGainsTracker]):
        """Test loss not washed if replacement shares repurchased after 30 days.  Ref Example 1.2"""
        tracker = tracker_class()

        # Buy 10 shares of ABC at $10
        start_date = pd.Timestamp('2022-01-05')
        tracker.buy(start_date, 'ABC', shares=10, price=10)
        # Sell 5 shares of ABC at $9, realizing capital loss of $1/share
        sell_date = start_date + pd.Timedelta(days=1)
        tracker.sell(sell_date, 'ABC', shares_to_sell=5, price=9)
        assert tracker.capital_gains[sell_date]['ShortTerm'] == -5.0, "Realized -$5 ST gain"
        # Buy 5 shares of ABC more than 30 days after to avoid wash sale rule
        buy_back_date = sell_date + pd.Timedelta(days=35)
        tracker.buy(buy_back_date, 'ABC', shares=5, price=10)
        assert tracker.capital_gains[sell_date]['ShortTerm'] == -5.0, "ST loss remains"
        assert len(tracker.positions['ABC']) == 2, "2 tax lots"
        assert len(tracker.wash_trades) == 0, "No wash trades"
        tracker.validate()
        # Just over a year later, sell all shares
        second_sell_date = start_date + pd.DateOffset(days=367)
        tracker.sell(second_sell_date, 'ABC', shares_to_sell=10, price=8)
        assert not tracker.positions['ABC'], "No open positions"
        assert tracker.capital_gains[second_sell_date]['LongTerm'] == -10, "Realized -$10 LT"
        assert tracker.capital_gains_df.sum()['ShortTerm'] == -15, "Realized -$15 ST"
        # Buy 10 shares back more than 30 days later to avoid wash sale
        second_buy_back_date = second_sell_date + pd.Timedelta(days=35)
        tracker.buy(second_buy_back_date, 'ABC', shares=10, price=7)
        assert len(tracker.positions['ABC']) == 1, "One position should exist for ABC"
        assert tracker.capital_gains[second_sell_date]['LongTerm'] == -10, "Realized LT unaffected"
        assert tracker.capital_gains_df.sum()['ShortTerm'] == -15, "Realized -$15 ST unaffected"
        tracker.validate()
        # Now sell everything and check final accounting
        date_out = second_buy_back_date + pd.Timedelta(days=1)
        tracker.sell(date_out, 'ABC', shares_to_sell=10, price=8.5)
        assert not tracker.positions['ABC'], "No open positions"
        assert tracker.capital_gains_df.sum()['LongTerm'] == -10, "Final LT gain"
        assert tracker.capital_gains_df.sum()['ShortTerm'] == 0, "Final ST gain"

    def no_wash_on_gains(self, tracker_class: Callable[[], CapGainsTracker]):
        """Test wash logic when there is no capital loss.  Ref Example 1.3"""
        tracker = tracker_class()

        # Short 10 shares of ABC at $10
        start_date = pd.Timestamp('2022-01-05')
        tracker.trade(start_date, 'ABC', shares=-10, price=10)
        # Cover 5 shares of ABC at $9, realizing capital gain of $1/share
        cover_date = start_date + pd.Timedelta(days=1)
        tracker.trade(cover_date, 'ABC', 5, price=9)
        assert tracker.capital_gains[cover_date]['ShortTerm'] == 5.0, "Realized $5 ST gain"
        # Short another 5 shares of ABC; no effect on capital gains
        second_short_date = cover_date + pd.Timedelta(days=25)
        tracker.trade(second_short_date, 'ABC', shares=-5, price=10)
        assert tracker.capital_gains[cover_date]['ShortTerm'] == 5.0, "ST gain remains"
        assert len(tracker.positions['ABC']) == 2, "2 tax lots"
        assert len(tracker.wash_trades) == 0, "No wash trades"
        tracker.validate()
        # Just over a year later, cover all shares
        second_cover_date = start_date + pd.DateOffset(days=367)
        tracker.trade(second_cover_date, 'ABC', shares=10, price=8)
        assert not tracker.positions['ABC'], "No open positions"
        assert tracker.capital_gains[second_cover_date]['LongTerm'] == 10, "$10 LT"
        assert tracker.capital_gains_df.sum()['ShortTerm'] == 15, "$15 ST"
        # Short 10 shares back within 30 days.  Still no loss, so no wash
        second_short_date = second_cover_date + pd.Timedelta(days=25)
        tracker.trade(second_short_date, 'ABC', shares=-10, price=7)
        assert len(tracker.positions['ABC']) == 1, "One position should exist for ABC"
        assert tracker.capital_gains[second_cover_date]['LongTerm'] == 10, "Realized $10 LT"
        assert tracker.capital_gains_df.sum()['ShortTerm'] == 15, "Realized $15 ST"

    def switch_sides(self, tracker_class: Callable[[], CapGainsTracker]):
        """Test Orders that not only wash losses, but also change the position side
        (from long to short, and vice versa).  Ref Example 1.4"""
        start_date = pd.Timestamp('2022-01-05')
        tracker = tracker_class()

        # Buy 5 shares of ABC at $10
        start_date = pd.Timestamp('2022-01-05')
        tracker.trade(start_date, 'ABC', shares=5, price=10)
        # Sell 3 shares of ABC at $9, realizing capital loss of $1/share
        sell_date = start_date + pd.Timedelta(days=1)
        tracker.trade(sell_date, 'ABC', -3, price=9)
        second_sell_date = start_date + pd.Timedelta(days=2)
        tracker.trade(second_sell_date, 'ABC', -7, price=8)
        tracker.validate()
        assert tracker.get_position('ABC') == -5, "Open position on ABC: -5 shares"
        assert tracker.capital_gains[sell_date]['ShortTerm'] == -3, "Realized -$3 ST"
        assert tracker.capital_gains[second_sell_date]['ShortTerm'] == -4, "-$4 ST"

        # Buy 10 shares.  This will wash the losses, and produce a loss of its own.
        buy_back_date = second_sell_date + pd.Timedelta(days=25)
        tracker.trade(buy_back_date, 'ABC', shares=10, price=10)
        assert len(tracker.wash_trades) == 2, "Now have two washed trades"
        assert tracker.capital_gains[buy_back_date]['ShortTerm'] == -10, "Realized -$10 ST"
        assert tracker.capital_gains_df.sum().sum() == -10, "Total realized cap gains = -$10"
        assert tracker.get_position('ABC') == 5, "Open position on ABC should be 5 shares"
        tracker.validate()
        # Sell 10 shares.  This will wash the current loss but realize the previously washed losses.
        sell_date = buy_back_date + pd.Timedelta(days=20)
        tracker.trade(sell_date, 'ABC', -10, price=9)
        assert len(tracker.wash_trades) == 3, "Now have washed losses on 3 trades"
        assert len(tracker.positions['ABC']) == 1, "One position should exist for ABC"
        assert tracker.positions['ABC'][0].basis_add == -10, "Position should have -$10 basis"
        assert tracker.positions['ABC'][0].basis_days == 25, "Position has 25 basis days"
        assert tracker.capital_gains[sell_date]['ShortTerm'] == -12, "Realized -$12 ST"

    def two_positions(self, tracker_class: Callable[[], CapGainsTracker]):
        """Buy and hold a two positions, on the same day, in RA for 367 days."""
        ledger = tracker_class()
        start_date = pd.Timestamp('2022-01-05')
        end_date = start_date + pd.DateOffset(days=367)
        ledger.buy(start_date, 'RA', shares=0.5, price=21.10)
        ledger.buy(start_date, 'RA', shares=0.5, price=21.30)
        assert ledger.sell(end_date, 'RA', 1, price=17) == 1, 'Shares sold'
        # We should have registered one closed position
        assert len(ledger.closed_lots_list) == 2, 'Number of Complete Trades'
        # Check price return matches capital gains
        assert list(ledger.capital_gains.values())[0]['LongTerm'] == pytest.approx(
            sum(t.tax_gain for t in ledger.closed_lots_list), rel=1e-6), 'Capital Gains'

    def multiple_repurchases(self, tracker_class: Callable[[], CapGainsTracker]):
        """Wash a capital loss over multiple repurchases.  Ref Example 1.5"""
        tracker = tracker_class()
        tracker.trade(pd.Timestamp('2023-01-15'), 'XYZ', shares=100, price=20)
        tracker.trade(pd.Timestamp('2023-02-10'), 'XYZ', shares=-100, price=15)
        tracker.trade(pd.Timestamp('2023-02-25'), 'XYZ', shares=40, price=17)
        assert tracker.capital_gains_df.sum()['ShortTerm'] == -300, "-$300 ST"
        assert tracker.get_position('XYZ') == 40, "Net position after 1st repurchase"
        for trade in tracker.closed_lots_list:
            if trade.washed:
                assert trade.tax_gain == -200, "Washed $200 ST"
            else:
                assert trade.tax_gain == -300, "Unwashed $300 ST"
        tracker.trade(pd.Timestamp('2023-03-08'), 'XYZ', shares=60, price=19)
        tracker.validate()
        assert tracker.capital_gains_df.sum().sum() == 0, "All losses washed"
        assert tracker.get_position('XYZ') == 100, "Net position after 2nd repurchase"
        assert len(tracker.wash_trades) == 2, "2 washed lots"

    def multiple_repo_buy_sell(self, tracker_class: Callable[[], CapGainsTracker]):
        """Make sure we get the same results with buy/sell methods.  Ref Example 1.5"""
        tracker = tracker_class()
        tracker.buy(pd.Timestamp('2023-01-15'), 'XYZ', 100, price=20)
        tracker.sell(pd.Timestamp('2023-02-10'), 'XYZ', -100, price=15)
        tracker.buy(pd.Timestamp('2023-02-25'), 'XYZ', 40, price=17)
        tracker.validate()
        assert tracker.capital_gains_df.sum()['ShortTerm'] == -300, "-$300 ST"
        assert tracker.get_position('XYZ') == 40, "Net position after 1st repurchase"
        for trade in tracker.closed_lots_list:
            if trade.washed:
                assert trade.tax_gain == -200, "Washed $200 ST"
            else:
                assert trade.tax_gain == -300, "Unwashed $300 ST"

    def wash_loss_plus_gain(self, tracker_class: Callable[[], CapGainsTracker]):
        """Washed loss and a gain in the same period.  Ref Example 1.6"""
        tracker = tracker_class()
        tracker.buy(pd.Timestamp('2023-01-15'), 'XYZ', 100, price=20)
        tracker.sell(pd.Timestamp('2023-02-10'), 'XYZ', -100, price=15)
        tracker.buy(pd.Timestamp('2023-02-25'), 'XYZ', 100, price=18)
        tracker.sell(pd.Timestamp('2023-03-15'), 'XYZ', 100, price=25)
        tracker.validate()
        assert tracker.capital_gains_df.sum()['ShortTerm'] == 200, "$200 ST"
        assert tracker.get_position('XYZ') == 0, "No open position"
        for trade in tracker.closed_lots_list:
            if trade.washed:
                assert trade.tax_gain == -500, "Washed $500 ST"
            else:
                assert trade.tax_gain == 200, "$200 net gain"

    def zero_trades(self, tracker_class: Callable[[], CapGainsTracker]):
        """Make sure zero trades don't break anything"""
        tracker = tracker_class()
        some_date = pd.Timestamp('2023-01-15')
        tracker.buy(some_date, 'XYZ', 0, price=20)
        another_date = pd.Timestamp('2023-01-16')
        tracker.sell(another_date, 'XYZ', 0, price=15)
        tracker.validate()
        assert tracker.last_date == another_date, "Last date"
        assert tracker.current_date == another_date, "Current date"
        assert tracker.capital_gains_df.sum().sum() == 0, "No cap gains"
        assert tracker.get_position('XYZ') == 0, "No position in XYZ"
        assert tracker.open_lots_str == 'No positions', "No positions"
        assert tracker.closed_lots_str == 'No trades', "No trades"

    def prior_purchase_wash(self, tracker_class: Callable[[], CapGainsTracker]):
        """Test wash sale logic when prior purchase is washed.  Ref Example 1.7a"""
        tracker = tracker_class()
        tracker.buy(pd.Timestamp('2023-01-15'), 'XYZ', 100, price=20)
        tracker.buy(pd.Timestamp('2023-01-16'), 'XYZ', 100, price=20)
        tracker.sell(pd.Timestamp('2023-02-10'), 'XYZ', -100, price=15)
        tracker.validate()
        assert tracker.capital_gains_df.sum().sum() == 0, "All losses washed"
        tracker.sell(pd.Timestamp('2023-02-11'), 'XYZ', -50, price=15)
        tracker.validate()
        assert tracker.capital_gains_df.sum().sum() == -500.0, "Realized -$500 ST"
        assert len(tracker.wash_trades) == 1, "1 washed lot"
        assert tracker.get_position('XYZ') == 50, "Net position"

    def prior_purchase_partial_wash(self, tracker_class: Callable[[], CapGainsTracker]):
        """Partial washes of closing losses due to previous opening trades.  Ref Example 1.7b"""
        tracker = tracker_class()
        tracker.buy(pd.Timestamp('2023-01-15'), 'XYZ', 60, price=20)
        tracker.buy(pd.Timestamp('2023-01-17'), 'XYZ', 50, price=21)
        tracker.buy(pd.Timestamp('2023-01-19'), 'XYZ', 40, price=22)
        tracker.sell(pd.Timestamp('2023-02-10'), 'XYZ', shares_to_sell=-20, price=16)
        tracker.validate()
        tracker.sell(pd.Timestamp('2023-02-11'), 'XYZ', -20, price=15)
        tracker.validate()
        assert len(tracker.wash_trades) == 2, "2 washed lots"
        tracker.sell(pd.Timestamp('2023-02-12'), 'XYZ', -40, price=14)
        tracker.validate()
        assert tracker.capital_gains_df.sum().sum() == 0, "All losses washed"

    def prior_short_partial_wash(self, tracker_class: Callable[[], CapGainsTracker]):
        """Partial washes of closing losses due to previous opening shorts.  Ref Example 1.7c"""
        tracker = tracker_class()
        tracker.trade(pd.Timestamp('2023-01-15'), 'XYZ', -60, price=16)
        tracker.trade(pd.Timestamp('2023-01-17'), 'XYZ', -50, price=15)
        tracker.trade(pd.Timestamp('2023-01-19'), 'XYZ', -40, price=14)
        tracker.trade(pd.Timestamp('2023-02-10'), 'XYZ', 20, price=20)
        tracker.validate()
        tracker.trade(pd.Timestamp('2023-02-11'), 'XYZ', 20, price=21)
        tracker.validate()
        tracker.trade(pd.Timestamp('2023-02-12'), 'XYZ', 40, price=22)
        tracker.validate()
        assert tracker.capital_gains_df.sum().sum() == 0, "All losses washed"
        assert tracker.get_position('XYZ') == -70, "Net position"

    def lot_selection(self, tracker_class: Callable[[], CapGainsTracker]):
        """Test different lot selection methods.  Ref Example 1.8"""
        def trade_sequence(tracker) -> pd.DataFrame:
            tracker.trade(pd.Timestamp('2024-01-15'), 'XYZ', 50, price=10)
            tracker.trade(pd.Timestamp('2024-01-30'), 'XYZ', 50, price=20)
            tracker.trade(pd.Timestamp('2024-02-17'), 'XYZ', 50, price=16)
            tracker.trade(pd.Timestamp('2024-02-27'), 'XYZ', 50, price=12)
            tracker.trade(pd.Timestamp('2025-01-20'), 'XYZ', -50, price=15)
            tracker.trade(pd.Timestamp('2025-01-25'), 'XYZ', -50, price=15)
            tracker.trade(pd.Timestamp('2025-02-25'), 'XYZ', -100, price=15)
            return tracker.capital_gains_df
        Config.LOT_SELECTION = TaxLotSelection.FIFO
        cg = trade_sequence(tracker_class())
        assert cg.sum()['ShortTerm'] == -100, "FIFO -$100 ST"
        assert cg.sum()['LongTerm'] == 200, "FIFO $200 LT"
        Config.LOT_SELECTION = TaxLotSelection.LIFO
        cg = trade_sequence(tracker_class())
        assert cg.sum()['ShortTerm'] == 100, "LIFO $100 ST"
        assert cg.sum()['LongTerm'] == 0, "LIFO $0 LT"
        Config.LOT_SELECTION = TaxLotSelection.HIFO
        cg = trade_sequence(tracker_class())
        assert cg.sum()['ShortTerm'] == -150, "HIFO -$150 ST"
        assert cg.sum()['LongTerm'] == 250, "HIFO $250 LT"
        Config.LOT_SELECTION = TaxLotSelection.LOFO
        cg = trade_sequence(tracker_class())
        assert cg.sum()['ShortTerm'] == 150, "LOFO $150 ST"
        assert cg.sum()['LongTerm'] == -50, "LOFO -$50 LT"


if __name__ == '__main__':
    pytest.main()
