"""Tests for PNLtracker classes"""
import pytest
import pandas as pd
from tax_tracker import PNLtracker

@pytest.mark.usefixtures("price_data", "distribution_data")
class TestPNL:
    """Tests for PNLtracker"""

    def test_buy_and_hold_single_position(self, price_data, distribution_data):
        """Buy and hold a single position in RA for at least 366 days.  Ref Example 3.1"""
        tracker = PNLtracker(price_data, distribution_data)
        start_date = tracker.prior_trading_date(pd.Timestamp('2022-01-06'))
        end_date = start_date + pd.DateOffset(days=366)
        last_date = None
        for date in tracker.trading_dates:
            tracker.BoD(date)
            if date == start_date:
                tracker.trade_dollars(date, 'RA', dollars=1_000)
            if date > end_date:
                tracker.close(date, 'RA')
            tracker.EoD(date)
            if date > end_date:
                last_date = date
                break
        #region Functionality checks
        tracker.validate()
        # We don't have any holdings after the end date
        assert tracker.holdings.loc[tracker.next_trading_date(end_date), 'RA'] == 0
        # We should have (last_date - start_date).days of holdings
        assert tracker.holdings[tracker.holdings != 0]['RA'].count() \
                == (last_date - start_date).days, 'Holding duration'
        assert len(tracker.closed_lots_list) == 1, 'Number of Complete Trades'
        summary = tracker.daily_df.sum()
        assert summary['OpenedTickers'] == summary['ClosedTickers'] == 1, 'Open/Close count'
        # Check price return matches capital gains
        trade = tracker.closed_lots_list[0]
        assert tracker.capital_gains_df.sum()['LongTerm'] \
                == pytest.approx(trade.tax_gain, rel=1e-6), 'Capital Gains'
        self.adjusted_price_check(tracker)
        #endregion

    def test_buy_and_hold_two_positions(self, price_data, distribution_data):
        """Buy and hold two positions, on the same day, in RA for 367 days."""
        tracker = PNLtracker(price_data, distribution_data)
        start_date = pd.Timestamp('2022-01-05')
        assert tracker.is_trading_date(start_date), 'Start date is a trading date'
        end_date = start_date + pd.DateOffset(days=367)
        tracker.BoD(start_date)
        tracker.buy(start_date, 'RA', shares=0.5, price=21.10)
        tracker.buy(start_date, 'RA', shares=0.5, price=21.30)
        tracker.EoD(start_date)
        tracker.validate()
        tracker.BoD(end_date)
        closed_lots = tracker.close(end_date, 'RA')
        assert sum(lot.shares for lot in closed_lots) == 1, 'Shares sold'
        tracker.EoD(end_date)
        #region Functionality checks
        tracker.validate()
        assert tracker.last_end_of_day == end_date, 'Last EoD'
        # We don't have any holdings after the end date
        assert tracker.holdings.loc[end_date + pd.DateOffset(days=1), 'RA'] == 0
        # We should have (end_date - start_date).days days of holdings
        assert tracker.holdings[tracker.holdings != 0]['RA'].count() \
                == (end_date - start_date).days, 'Holding duration'
        # We should have registered one closed position
        assert sum(tracker.closed_tickers_count.values()) == 1
        assert len(tracker.closed_lots_list) == 2, 'Number of Complete Trades'
        # Check price return matches capital gains
        assert tracker.capital_gains[end_date]['LongTerm'] == pytest.approx(
            sum(t.tax_gain for t in tracker.closed_lots_list), rel=1e-5, abs=1e-5), 'Capital Gains'
        self.adjusted_price_check(tracker)
        #endregion

    def adjusted_price_check(self, tracker: PNLtracker):
        """Compare DailyReturn to daily AdjPx returns."""
        if 'AdjPx' in tracker.data.columns:
            df = tracker.data.copy()
            df['AdjPxReturn'] = (df['AdjPx'] / df.groupby('Ticker')['AdjPx'].shift(1)) - 1.0
            date_range = tracker.detail_df.index.get_level_values('Date')
            daily_pnl = tracker.daily_returns_df[date_range.min():date_range.max()] \
                                .join(df.loc['RA']['AdjPxReturn'], how='left')
            assert daily_pnl[1:].sum()['Return'] == pytest.approx(
                   daily_pnl[1:].sum()['AdjPxReturn'], rel=1e-5, abs=1e-5)

if __name__ == '__main__':
    pytest.main()
