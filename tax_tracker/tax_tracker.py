"""Tax Trackers calculate tax treatment of trades and distributions.

This module provides classes for tracking and calculating the tax treatment of trades and distributions.
It includes functionality for handling capital gains, wash sales, and distributions (like dividends).
Given price data, PNLtracker can calculate daily profit and loss (P&L, a.k.a. PNL) and net asset value (NAV).

Classes:
    Config: Configuration settings for tax trackers.
    Lot: A tax lot, tracking trading activity for specific shares of a specific asset.
    WashLoss: Details a closed Lot with a washed loss.
    CapGainsTracker: Tracks capital gains and wash sales for trading activity.
    DistributionTracker: Extends CapGainsTracker to handle tax characterization of distributions.
    PNLtracker: Extends DistributionTracker to calculate daily P&L and NAV.

Constants:
    LONG_TERM_HOLDING_PERIOD: Minimum holding period (in days) for a gain to qualify as long-term.
    PIL_CONVERSION_PERIOD: Holding period (in days) after which payments-in-lieu (PIL) become interest expenses.
    PREFERRED_QUALIFIED_WINDOW: Qualification window (in days) for preferred dividends.
    QUALIFIED_WINDOW: Qualification window (in days) for regular dividends.
    WASH_SALE_WINDOW: Window (in days) to check for wash sales before and after a sale.

This module is designed to handle complex tax scenarios resulting from trading assets long and short.
It assumes FIFO for tax lots.  It applies wash sale and dividend qualification rules, based on U.S. law
as described by IRS Publication 550 (2024 version) at https://www.irs.gov/publications/p550.
It provides detailed accounting and reporting to follow the tax implications of trades and distributions.
"""

__all__ = ('Config', 'Lot', 'CapGainsTracker', 'DistributionTracker', 'PNLtracker')
__version__ = '0.1.0'
__author__ = 'David Bookstaber'

import math
import warnings
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, DefaultDict, Dict, Deque, List, Set, Tuple
import pandas as pd
warnings.filterwarnings('once', category=UserWarning)  # Doesn't yet work for Jupyter
# pylint: disable=invalid-name,line-too-long,pointless-string-statement

LONG_TERM_HOLDING_PERIOD = 366  # Minimum number of holding days for a gain to qualify as long-term
PIL_CONVERSION_PERIOD = 45  # Number of holding days after which payments-in-lieu (PIL) become interest expense
PREFERRED_QUALIFIED_WINDOW = 90  # Number of days before and after a preferred dividend to check for qualification
QUALIFIED_WINDOW = 60  # Number of days before and after a dividend to check for qualification
WASH_SALE_WINDOW = 30  # Number of days before and after a sale to check for wash sales

class Config:
    """Settings for tax_trackers.

    Attributes:
        SHARE_PRECISION (int): Maximum number of decimal places to use for shares; -1 for no limit.
        MIN_SHARE_SIZE (float): Minimum share size to create a lot. Smaller lots are disregarded.
    """
    SHARE_PRECISION = 4
    MIN_SHARE_SIZE = 0.0001


@dataclass
class Lot:
    """Tracks trading of one tax lot of an asset (ticker).

    A tax lot is opened with the first trade in a ticker, which can be a buy or sell (short).
    Lots can also be opened due to other trading activity requiring a lot to be split in two.
    Lots are closed when the opening trade is reversed.

    Attributes:
        ticker (str): The asset ticker symbol.
        shares (float): The number of shares in the lot.
        open_date (pd.Timestamp): The date the lot was opened.
        open_px (float): The price at which the lot was opened.
        close_px (float): The price at which the lot was closed. Defaults to NaN.
        close_date (pd.Timestamp): The date the lot was closed. Defaults to pd.Timestamp.min.
        basis_add (float): Basis adjustment in dollars, carried from washed losses and PILs.
        basis_days (int): Additional days added to the trade duration for tax term characterization.
        washed (bool): Whether the loss on this lot was washed by another trade.
        distributions (float): Total distributions received on this lot, in dollars.
        payment_in_lieu (DefaultDict[pd.Timestamp, float]): Payments-in-lieu (PIL) made on short positions
            held <= 45 days by DistributionTrackers, recorded in $/share, indexed by date of distribution.
    """
    ticker: str
    shares: float
    open_date: pd.Timestamp
    open_px: float
    close_px: float = float('nan')
    close_date: pd.Timestamp = pd.Timestamp.min
    basis_add: float = .0
    basis_days: int = 0
    washed: bool = False
    distributions: float = .0
    payment_in_lieu: DefaultDict[pd.Timestamp, float] = field(default_factory=lambda: defaultdict(float))

    def split(self, split_shares: float) -> 'Lot':
        """Split this into two lots, keeping (`self.shares - split_shares`) in the original lot.

        Args:
            split_shares (float): The number of shares to split off into a new lot.

        Returns:
            Lot: The split-off lot containing `split_shares` of the original's shares, basis, and P&L.
        """
        proportion = abs(split_shares / self.shares)
        split_basis = self.basis_add * proportion
        split_dists = self.distributions * proportion
        self.shares -= split_shares
        self.basis_add -= split_basis
        self.distributions -= split_dists
        split_lot = Lot(self.ticker, split_shares, self.open_date, self.open_px, self.close_px,
                        self.close_date, split_basis, self.basis_days, self.washed, split_dists,
                        self.payment_in_lieu.copy())
        return split_lot

    @property
    def is_open(self) -> bool:
        """Check if the position is still open.

        Returns:
            bool: True if the position is still open, False otherwise.
        """
        return self.close_date == pd.Timestamp.min

    @property
    def duration(self) -> int:
        """Calculate the holding period in days.

        Returns:
            int: The holding period in days.

        Raises:
            AssertionError: If the position is still open.
        """
        assert not self.is_open, "Cannot compute duration for open position"
        return (self.close_date - self.open_date).days

    @property
    def term(self) -> int:
        """Calculate the tax holding period in days.

        Returns:
            int: The tax holding period in days (i.e., duration + any basis adjustment for washes).

        Raises:
            AssertionError: If the position is still open.
        """
        assert not self.is_open, "Cannot compute term for open position"
        return self.duration + self.basis_days

    @property
    def price_pnl(self) -> float:
        """Calculate the PNL on a closed position based only on price change.

        Returns:
            float: The PNL on the closed position.

        Raises:
            AssertionError: If the position is still open.
        """
        assert not self.is_open, "Cannot compute price gain for open position"
        return self.shares * (self.close_px - self.open_px)

    @property
    def tax_gain(self) -> float:
        """Calculate the taxable gain on a closed position.

        Returns:
            float: The taxable gain on the closed position.
        """
        return self.price_pnl + self.basis_add

    @property
    def gain_type(self) -> str:
        """Characterize the gain as Short-term (ST) or Long-term (LT).

        Returns:
            str: 'LT' if the term is greater than or equal to LONG_TERM_HOLDING_PERIOD,
                 'ST' otherwise.
        """
        return 'LT' if self.term >= LONG_TERM_HOLDING_PERIOD else 'ST'

    @property
    def gain_type_verbose(self) -> str:
        """Characterize the gain as "ShortTerm" or LongTerm.

        Returns:
            str: 'LongTerm' if the term is greater than or equal to LONG_TERM_HOLDING_PERIOD,
                 'ShortTerm' otherwise.
        """
        return 'LongTerm' if self.term >= LONG_TERM_HOLDING_PERIOD else 'ShortTerm'

    def __str__(self) -> str:
        """Return a string representation of the lot.

        Returns:
            str: A string representation of the lot.
        """
        return self.str_open() if self.is_open else self.str_closed()

    def str_open(self) -> str:
        """Return a description of the open lot.

        Returns:
            str: A description of the open lot.
        """
        return Lot.no_trailing_zeros(self.shares) + f' @ ${self.open_px:.2f} ' + \
               f'({self.open_date.strftime("%Y-%m-%d")}) ' + \
               (f' Basis ${self.basis_add:+.2f}' if self.basis_add else '') + \
               (f' term+{self.basis_days} day{"s" if self.basis_days > 1 else ""}' if self.basis_days else '') + \
               (f' PIL ${(self.shares * sum(self.payment_in_lieu.values())):.2f}' if self.payment_in_lieu else '') + \
               (f' Distributions ${self.distributions:.2f}' if self.distributions else '')

    def str_closed(self) -> str:
        """Return a description of the closed lot.

        Returns:
            str: A description of the closed lot.
        """
        return f'{self.ticker}: ' \
            f'{self.open_date.strftime("%Y-%m-%d")} ' + Lot.no_trailing_zeros(self.shares) + \
            f' @ ${self.open_px:.2f} ' + \
            f'>> {self.close_date.strftime("%Y-%m-%d")} ' + Lot.no_trailing_zeros(-self.shares) + \
            f' @ ${self.close_px:.2f} ' \
            f'= ${self.tax_gain:.2f} {self.gain_type}' + \
            f' ({self.duration}{"+"+str(self.basis_days) if self.basis_days > 0 else ""} ' + \
                f'day{"s" if self.term > 1 else ""})' + \
            (' washed' if self.washed else '')

    @staticmethod
    def no_trailing_zeros(f: float, max_digits: Optional[int] = None) -> str:
        """Remove trailing zeros and decimal point from a number.

        Args:
            f (float): The number to format.
            max_digits (Optional[int]): Maximum number of decimal places to display.
                Defaults to None; otherwise overrides Config.SHARE_PRECISION.

        Returns:
            str: The formatted number as a string.
        """
        if max_digits is None:
            max_digits = Config.SHARE_PRECISION
        if max_digits >= 0:
            # Make sure we display at least two figures if max_digits would show it as zero
            if abs(f) < 10**(-max_digits):
                return f'{f:.2e}'
            f = round(f, max_digits)
        return str(f).rstrip('0').rstrip('.') if '.' in str(f) else str(f)


@dataclass
class WashLoss:
    """A wash loss is a trade to close a position at a loss, and an opposing trade of the same asset within +/-30 days.

    The loss cannot be claimed then for tax purposes, and instead is added to the basis of the "repurchased" asset.
    Reference IRS Publication 550, p.87: "Wash Sales" (https://www.irs.gov/publications/p550)

    Attributes:
        trade (Lot): The lot representing the trade that incurred the wash loss.
        wash_date (pd.Timestamp): The date of the wash trade.
        washed_quantity (float): The quantity of shares washed.
        washed_loss (float): The amount of loss washed.
    """
    trade: Lot
    wash_date: pd.Timestamp
    washed_quantity: float
    washed_loss: float

    def __str__(self) -> str:
        """Return a string representation of the wash loss.

        Returns:
            str: A string representation of the wash loss.
        """
        return f'{self.trade} on {self.wash_date.strftime("%Y-%m-%d")}'


class CapGainsTracker:
    """Computes capital gains from trading activity, accounting for wash sale rule,
    and characterizing gains as Short- or Long-term.  Assumes FIFO accounting.
    Keeps track of open positions, closed trades, wash sales, and capital gains.
    Trades must be entered in chronological date order.  (Trade order on same day is irrelevant.)

    Attributes:
        _last_date (Optional[pd.Timestamp]): The last date on which a trade was recorded.
        wash_trades (List[WashLoss]): List of trades with washed losses.
        positions (DefaultDict[str, Deque[Lot]]): Open Lots indexed by ticker.
        loss_longs (DefaultDict[str, Deque[Lot]]): Long Lots that lost money; might qualify as wash sales.
        loss_shorts (DefaultDict[str, Deque[Lot]]): Short Lots that lost money; might qualify as wash sales.
        complete_trades (DefaultDict[pd.Timestamp, List[Lot]]): Closed Lots indexed by sell date.
        capital_gains (DefaultDict[pd.Timestamp, dict]): Capital gains indexed by date.
    """
    def __init__(self):
        self._last_date: Optional[pd.Timestamp] = None
        self.wash_trades: List[WashLoss] = []
        # Dictionaries indexed by ticker.  NOTE: Once traded, the key remains even when the deque is empty.
        self.positions: DefaultDict[str, Deque[Lot]] = defaultdict(deque)
        self.loss_longs: DefaultDict[str, Deque[Lot]] = defaultdict(deque)
        self.loss_shorts: DefaultDict[str, Deque[Lot]] = defaultdict(deque)
        # Dictionaries indexed by date
        self.complete_trades: DefaultDict[pd.Timestamp, List[Lot]] = defaultdict(list)
        self.capital_gains: DefaultDict[pd.Timestamp, dict] = defaultdict(lambda: {'ShortTerm': 0, 'LongTerm': 0})

    def sign(self, x) -> int:
        """Return the sign of x as -1, 0, or 1.

        Args:
            x: The value to check.

        Returns:
            int: The sign of x.
        """
        return bool(x > 0) - bool(x < 0)

    def trade(self, date: pd.Timestamp, ticker: str, shares: float, price: float) -> Tuple[List[Lot], List[Lot]]:
        """Record a trade, processing as opening or closing based on existing position in ticker.

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            shares (float): The number of shares traded.
            price (float): The price at which the trade was executed.

        Returns:
            Tuple[List[Lot], List[Lot]]: A tuple containing:
                1. List of new lots created during this trade.
                2. List of closed lots created during this trade.

        Raises:
            ValueError: If trades are not entered in date order.
        """
        if self._last_date and (date < self._last_date):
            raise ValueError(f"Trades must be entered in date order. "
                             f"Received order for {ticker} on {date.strftime('%Y-%m-%d')} "
                             f"after trades on {self._last_date.strftime('%Y-%m-%d')}")
        current_position = self.get_position(ticker)
        closed_lots: List[Lot] = []
        new_lots: List[Lot] = []
        if self.sign(shares) == -self.sign(current_position):
            shares_to_close = math.copysign(min(abs(shares), abs(current_position)), shares)
            new_lots, closed_lots = self._close(date, ticker, shares_to_close, price)
            shares -= shares_to_close
        if shares != 0:
            more_new_lots, more_closed_lots = self._open(date, ticker, shares, price)
            new_lots.extend(more_new_lots)
            closed_lots.extend(more_closed_lots)

        return new_lots, closed_lots

    def _open(self, date: pd.Timestamp, ticker: str, shares: float, price: float) -> Tuple[List[Lot], List[Lot]]:
        """Process trade that opens or increases existing position in `ticker`.

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            shares (float): The number of shares traded.
            price (float): The price at which the trade was executed.

        Returns:
            Tuple[List[Lot], List[Lot]]: A tuple containing:
                1. List of new lots created during this trade.
                2. List of closed lots created during this trade.
        """
        new_lots = []
        closed_lots = []
        # Check for loss trades on same side that would be washed by this
        side = self.sign(shares)
        loss_trades = self.loss_longs if (side > 0) else self.loss_shorts
        while ticker in loss_trades and loss_trades[ticker] and abs(shares) > Config.MIN_SHARE_SIZE:
            loss_trade = loss_trades[ticker].pop()
            if (date - loss_trade.close_date).days <= WASH_SALE_WINDOW:
                # This loss_trade washes a capital loss
                washed_loss = loss_trade.tax_gain
                if (side*shares) >= (side*loss_trade.shares + Config.MIN_SHARE_SIZE):
                    # Entire loss-trade is washed
                    washed_quantity = loss_trade.shares
                else:
                    # We only washed part of the loss; pro-rate it
                    washed_quantity = shares
                    washed_fraction = washed_quantity / loss_trade.shares
                    washed_loss *= washed_fraction
                    # Put the un-washed quantity back onto loss_trades
                    unwashed_loss_trade = loss_trade.split(loss_trade.shares - washed_quantity)
                    if abs(unwashed_loss_trade.shares) > Config.MIN_SHARE_SIZE:
                        loss_trades[ticker].append(unwashed_loss_trade)
                        closed_lots.append(unwashed_loss_trade)
                        self.complete_trades[loss_trade.close_date].append(unwashed_loss_trade)
                loss_trade.washed = True
                # Reverse the wash trade
                self.capital_gains[loss_trade.close_date][loss_trade.gain_type_verbose] -= washed_loss
                # Buy the washed_quantity, but account for the washed trade's loss and effective purchase date
                new_lots.append(Lot(ticker, washed_quantity, date, price, basis_add=washed_loss, basis_days=loss_trade.term))
                shares -= washed_quantity
                self.wash_trades.append(WashLoss(loss_trade, date, washed_quantity, washed_loss))
        if abs(shares) > Config.MIN_SHARE_SIZE:
            new_lots.append(Lot(ticker, shares, date, price))
        self.positions[ticker].extend(new_lots)
        self._last_date = date
        return new_lots, closed_lots

    def _close(self, date: pd.Timestamp, ticker: str, shares: float, price: float) -> Tuple[List[Lot], List[Lot]]:
        """Process a trade that closes or reduces existing position in `ticker`.

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            shares (float): The number of shares traded.
            price (float): The price at which the trade was executed.

        Returns:
            Tuple[List[Lot], List[Lot]]: A tuple containing:
                1. List of new lots created during this trade.
                2. List of closed lots created during this trade.
        """
        new_lots = []
        closed_lots = []
        side = -self.sign(shares)  # Side of existing position (which we are trading to close)
        while abs(shares) > Config.MIN_SHARE_SIZE and self.positions[ticker]:
            closing_lot = self.positions[ticker].popleft()
            if (-side*shares + Config.MIN_SHARE_SIZE) < (side*closing_lot.shares):
                # Partial close: Put unclosed quantity back in positions, pro-rating capital gains
                unclosed_position = closing_lot.split(closing_lot.shares + shares)
                if abs(unclosed_position.shares) > Config.MIN_SHARE_SIZE:
                    new_lots.append(unclosed_position)
                    self.positions[ticker].appendleft(unclosed_position)
            closing_lot.close_px = price
            closing_lot.close_date = date
            closed_lots.append(closing_lot)

            # If position.gain is negative, check whether this trade is a wash sale
            gain = closing_lot.tax_gain  # Following logic is destructive, so work on a copy of the value
            if gain < .0:
                washable_shares = closing_lot.shares
                washed_shares = .0
                washed_gain = .0
                for open_position in list(self.positions[ticker]):
                    if abs(washable_shares) < Config.MIN_SHARE_SIZE:
                        break
                    # Find opening trades in this ticker in the last WASH_SALE_WINDOW days, made on days other than this trade's date:
                    if (open_position.open_date != closing_lot.open_date) and (date - open_position.open_date).days <= WASH_SALE_WINDOW:
                        assert self.sign(open_position.shares) == self.sign(closing_lot.shares), 'Washing opening trades must be same side'
                        # This open washes the loss on current close; we have to adjust the opening trade's basis
                        if (Config.MIN_SHARE_SIZE + abs(washable_shares)) < abs(open_position.shares):
                            # We have to split the opening trade as it is only partially washed
                            washed_proportion = washable_shares / open_position.shares
                            washed_position = open_position.split(washable_shares)
                            washed_position.basis_add += gain
                            washed_position.basis_days += closing_lot.term
                            # Create a new Lot for the washed quantity
                            self.positions[ticker].append(washed_position)
                            new_lots.append(washed_position)
                            washed_shares += washable_shares
                            washed_gain += washed_position.basis_add
                            washable_shares = .0
                            gain = .0
                        else:  # open.shares <= washable_shares: This opening trade is fully washed
                            washed_proportion = open_position.shares / washable_shares
                            open_position.basis_add += gain * washed_proportion
                            open_position.basis_days += closing_lot.term
                            washed_shares += open_position.shares
                            washed_gain += washed_proportion * gain
                            washable_shares -= open_position.shares
                            gain *= (1 - washed_proportion)
                if washed_shares:
                    closing_lot.washed = True
                    self.wash_trades.append(WashLoss(closing_lot, date, washed_shares, washed_gain))

            # Process capital gains
            if gain != .0:
                holding_period = (date - closing_lot.open_date).days + closing_lot.basis_days
                if holding_period >= LONG_TERM_HOLDING_PERIOD:
                    self.capital_gains[date]['LongTerm'] += gain
                else:
                    self.capital_gains[date]['ShortTerm'] += gain
                if gain < .0:
                    loss_trades = self.loss_longs if (side > 0) else self.loss_shorts
                    loss_trades[ticker].append(closing_lot)

            shares += closing_lot.shares

        self._last_date = date
        self.complete_trades[date].extend(closed_lots)
        return new_lots, closed_lots

    def buy(self, date: pd.Timestamp, ticker: str, shares: float, price: float):
        """Synonym for trade().

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            shares (float): The number of shares traded.
            price (float): The price at which the trade was executed.
        """
        self.trade(date, ticker, shares, price)

    def sell(self, date: pd.Timestamp, ticker: str, shares_to_sell: float, price: float) -> float:
        """Synonym for trade().

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            shares_to_sell (float): The number of shares to sell. (Positive value will be converted to negative.)
            price (float): The price at which the trade was executed.

        Returns:
            float: The number of shares closed.
        """
        if shares_to_sell > .0:
            shares_to_sell *= -1.0
        _, closed_lots = self.trade(date, ticker, shares_to_sell, price)
        return sum(trade.shares for trade in closed_lots)

    def get_position(self, ticker: str) -> float:
        """Return the current position in the ticker, as number of shares.

        Args:
            ticker (str): The asset ticker symbol.

        Returns:
            float: The current position in the ticker.
        """
        if ticker not in self.positions:
            return .0
        return sum(position.shares for position in self.positions[ticker])

    def validate(self):
        """Run checks that should always be true.

        Raises:
            AssertionError: If the validation checks fail.
        """
        # CapitalGains + CarriedGains = PriceGains on closed positions
        assert math.isclose(self.capital_gains_df.sum().sum() + sum(position.basis_add for position in self.open_lots_list),
                            sum(trade.price_pnl for trade in self.closed_lots_list), rel_tol=1e-6), \
                "CapitalGains + CarriedGains = PriceGains " + (self.last_date.strftime("%Y-%m-%d") if self.last_date else '')

    #region Properties
    @property
    def last_date(self) -> Optional[pd.Timestamp]:
        """Return the last date on which we received activity.

        Returns:
            Optional[pd.Timestamp]: The last date on which we received activity.
        """
        return self._last_date

    @property
    def current_date(self) -> Optional[pd.Timestamp]:
        """Return the current date.

        Returns:
            Optional[pd.Timestamp]: The current date.
        """
        return self._last_date

    @property
    def capital_gains_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily capital gains.

        Returns:
            pd.DataFrame: A DataFrame of daily capital gains.
        """
        return pd.DataFrame.from_dict(self.capital_gains, orient='index')

    @property
    def open_lots_str(self) -> str:
        """Return a string summarizing open positions.

        Returns:
            str: A string summarizing open positions.
        """
        return '\n'.join([f'{ticker}: {position}'
                          for ticker, positions in self.positions.items()
                          for position in positions] or ['No positions'])

    @property
    def open_lots_list(self) -> List[Lot]:
        """Return a list of all open positions.

        Returns:
            List[Lot]: A list of all open positions.
        """
        return [position for positions in self.positions.values() for position in positions]

    @property
    def closed_lots_str(self) -> str:
        """Return a string summarizing all completed trades.

        Returns:
            str: A string summarizing all completed trades.
        """
        return '\n'.join([str(trade) for trades in self.complete_trades.values() for trade in trades]
                          or ['No trades'])

    @property
    def closed_lots_list(self) -> List[Lot]:
        """Return a list of all completed trades.

        Returns:
            List[Lot]: A list of all completed trades.
        """
        return [trade for trades in self.complete_trades.values() for trade in trades]

    @property
    def washed_lots_str(self) -> str:
        """Return a string summarizing wash sales.

        Returns:
            str: A string summarizing wash sales.
        """
        return '\n'.join([str(trade) for trade in self.wash_trades] or ['No washed trades'])
    #endregion Properties


class DistributionTracker(CapGainsTracker):
    """Adds tax characterization of distributions to CapGainsTracker.

    Attributes:
        distributions (pd.DataFrame): DataFrame of distribution data.  See __init__() for details.
        _last_BoD (Optional[pd.Timestamp]): The last Beginning-of-Day (BoD) run date.
        _last_EoD (Optional[pd.Timestamp]): The last End-of-Day (EoD) run date.
        daily_dividends (DefaultDict[pd.Timestamp, dict]): Daily dividends indexed by date.
        holdings (pd.DataFrame): DataFrame of holdings: number of shares of each ticker held at end of each date.
        already_qualified (DefaultDict[Tuple[pd.Timestamp, str], float]): Shares already qualified for a given ex-date.
        pending_pil_positions (DefaultDict[pd.Timestamp, List[Lot]]): Open shorts with payments-in-lieu (PIL) that have not yet qualified as interest expenses.
        pending_qualified (Deque[Tuple[pd.Timestamp, Lot, float]]): Dividends that can be qualified if/when holding period is satisfied.
        pending_qualified_preferred (Deque[Tuple[pd.Timestamp, Lot, float]]): Preferred dividends that can be qualified if/when holding period is satisfied.
    """
    def __init__(self, distribution_data: Optional[pd.DataFrame] = None):
        """Initialize the DistributionTracker.

        Args:
            distribution_data (Optional[pd.DataFrame]): DataFrame multi-indexed by [Ticker, Date, Type]
                with column [Distribution] of type float listing distribution amount per share.
                [Date] is the ex-dividend date: positions held at end of previous day receive the distribution.
                [Type] characterizes the [Distribution] as one of:
                 * [D]ividend (regular)
                 * [P]referred ("due to periods totaling more than 366 days")
                 * [N]ot qualified (i.e., ineligible to be qualified regardless of holding period)
                 * [E]xempt (tax-exempt)
                 * [R]eturn of Capital (also tax-exempt)
                 * [L]ong-Term Gain
                 * [S]hort-Term Gain
        
        Notes: Rules on qualifying dividends are in IRS Publication 550 p.28
        """
        super().__init__()
        self.distributions = distribution_data
        self._last_BoD: Optional[pd.Timestamp] = None
        self._last_EoD: Optional[pd.Timestamp] = None
        # Distributions are processed and listed on ex-date, not on paid date.
        self.daily_dividends: DefaultDict[pd.Timestamp, dict] = defaultdict(
            lambda: {'Regular': .0,
                     'Qualified': .0,
                     'Exempt': .0,
                     'LongTermGain': .0,   # Add to .capital_gains['LongTerm']
                     'ShortTermGain': .0,  # Add to .capital_gains['ShortTerm']
                     'PayInLieu': .0,      # Add to .capital_gains['ShortTerm']
                     'Expense': .0,
                    })

        # `holdings` records the total quantity of each ticker held at the end of each date;
        #   Must include zeroes and non-trading calendar days so we can compute holding term.
        if (self.distributions is None) or self.distributions.empty:
            self.holdings = pd.DataFrame(index=pd.DatetimeIndex([]), dtype=float)
        else:
            # Preallocate holdings DataFrame
            unique_tickers = self.distributions.index.get_level_values('Ticker').unique()
            start_date = self.distributions.index.get_level_values('Date').min()
            end_date = self.distributions.index.get_level_values('Date').max()
            self.holdings = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date),
                                     columns=unique_tickers, dtype=float).fillna(.0)
        # Keep track of shares that have already been qualified for a given ex-date
        # so that we don't qualify a share for more than one tax lot
        self.already_qualified: DefaultDict[Tuple[pd.Timestamp, str], float] = defaultdict(float)

        # Open shorts with payments-in-lieu (PIL) that have not yet qualified as interest expenses.
        # Indexed by Lot .open_date
        self.pending_pil_positions: DefaultDict[pd.Timestamp, List[Lot]] = defaultdict(list)

        # Dividends that can be qualified if/when holding period is satisfied.
        # Deque of (ex-date, Lot, $dividend/share)
        self.pending_qualified: Deque[Tuple[pd.Timestamp, Lot, float]] = deque()
        self.pending_qualified_preferred: Deque[Tuple[pd.Timestamp, Lot, float]] = deque()

    def trade(self, date: pd.Timestamp, ticker: str, shares: float, price: float) -> Tuple[List[Lot], List[Lot]]:
        """Record a trade, processing as opening or closing based on existing position in ticker.

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            shares (float): The number of shares traded.
            price (float): The price at which the trade was executed.

        Returns:
            Tuple[List[Lot], List[Lot]]: A tuple containing:
                1. List of new lots created during this trade.
                2. List of closed lots created during this trade.
        """
        if not self._last_BoD or (date > self._last_BoD):
            self.BoD(date)

        new_lots, closed_lots = super().trade(date, ticker, shares, price)
        
        # If we created a Lot with pending PIL, add it to the list
        for trade in new_lots:
            if trade.payment_in_lieu:
                assert trade.shares < 0, "PILs are only for short positions"
                self.pending_pil_positions[trade.open_date].append(trade)
        # If we closed a Lot with pending PIL, remove it from those lists
        for trade in closed_lots:
            if trade.payment_in_lieu:
                self.pending_pil_positions[trade.open_date].remove(trade)
        return new_lots, closed_lots

    def receive_distribution(self, date: pd.Timestamp, ticker: str, dist_amount: float, dist_type: str):
        """Record a distribution for a ticker on a date.

        Args:
            date (pd.Timestamp): The date of the distribution.
            ticker (str): The asset ticker symbol.
            dist_amount (float): The distribution amount per share.
            dist_type (str): The type of distribution.

        Notes: Distributions are paid on holdings as of the end of the previous day.
        """
        if not self._last_BoD or (date > self._last_BoD):
            self.BoD(date)
        assert self._last_BoD == date, \
            f"Received distribution for {date} with last BoD {self._last_BoD}"

        dist_type = dist_type[0].upper()
        shares = self.get_eod_shares(date - pd.Timedelta(days=1), ticker)
        if shares > 0:
            for position in self.positions[ticker]:
                position.distributions += position.shares * dist_amount
            if dist_type == 'L':
                self.daily_dividends[date]['LongTermGain'] += shares * dist_amount
            elif dist_type == 'S':
                self.daily_dividends[date]['ShortTermGain'] += shares * dist_amount
            elif dist_type in ('E', 'R'):  # Tax-exempt or Return of Capital
                self.daily_dividends[date]['Exempt'] += shares * dist_amount
            elif dist_type == 'N':  # Not qualified
                self.daily_dividends[date]['Regular'] += shares * dist_amount
            # For dividends that can become qualified, we add them as 'Regular' to begin with
            # and later move them to 'Qualified' if we subsequently find the holding period is satisfied
            elif dist_type == 'D':  # Regular dividend
                self.daily_dividends[date]['Regular'] += shares * dist_amount
                for position in self.positions[ticker]:
                    self.pending_qualified.append((date, position, dist_amount))
            elif dist_type == 'P':  # Preferred dividend
                self.daily_dividends[date]['Regular'] += shares * dist_amount
                for position in self.positions[ticker]:
                    self.pending_qualified_preferred.append((date, position, dist_amount))
            else:
                raise ValueError(f"Unknown distribution type {dist_type}")

        elif shares < 0:  # Short positions make payments in lieu of dividends
            for position in self.positions[ticker]:
                position.distributions += position.shares * dist_amount
                # After 45 days, payments in lieu of dividends on short positions are interest expenses.
                # Ref IRS Publication 550 p.86.
                if (date - position.open_date).days > PIL_CONVERSION_PERIOD:
                    self.daily_dividends[date]['Expense'] += position.shares * dist_amount
                else:
                    # Log the payment in lieu to the position; to be accounted
                    #  when the position is closed or held for more than PIL_CONVERSION_PERIOD.
                    if not position.payment_in_lieu:
                        self.pending_pil_positions[position.open_date].append(position)
                    position.payment_in_lieu[date] += dist_amount
                    self.daily_dividends[date]['PayInLieu'] += position.shares * dist_amount

    def count_qualified_shares(self, ticker: str, exdate: pd.Timestamp, period: int) -> float:
        """Return the number of shares of `ticker` that were held for at least `period`+1 days
            when looking at the date range of `period` days before and after `exdate`.

        Args:
            ticker (str): The asset ticker symbol.
            exdate (pd.Timestamp): The ex-dividend date.
            period (int): The limiting holding period in days before qualification.

        Returns:
            float: The number of shares that were held for at least `period`+1 days and not already qualified.
        """
        q = self.holdings.shift().loc[(exdate - pd.Timedelta(days=period)):(exdate + pd.Timedelta(days=period)),
                                      ticker].fillna(0).nlargest(period+1)
        if len(q) < period+1:
            return 0
        return q.min() - self.already_qualified[(exdate, ticker)]

    def EoD(self, date: pd.Timestamp):
        """End of Day: Must be called after all transactions are made for the date.
            Updates `holdings` to reflect the end-of-day positions.

        Args:
            date (pd.Timestamp): The date for which the EoD processing is being performed.

        Raises:
            AssertionError: If the method is called with a date that is not later than the last BoD date.
        """
        assert self._last_BoD == date, \
            f"In EoD: BoD was not called for {date}"
        assert (not self._last_EoD) or date > self._last_EoD, \
            f"EoD called {date} with last EoD={self._last_EoD}"
        self._last_EoD = date

        for ticker, positions in self.positions.items():
            shares = sum(position.shares for position in positions)
            if ticker not in self.holdings.columns:
                self.holdings[ticker] = .0
            self.holdings.loc[date, ticker] = shares
    run_end_of_day = EoD  # Alias that complies with method naming convention

    def get_eod_shares(self, date: pd.Timestamp, ticker: str) -> float:
        """Get the number of shares of `ticker` held at the end of the day `date`.

        Args:
            date (pd.Timestamp): The date for which the end-of-day shares are being retrieved.
            ticker (str): The asset ticker symbol.

        Returns:
            float: The number of shares held at the end of the day `date`.  If none return .0.
        """
        if date not in self.holdings.index:
            return .0
        if ticker not in self.holdings.columns:
            return .0
        return self.holdings.loc[date, ticker]

    def BoD(self, date: pd.Timestamp) -> Dict[str, float]:
        """Beginning of Day (BoD) processing for a given date. This method must be called before 
            any transactions are entered on the specified date to ensure that incoming positions 
            going ex-dividend are properly accounted for.

        Args:
            date (pd.Timestamp): The date for which the BoD processing is being performed.

        Returns:
            Dict[str, float]: A dictionary keyed by ticker, listing total dividends received per share
                  for all tickers held on the specified date.

        Raises:
            AssertionError: If the method is called with a date not later than the last BoD date.

        Notes:
            - Ensures that the holdings DataFrame is extended to include the specified date.
            - Fills in skipped days by iteratively calling BoD for intermediate dates.
            - Calls End of Day (EoD) processing for the previous day if it was not already called.
            - Processes distributions for stocks held on the specified date.
            - Reviews pending distributions to determine if they qualify as "qualified dividends" 
              based on IRS holding period rules.
            - Handles pending payment-in-lieu (PIL) positions to account for interest expenses 
              on short positions held for more than PIL_CONVERSION_PERIOD days.
        """
        assert (not self._last_BoD) or date > self._last_BoD, \
            f"BoD called {date} with last BoD={self._last_BoD}"

        # Make sure holdings extends to current date:
        if not date in self.holdings.index:
            self.holdings = self.holdings.reindex(self.holdings.index.union(
                pd.date_range(start=min(date, self.holdings.index.min() or date),
                              end=max(date, self.holdings.index.max() or date)))).fillna(.0)

        # If we skipped any days, go back and fill them in:
        if (last_date := self._last_BoD) and (date - last_date).days > 1:
            for d in pd.date_range(start=last_date, end=date, inclusive='neither'):
                self.BoD(d)
        # If EoD wasn't called for previous day then call it:
        if self._last_BoD and ((not self._last_EoD) or (date - self._last_EoD).days > 1):
            self.EoD(self._last_BoD)
        self._last_BoD = date

        # Receive distributions for this exdate:
        dividends_received: Dict[str, float] = {}
        if self.distributions is not None:
            for ticker, group in self.distributions[self.distributions.index.get_level_values('Date') == date].groupby(level='Ticker'):
                if self.get_position(ticker):
                    for idx, row in group.iterrows():
                        dist_type = idx[2]
                        dist_amount = row['Distribution']
                        self.receive_distribution(date, ticker, dist_amount, dist_type)

                    if ticker not in dividends_received:
                        dividends_received[ticker] = 0.0
                    dividends_received[ticker] += dist_amount

        #region Review pending distributions to see if any have become qualified due to holding period
        '''From IRS Publication 550, p.28 – HOLDING PERIOD:
            You must have held the stock for more than 60 days during the 121-day period that begins
                60 days before the ex-dividend date. When counting the number of days you held the stock,
                include the day you disposed of the stock, but not the day you acquired it.
            In the case of preferred stock, you must have held the stock more than 90 days during the
                181-day period that begins 90 days before the ex-dividend date if the dividends are
                due to periods totaling more than 366 days.
        '''
        for pending in list(self.pending_qualified):
            exdate, position, dist_amount = pending
            if (date - exdate).days > QUALIFIED_WINDOW:
                # If not qualified by now then it never will be
                self.pending_qualified.remove(pending)
            elif (qualified_shares := self.count_qualified_shares(position.ticker, exdate, QUALIFIED_WINDOW)) > 0:
                # Number of shares held at the close of the day before the ex-dividend date
                dividend_shares = self.get_eod_shares(exdate-pd.Timedelta(days=1), position.ticker)
                assert dividend_shares >= qualified_shares
                self.already_qualified[(exdate, position.ticker)] += qualified_shares
                self.daily_dividends[exdate]['Qualified'] += qualified_shares * dist_amount
                self.daily_dividends[exdate]['Regular'] -= qualified_shares * dist_amount
                self.pending_qualified.remove(pending)

        for pending in list(self.pending_qualified_preferred):
            exdate, position, dist_amount = pending
            if (date - exdate).days > PREFERRED_QUALIFIED_WINDOW:
                # If not qualified by now then it never will be
                self.pending_qualified_preferred.remove(pending)
            elif (qualified_shares := self.count_qualified_shares(position.ticker, exdate, PREFERRED_QUALIFIED_WINDOW)) > 0:
                # Number of shares held at the close of the day before the ex-dividend date
                dividend_shares = self.get_eod_shares(exdate-pd.Timedelta(days=1), position.ticker)
                assert dividend_shares >= qualified_shares
                self.already_qualified[(exdate, position.ticker)] += qualified_shares
                self.daily_dividends[exdate]['Qualified'] += qualified_shares * dist_amount
                self.daily_dividends[exdate]['Regular'] -= qualified_shares * dist_amount
                self.pending_qualified_preferred.remove(pending)

        # Review pending_pil_positions to see if any have become interest expenses
        #  because the short position was open for more than PIL_CONVERSION_PERIOD:
        for open_date in sorted(self.pending_pil_positions.keys()):
            if (date - open_date).days <= PIL_CONVERSION_PERIOD:
                break
            for position in self.pending_pil_positions[open_date]:
                assert position.is_open
                assert position.shares < 0
                for dist_date, dist_amount in position.payment_in_lieu.items():
                    pil = position.shares * dist_amount
                    self.daily_dividends[dist_date]['PayInLieu'] -= pil
                    self.daily_dividends[dist_date]['Expense'] += pil
                position.payment_in_lieu.clear()
            del self.pending_pil_positions[open_date]
        #endregion Review pending distributions
        return dividends_received
    run_beginning_of_day = BoD  # Alias that complies with method naming convention

    def validate(self):
        """Run checks that should always be true.

        Raises:
            AssertionError: If the validation checks fail.
        """
        super().validate()
        # Sum of daily_dividends should equal sum of open- and closed-lot distributions.
        lot_dists = sum(position.distributions for position in self.open_lots_list) \
                    + sum(trade.distributions for trade in self.closed_lots_list)
        assert math.isclose(self.dividends_df.sum().sum(), lot_dists, rel_tol=1e-6), \
            "Distributions " + (self.current_date.strftime("%Y-%m-%d") if self.current_date else '')

    #region Properties
    @property
    def current_date(self) -> Optional[pd.Timestamp]:
        """Return the most recent BoD.

        Returns:
            Optional[pd.Timestamp]: The most recent BoD.
        """
        return self._last_BoD

    @property
    def taxable_cashflows_all_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily capital gains and distributions,
            merging distributions that are taxed as capital gains,
            including empty columns.

        Returns:
            pd.DataFrame: A DataFrame of daily capital gains and distributions.
        """
        if len(df := self.dividends_all_df) > 0:
            if len(base_cap_gains := super().capital_gains_df) > 0:
                df = df.join(base_cap_gains, how='outer').fillna(.0)
            else:
                df['ShortTerm'] = .0
                df['LongTerm'] = .0
            df['ShortTerm'] += df['ShortTermGain'] + df['PayInLieu']
            df['LongTerm'] += df['LongTermGain']
            return df.drop(columns=['ShortTermGain', 'LongTermGain', 'PayInLieu'])
        return super().capital_gains_df

    @property
    def taxable_cashflows_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily capital gains and distributions,
            merging distributions that are taxed as capital gains,
            excluding empty columns.

        Returns:
            pd.DataFrame: A DataFrame of daily capital gains and distributions.
        """
        df = self.taxable_cashflows_all_df
        return df.loc[:, (df != 0).any(axis=0)        # Drop zero columns
                     ].loc[(df != 0).any(axis=1), :]  # Drop zero rows

    @property
    def dividends_all_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily distributions, including empty columns.

        Returns:
            pd.DataFrame: A DataFrame of daily distributions.
        """
        return pd.DataFrame.from_dict(self.daily_dividends, orient='index')

    @property
    def dividends_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily distributions, excluding empty columns.

        Returns:
            pd.DataFrame: A DataFrame of daily distributions.
        """
        divs = self.dividends_all_df
        if len(divs) > 0:
            return divs.loc[:, (divs != 0).any()]
        return divs
    #endregion Properties


class PNLtracker(DistributionTracker):
    """Adds daily P&L and NAV calculation to tax characterization of distributions and capital gains.

    Attributes:
        trading_dates (pd.DatetimeIndex): List of trading dates.
        data (pd.DataFrame): DataFrame of price data.  See __init__() for details.
        warned_no_price (Set): Tickers we have warned are missing price data.
        pnl (DefaultDict[pd.Timestamp, float]): Distributions received plus dollar returns on assets, indexed by date.
        nav (DefaultDict[pd.Timestamp, float]): NAV (sum dollar value of all assets at end of day), indexed by date.
        opened_tickers_count (DefaultDict[pd.Timestamp, int]): Count of tickers with a position on the date and no position on the previous date.
        closed_tickers_count (DefaultDict[pd.Timestamp, int]): Count of tickers with a position on the previous date and no position ending on the date.
        detail (DefaultDict[pd.Timestamp, DefaultDict[str, dict]]): Detailed activity indexed by date and ticker.
    """
    def __init__(self, price_data: pd.DataFrame, distribution_data: Optional[pd.DataFrame] = None):
        """Initialize the PNLtracker.

        Args:
            price_data (pd.DataFrame): DataFrame multi-indexed by [Ticker, Date]
                with columns [Px, (AdjPx)] of type float.
                * `Px` is the split-adjusted closing price for the date.
                * Optional `AdjPx` is a closing price adjusted to reflect the total return.
                    (If provided then `AdjPx` can be used to error-check P&L calculations.)
            distribution_data (Optional[pd.DataFrame]): DataFrame of distribution data.
        """
        super().__init__(distribution_data)
        #region Prepare market data
        self.trading_dates: pd.DatetimeIndex = price_data.index.get_level_values('Date').unique().sort_values()
        # # Fill price forward on non-trading dates so we have entries for all calendar dates
        # self.data = price_data.groupby(level=0).apply(lambda x: x.reset_index(level=0, drop=True).asfreq("D"))
        # self.data = self.data.groupby('Ticker').ffill()
        self.data = price_data
        self.data['PrevPx'] = self.data.groupby(['Ticker'])['Px'].shift(1)
        #self.data['PxReturn'] = (self.data['Px'] / self.data['PrevPx']) - 1.0
        #endregion
        self.warned_no_price: Set = set()

        #region Dictionaries indexed by date
        self.pnl: DefaultDict[pd.Timestamp, float] = defaultdict(float)
        self.nav: DefaultDict[pd.Timestamp, float] = defaultdict(float)
        self.opened_tickers_count: DefaultDict[pd.Timestamp, int] = defaultdict(int)
        self.closed_tickers_count: DefaultDict[pd.Timestamp, int] = defaultdict(int)
        #endregion Dictionaries indexed by date

        # Detailed activity indexed by date, ticker.  Values as of EoD.  PNL does not include Divs.
        self.detail: DefaultDict[pd.Timestamp, DefaultDict[str, dict]] = defaultdict(lambda: defaultdict(lambda:
            {'Lots': 0, 'SharesEoD': .0, 'SharesClosed': .0, 'Px': .0, 'PNL': .0, 'Divs': .0}))

    def is_trading_date(self, date: pd.Timestamp) -> bool:
        """Check if the date is a trading date.

        Args:
            date (pd.Timestamp): The date to check.

        Returns:
            bool: True if the date is a trading date, False otherwise.
        """
        return date in self.trading_dates

    def next_trading_date(self, date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Get the next trading date after the specified date.

        Args:
            date (pd.Timestamp): The date for which to find the next trading date.

        Returns:
            Optional[pd.Timestamp]: The next trading date after the specified date,
                or None if there are no more trading dates.
        """
        try:
            return self.trading_dates[self.trading_dates > date][0]
        except IndexError:
            return None

    def prior_trading_date(self, date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Get the trading date before the specified date.

        Args:
            date (pd.Timestamp): The date for which to find the prior trading date.

        Returns:
            Optional[pd.Timestamp]: The trading date before the specified date,
                or None if there are no prior trading dates listed in data.
        """
        try:
            return self.trading_dates[self.trading_dates < date][-1]
        except IndexError:
            return None

    def get_price(self, date: pd.Timestamp, ticker: str, fill: bool = False) -> Optional[float]:
        """Get the price of a ticker on a date.

        Args:
            date (pd.Timestamp): The date for which to get the price.
            ticker (str): The asset ticker symbol.
            fill (bool): Whether to return the last price available prior to the date if no price is listed. Defaults to False.

        Returns:
            Optional[float]: The price of the ticker on the date, or None if no price is available.
        """
        if ticker not in self.data.index.get_level_values('Ticker'):
            if ticker not in self.warned_no_price:
                self.warned_no_price.add(ticker)
                warnings.warn(f"No price data for {ticker}")
            return None
        if date not in self.data.loc[ticker].index:
            if fill:
                return float(self.data.loc[ticker].Px.asof(date))
            return None
        return pd.to_numeric(self.data.loc[(ticker, date), 'Px'], errors='coerce', downcast='float')

    def trade(self, date: pd.Timestamp, ticker: str, shares: float, price: Optional[float] = None) -> Tuple[List[Lot], List[Lot]]:
        """Record a trade. If price is not specified, use the closing price for the date.

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            shares (float): The number of shares traded.
            price (Optional[float]): The price at which the trade was executed. Defaults to None,
                    in which case the closing price for the date is used.

        Returns:
            Tuple[List[Lot], List[Lot]]: A tuple containing:
                1. List of new lots created during this trade.
                2. List of closed lots created during this trade.
        """
        if price is None:
            price = self.get_price(date, ticker, fill=True)
        assert price is not None, f"Price for {ticker} on {date} is not available."

        # Increment counter if there are no current positions in this ticker:
        if not self.positions[ticker]:
            self.opened_tickers_count[date] += 1

        new_lots, closed_lots = super().trade(date, ticker, shares, price)

        if (previous_price := self.get_price(date-pd.DateOffset(days=1), ticker, fill=True)) is not None:
            # Calculate PNL for closed lots:
            for trade in closed_lots:
                self.pnl[date] += trade.shares * (trade.close_px - previous_price)
                self.detail[date][ticker]['PNL'] += trade.shares * (trade.close_px - previous_price)
                self.detail[date][ticker]['SharesClosed'] += trade.shares

        # Increment counter if there are no current positions in this ticker:
        if not self.positions[ticker]:
            self.closed_tickers_count[date] += 1

        return new_lots, closed_lots

    def trade_dollars(self, date: pd.Timestamp, ticker: str, dollars: float, price: Optional[float] = None) -> Tuple[List[Lot], List[Lot]]:
        """Record a trade, sized to use `dollars`. If price is not specified, use the closing price for the date.

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            dollars (float): The dollar amount of the trade.
            price (Optional[float]): The price at which the trade was executed. Defaults to None,
                    in which case the closing price for the date is used.

        Returns:
            Tuple[List[Lot], List[Lot]]: A tuple containing:
                1. List of new lots created during this trade.
                2. List of closed lots created during this trade.
        """
        if price is None:
            price = self.get_price(date, ticker, fill=True)
        assert price is not None, f"Price for {ticker} on {date} is not available."
        shares = dollars / price
        return self.trade(date, ticker, shares, price)

    def close(self, date: pd.Timestamp, ticker: str, price: Optional[float] = None) -> List[Lot]:
        """Close all open lots of `ticker`. If price not given, use the closing price for the date.

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            price (Optional[float]): The price at which the trade was executed. Defaults to None,
                    in which case the closing price for the date is used.

        Returns:
            List[Lot]: A list of closed lots created during this trade.
        """
        closed = []
        while self.positions[ticker]:
            _, closed_lots = self.trade(date, ticker, -self.positions[ticker][0].shares, price)
            closed.extend(closed_lots)
        return closed

    def buy(self, date: pd.Timestamp, ticker: str, shares: float, price: Optional[float] = None) -> Tuple[List[Lot], List[Lot]]:
        """Record a trade to buy. If price is not specified, use the closing price for the date.

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            shares (float): The number of shares traded.
            price (Optional[float]): The price at which the trade was executed. Defaults to None,
                    in which case the closing price for the date is used.

        Returns:
            Tuple[List[Lot], List[Lot]]: A tuple containing:
                1. List of new lots created during this trade.
                2. List of closed lots created during this trade.
        """
        return self.trade(date, ticker, shares, price)

    def sell(self, date: pd.Timestamp, ticker: str, shares_to_sell: float, price: Optional[float] = None) -> float:
        """Record a trade to sell. If price is not specified, uses the closing price for the date.

        Args:
            date (pd.Timestamp): The date of the trade.
            ticker (str): The asset ticker symbol.
            shares_to_sell (float): The number of shares to sell. Positive value will be converted to negative.
            price (Optional[float]): The price at which the trade was executed. Defaults to None,
                    in which case the closing price for the date is used.

        Returns:
            float: The number of shares closed.
        """
        if shares_to_sell > .0:
            shares_to_sell *= -1.0
        _, closed_lots = self.trade(date, ticker, shares_to_sell, price)
        return sum(trade.shares for trade in closed_lots)

    def BoD(self, date: pd.Timestamp):
        """Beginning of Day (BoD) processing for a given date.

        Args:
            date (pd.Timestamp): The date for which the BoD processing is being performed.

        Notes:
            - Adds distributions to daily PNL.
        """
        divs = super().BoD(date)
        if divs:
            sum_divs = .0
            for ticker, div in divs.items():
                paid = self.get_eod_shares(date - pd.Timedelta(days=1), ticker) * div
                self.detail[date][ticker]['Divs'] += paid
                sum_divs += paid
            self.pnl[date] += sum_divs
    run_beginning_of_day = BoD  # Alias that complies with method naming convention

    def is_in_detail(self, date: pd.Timestamp, ticker: str) -> bool:
        """Check if the ticker is in detail for the date.

        Args:
            date (pd.Timestamp): The date to check.
            ticker (str): The asset ticker symbol.

        Returns:
            bool: True if the ticker is in detail for the date, False otherwise.
        """
        return (date in self.detail) and (ticker in self.detail[date])

    def EoD(self, date: pd.Timestamp):
        """End of Day: Calculates daily P&L and NAV using closing price.
            Must be called after all transactions are made for the date.

        Args:
            date (pd.Timestamp): The date for which the EoD processing is being performed.
        """
        super().EoD(date)
        if date not in self.trading_dates:
            return
        total_exposure = .0
        for ticker, positions in self.positions.items():
            #region Exclude tickers with no positions or activity
            # Remove this region to include daily Px in self.detail for every ticker
            #   previously traded even when they have no position or activity.
            if len(positions) == 0 and not self.is_in_detail(date, ticker):
                continue
            #endregion Exclude tickers with no positions
            # Calculate NAV and P&L if we have pricing data
            if (px := self.get_price(date, ticker, fill=False)) is not None:
                shares = .0
                prev_px = self.data.loc[(ticker, date), 'PrevPx']
                for position in positions:
                    shares += position.shares
                    if position.open_date == date:  # Positions bought today
                        # Today's price P&L is from purchase price, not previous close
                        price_return = px - position.open_px
                    else:
                        price_return = px - prev_px
                    gain = position.shares * price_return
                    self.pnl[date] += gain
                    self.detail[date][ticker]['PNL'] += gain
                self.detail[date][ticker]['SharesEoD'] = shares
                self.detail[date][ticker]['Px'] = px
                self.detail[date][ticker]['Lots'] = len(positions)
                total_exposure += shares * px
        self.nav[date] = total_exposure
        if not self.pnl[date]:
            self.pnl[date] = .0
    run_end_of_day = EoD  # Alias that complies with method naming convention

    def validate(self):
        """Run checks that should always be true.

        Raises:
            AssertionError: If the validation checks fail.
        """
        super().validate()
        for date, pnl in self.pnl.items():
            detail_pnl = sum(details['PNL'] + details['Divs'] for ticker, details in self.detail[date].items())
            assert math.isclose(pnl, detail_pnl, rel_tol=1e-6), f"P&L validation on {date}"
            detail_nav = sum(details['SharesEoD'] * details['Px'] for ticker, details in self.detail[date].items())
            assert math.isclose(self.nav[date], detail_nav, rel_tol=1e-6), f"NAV validation on {date}"

    #region Properties
    @property
    def last_end_of_day(self) -> Optional[pd.Timestamp]:
        """Return the last date on which the tracker ran EoD.

        Returns:
            Optional[pd.Timestamp]: The last date on which the tracker ran EoD.
        """
        return self._last_EoD

    @property
    def daily_nav_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily NAV.

        Returns:
            pd.DataFrame: A DataFrame of daily NAV.
        """
        return pd.DataFrame.from_dict(self.nav, orient='index', columns=['NAV'])

    @property
    def daily_pnl_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily P&L.

        Returns:
            pd.DataFrame: A DataFrame of daily P&L.
        """
        return pd.DataFrame.from_dict(self.pnl, orient='index', columns=['PNL'])

    @property
    def daily_returns_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily PNL, NAV, returns and log returns.

        Returns:
            pd.DataFrame: A DataFrame of daily PNL, NAV, returns and log returns.
        """
        df = self.daily_pnl_df.join(self.daily_nav_df.shift(1), how='left')
        df['Return'] = df['PNL'] / df['NAV']
        df['LogReturn'] = df['Return'].apply(lambda x: math.log(1+x))
        return df.dropna()

    @property
    def daily_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily tracker-level values.

        Returns:
            pd.DataFrame: A DataFrame of daily tracker-level values.
        """
        return self.daily_pnl_df.join(self.daily_nav_df.shift(1), how='left') \
            .join(pd.DataFrame.from_dict(self.opened_tickers_count, orient='index', columns=['OpenedTickers'])) \
            .join(pd.DataFrame.from_dict(self.closed_tickers_count, orient='index', columns=['ClosedTickers']))

    @property
    def detail_df(self) -> pd.DataFrame:
        """Return a DataFrame of asset-level `detail` values indexed by Date, Ticker.

        Returns:
            pd.DataFrame: A DataFrame of asset-level `detail` values indexed by Date, Ticker.
        """
        data = [
            {'Date': date, 'Ticker': ticker, **details}
            for date, tickers in self.detail.items()
            for ticker, details in tickers.items()
        ]
        df = pd.DataFrame(data)
        df.set_index(['Date', 'Ticker'], inplace=True)
        return df
    #endregion Properties
