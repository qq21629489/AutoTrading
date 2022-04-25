import pytest
import pandas as pd

from profit_calculator import (
    calculate_profit, check_stock_actions_length, StockTrader,
    InvalidActionError, StockNumExceedError, InvalidActionNumError
)


@pytest.fixture(scope='function')
def stocks():
    FEATURE_NAMES = ('open', 'high', 'low', 'close')
    return pd.read_csv('tests/testing_data.csv', names=FEATURE_NAMES)


@pytest.fixture(scope='function')
def init_actions(stocks):
    return [0] * (len(stocks) - 1)


class TestCalculateProfit:
    def test_buy_and_hold(self, stocks, init_actions):
        actions = init_actions
        actions[0] = 1

        profit = calculate_profit(stocks, actions)
        assert profit == pytest.approx(45.037079)

    def test_sellshort_and_hold(self, stocks, init_actions):
        actions = init_actions
        actions[0] = -1

        profit = calculate_profit(stocks, actions)
        assert profit == pytest.approx(-45.037079)

    def test_holding_more_than_one(self, stocks, init_actions):
        actions = init_actions
        actions[0] = 1
        actions[2] = 1

        with pytest.raises(StockNumExceedError) as err_info:
            profit = calculate_profit(stocks, actions)

        assert err_info.value.args[0] == 'You cannot buy stocks when you hold one'

    def test_selling_short_more_than_one(self, stocks, init_actions):
        actions = init_actions
        actions[0] = -1
        actions[2] = -1

        with pytest.raises(StockNumExceedError) as err_info:
            profit = calculate_profit(stocks, actions)

        assert err_info.value.args[0] == "You cannot sell short stocks when you've already sell short one"

    def test_invalid_action(self, stocks, init_actions):
        actions = init_actions
        actions[0] = 'Invalid'

        with pytest.raises(InvalidActionError) as err_info:
            profit = calculate_profit(stocks, actions)

        assert err_info.value.args[0] == 'Invalid Action'


class TestCheckStockActionsLength:
    def test_valid_length(self, stocks, init_actions):
        assert check_stock_actions_length(stocks, init_actions) is True

    def test_invalid_length(self, stocks, init_actions):
        actions = init_actions
        actions.append(0)

        assert check_stock_actions_length(stocks, init_actions) is False


class TestStockTrader:
    @pytest.fixture(scope='function')
    def stock_tracker(self):
        return StockTrader()

    def test_is_not_holding_stock(self, stock_tracker):
        assert stock_tracker.is_holding_stock is False

    def test_is_holding_stock(self, stock_tracker):
        stock_tracker.holding_price = 0

        assert stock_tracker.is_holding_stock is True

    def test_is_not_shorting_stock(self, stock_tracker):
        assert stock_tracker.is_holding_stock is False

    def test_is_shorting_stock(self, stock_tracker):
        stock_tracker.sell_short_price = 0

        assert stock_tracker.is_holding_stock is False

    def test_buy_when_not_holding(self, stock_tracker):
        stock_tracker.buy(10)

        assert stock_tracker.holding_price == 10
        assert stock_tracker.is_holding_stock is True
        assert stock_tracker.is_shorting_stock is False
        assert stock_tracker.accumulated_profit == 0

    def test_sell_when_not_shorting(self, stock_tracker):
        stock_tracker.sell(15)

        assert stock_tracker.sell_short_price == 15
        assert stock_tracker.is_holding_stock is False
        assert stock_tracker.is_shorting_stock is True
        assert stock_tracker.accumulated_profit == 0

    def test_buy_when_shorting_stock(self, stock_tracker):
        stock_tracker.sell(15)
        stock_tracker.buy(10)

        assert stock_tracker.is_holding_stock is False
        assert stock_tracker.is_shorting_stock is False
        assert stock_tracker.accumulated_profit == 5

    def test_buy_when_holding_stock(self, stock_tracker):
        stock_tracker.buy(10)

        with pytest.raises(StockNumExceedError) as err_info:
            stock_tracker.buy(15)

        assert err_info.value.args[0] == 'You cannot buy stocks when you hold one'

    def test_sell_when_holding_stock(self, stock_tracker):
        stock_tracker.buy(10)
        stock_tracker.sell(15)

        assert stock_tracker.is_holding_stock is False
        assert stock_tracker.is_shorting_stock is False
        assert stock_tracker.accumulated_profit == 5

    def test_sell_when_shorting_stock(self, stock_tracker):
        stock_tracker.sell(10)

        with pytest.raises(StockNumExceedError) as err_info:
            stock_tracker.sell(15)

        assert err_info.value.args[0] == "You cannot sell short stocks when you've already sell short one"
