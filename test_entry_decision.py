from live_trader.db_decision import entry_decision


def test_enter_when_cheap_and_before_deadline():
    assert entry_decision(0.48, secs_to_close=120, max_ask=0.50, deadline_s=30) == "enter"

def test_enter_at_exactly_max_ask():
    assert entry_decision(0.50, secs_to_close=120, max_ask=0.50, deadline_s=30) == "enter"

def test_wait_when_expensive_and_before_deadline():
    assert entry_decision(0.62, secs_to_close=120, max_ask=0.50, deadline_s=30) == "wait"

def test_skip_when_past_deadline_and_not_cheap():
    assert entry_decision(0.62, secs_to_close=20, max_ask=0.50, deadline_s=30) == "skip"

def test_enter_takes_priority_at_deadline_if_cheap():
    # cheap wins even if we are at/under the deadline
    assert entry_decision(0.49, secs_to_close=20, max_ask=0.50, deadline_s=30) == "enter"

def test_wait_when_ask_missing():
    assert entry_decision(None, secs_to_close=120, max_ask=0.50, deadline_s=30) == "wait"

def test_skip_when_ask_missing_past_deadline():
    assert entry_decision(None, secs_to_close=20, max_ask=0.50, deadline_s=30) == "skip"

def test_standard_bot_enters_on_first_poll_when_max_ask_one():
    # MAX_ASK=1.0 -> any real ask <= 1.0 -> immediate enter (standard behavior)
    assert entry_decision(0.87, secs_to_close=240, max_ask=1.0, deadline_s=0) == "enter"
