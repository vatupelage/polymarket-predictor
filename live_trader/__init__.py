"""Live trading bot for Polymarket BTC up/down 5m markets.

HIGH-same strategy only: when the v4 predictor emits confidence >= HIGH_CONF
threshold, buy the share matching the predicted direction on Polymarket CLOB.
"""
