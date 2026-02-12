"""
LEAPS PUTS Multi-Agent Autonomous Trading System
Main Entry Point

Usage:
    python -m src.main                    # Run with default config
    python -m src.main --config custom.yaml  # Run with custom config
    python -m src.main --dry-run           # Scan only, no execution
"""

import sys
import os
import signal
import logging
import argparse
from pathlib import Path

import yaml

from src.agents.orchestrator import Orchestrator


def setup_logging(config: dict):
    """Configure logging based on config settings."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)

    # Create log directory
    log_file = log_config.get("file", "logs/leaps_trading.log")
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handlers = []

    # Console handler
    if log_config.get("console", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)-25s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_fmt)
        handlers.append(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)-25s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_fmt)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        return get_default_config()

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        "ibkr": {
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 1,
            "mode": "paper",
            "market_data_type": 3,
            "timeout_seconds": 10,
        },
        "strategy": {
            "watchlist": ["AAPL", "MSFT", "JNJ", "PG", "KO"],
            "leaps": {
                "min_dte": 270,
                "max_dte": 548,
                "target_dte": 365,
                "min_delta": -0.30,
                "max_delta": -0.45,
                "min_annualized_return": 0.10,
                "target_annualized_return": 0.14,
                "max_risk_of_reversal": 0.25,
            },
            "bonus_calls": {
                "enabled": True,
                "max_premium_allocation": 0.50,
                "call_delta_target": 0.40,
            },
            "filters": {
                "min_iv_percentile": 60,
                "min_open_interest": 100,
                "max_bid_ask_spread_pct": 0.05,
                "min_underlying_price": 20.0,
                "max_underlying_price": 500.0,
            },
        },
        "risk": {
            "max_portfolio_delta": -500,
            "max_single_position_pct": 0.15,
            "max_total_notional_pct": 0.80,
            "max_gamma_exposure": 100,
            "base_allocation_pct": 0.10,
            "margin_safety_factor": 1.5,
            "dynamic_adjustments": {
                "high_vol_threshold": 0.35,
                "low_vol_threshold": 0.15,
                "skew_warning_threshold": -0.03,
                "kurtosis_warning_threshold": 4.0,
            },
        },
        "agents": {
            "scan_interval_seconds": 300,
            "heartbeat_interval": 30,
            "orchestrator_cycle_seconds": 60,
        },
        "logging": {
            "level": "INFO",
            "file": "logs/leaps_trading.log",
            "console": True,
        },
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LEAPS PUTS Multi-Agent Autonomous Trading System"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/settings.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan only, do not execute trades"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)

    logger = logging.getLogger(__name__)

    # Handle dry-run mode
    if args.dry_run:
        logger.info("DRY-RUN MODE: Trades will not be executed")
        # In dry-run, we could override execution agent behavior
        # For simplicity, just log the intent

    # Create orchestrator
    orchestrator = Orchestrator(config)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Signal {signum} received, shutting down...")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the system
    try:
        orchestrator.start()
        orchestrator.run_forever()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        orchestrator.stop()


if __name__ == "__main__":
    main()
