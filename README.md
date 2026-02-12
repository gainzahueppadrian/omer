# LEAPS PUTS Multi-Agent Autonomous Trading System

## Overview

A production-grade autonomous trading system for selling LEAPS puts on quality
stocks, with optional bonus LEAPS call purchases funded by put premiums.

The system uses a **multi-agent architecture** where specialized agents handle
scanning, risk management, execution, and portfolio monitoring, coordinated by
a central orchestrator via a priority message bus.

## Key Features

### Mathematical Edge
- **Risk-Neutral Density (RND) Models**: Goes beyond Black-Scholes delta to
  compute true market-implied assignment probabilities
- **Malz (1997) Smile Interpolation**: Captures volatility skew from market data
- **Melick & Thomas (1997) Mixture of Log-Normals**: Models fat tails and
  bimodal distributions in the risk-neutral measure
- **"Risk of Reversal"**: Market-implied assignment probability incorporating
  skew and kurtosis, not just delta

### Multi-Agent Architecture
- **Scanner Agent**: Autonomously scans watchlist for opportunities
- **Risk Agent**: Dynamic risk management with regime detection
- **Execution Agent**: Smart order routing with optimal pricing
- **Portfolio Agent**: Real-time P&L, Greeks, and position management
- **Orchestrator**: Central coordination and health monitoring
- **Message Bus**: Priority-based pub/sub for agent communication

### Strategy
- Sell LEAPS puts (9-18 months) on quality stocks
- Target 30-45 delta, 10%+ annualized return
- Maximum 25% risk of reversal (RND-based)
- Optional: Buy LEAPS calls funded by put premium (bonus strategy)
- Dynamic risk adjustment based on market regime (high/normal/low vol)

## Requirements

- Python 3.10+
- Interactive Brokers TWS or IB Gateway
- IBKR account with options trading permissions

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/settings.yaml`:

- **IBKR connection**: host, port, client_id
- **Watchlist**: stocks to scan
- **Strategy parameters**: DTE range, delta range, return targets
- **Risk parameters**: portfolio limits, margin safety, regime thresholds

## Usage

### Paper Trading (Default)
```bash
python -m src.main
```

### Custom Config
```bash
python -m src.main --config path/to/config.yaml
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  ORCHESTRATOR                        │
│                                                     │
│  Scanner ──→ Risk Agent ──→ Execution Agent         │
│     ↑              ↑              ↓                 │
│     │              │         ┌────────────┐         │
│     └──────────────┼─────────│  Portfolio  │         │
│                    └─────────│   Agent     │         │
│                              └────────────┘         │
│                        ↕                            │
│              ┌──────────────────┐                   │
│              │   Message Bus    │                   │
│              └──────────────────┘                   │
│                        ↕                            │
│              ┌──────────────────┐                   │
│              │   IBKR Client    │                   │
│              └──────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

## Workflow

1. **Scanner** periodically scans watchlist for LEAPS put opportunities
2. For each symbol: fetches price, option chain, volatility smile
3. Calibrates RND model (Malz + Melick-Thomas) for each name
4. Scores candidates by return, safety, vega, and skew attractiveness
5. **Risk Agent** receives candidates, detects market regime, adjusts parameters
6. Approves/rejects based on dynamic risk criteria
7. **Execution Agent** places orders with smart limit pricing
8. Optionally executes bonus call strategy
9. **Portfolio Agent** monitors positions for profit targets, loss limits,
   IV crush, and roll opportunities
10. **Orchestrator** coordinates everything, monitors health, logs activity

## Risk Management

The system dynamically adjusts risk based on:

| Condition | Response |
|-----------|----------|
| High IV (VIX > 35) | Reduce allocation 40%, widen margin safety |
| Low IV (VIX < 15) | Slight allocation increase |
| Steep negative skew | Allocation bonus (puts are rich) |
| High kurtosis (>4) | Reduce allocation for fat tail risk |
| Delta > 0.55 | Reject trade |
| RoR > 25% | Reject trade |
| Profit > 50% | Close position |
| Loss > 200% | Emergency close |

## Disclaimer

This software is for educational and research purposes. Options trading
involves substantial risk. Past performance does not guarantee future results.
Always test thoroughly in paper trading before any live deployment.
