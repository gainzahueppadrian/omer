import argparse
import time
import sys
from src.client.ib_client import IBClient
from src.agents.risk_agent import RiskAgent
from src.strategy.leaps_puts import LeapsPutsStrategy

def main():
    parser = argparse.ArgumentParser(description='LEAPS Puts Trading Bot')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode')
    parser.add_argument('--symbols', type=str, default='SPY,AAPL,MSFT,TSLA,NVDA', help='Comma-separated list of symbols')
    args = parser.parse_args()

    # 1. Initialize Components
    print(f"Initializing Bot (Mock Mode: {args.mock})...")
    client = IBClient(mock=args.mock)

    if not client.connect_sync():
        print("Failed to connect. Exiting.")
        sys.exit(1)

    risk_agent = RiskAgent(max_risk_of_reversal=0.40)
    strategy = LeapsPutsStrategy(client, risk_agent)

    symbols = args.symbols.split(',')

    # 2. Main Loop (Run Once for Demo)
    print(f"Scanning opportunities for: {symbols}")

    try:
        opportunities = strategy.scan_opportunities(symbols, expiry_months=12)

        print(f"\nFound {len(opportunities)} opportunities:")
        for opp in opportunities:
            print("-" * 50)
            print(f"SYMBOL: {opp['symbol']}")
            print(f"STRIKE: {opp['strike']:.2f}")
            print(f"PREMIUM: ${opp['premium']:.2f}")
            print(f"ANN. ROC: {opp['roc_annualized']:.2%}")
            print(f"RISK OF REVERSAL: {opp['risk_of_reversal']:.2%}")
            print(f"ALLOCATION: {opp['allocation_mult']:.2f}x")
            print(f"REASON: {opp['reason']}")

            # Here we would place the order if allocation > 0
            if opp['allocation_mult'] > 0:
                print(">> ACTION: PLACING ORDER (Simulated)")
                contract = strategy.make_contract(opp['symbol'])
                # Create Order object...
                # client.place_order_sync(...)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if not args.mock:
            client.disconnect()

if __name__ == "__main__":
    main()
