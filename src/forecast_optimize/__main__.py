"""Command-line interface: ``python -m forecast_optimize <command>``.

Commands
--------
generate     write the synthetic data set to CSV
ingest       generate data and load it into the database (runs migrations)
experiment   run the full back-test, write results/ and docs/figures/
tables       print the README result tables from results/metrics.json
plan         forecast and plan one day, print the plan
serve        start the REST API (uvicorn)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecast_optimize", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write synthetic data to CSV")
    g.add_argument("--out", default="data/synthetic_load.csv")
    g.add_argument("--seed", type=int, default=7)

    i = sub.add_parser("ingest", help="load synthetic data into the database")
    i.add_argument("--db", default=None, help="database URL (default: FO_DATABASE_URL or sqlite file)")
    i.add_argument("--seed", type=int, default=7)

    e = sub.add_parser("experiment", help="run the full back-test")
    e.add_argument("--out", default="results")
    e.add_argument("--figures", default="docs/figures")

    sub.add_parser("tables", help="print README tables from results/metrics.json").add_argument(
        "--results", default="results/metrics.json"
    )

    p = sub.add_parser("plan", help="forecast and plan one day")
    p.add_argument("--date", default="2024-08-14")
    p.add_argument("--strategy", default="stochastic", choices=["deterministic", "quantile", "stochastic"])
    p.add_argument("--cutoff", default="2024-07-01")

    s = sub.add_parser("serve", help="run the REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        from forecast_optimize.data.synthetic import SyntheticConfig, generate

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = generate(SyntheticConfig(seed=args.seed))
        df.to_csv(out, float_format="%.4f")
        print(f"wrote {len(df)} rows to {out}")
    elif args.cmd == "ingest":
        from forecast_optimize.data.synthetic import SyntheticConfig, generate
        from forecast_optimize.storage.db import connect

        db = connect(args.db)
        applied = db.migrate()
        n = db.upsert_measurements(generate(SyntheticConfig(seed=args.seed)))
        print(f"migrations applied: {applied or 'none'}; upserted {n} measurement rows")
    elif args.cmd == "experiment":
        from forecast_optimize.experiment import markdown_tables, run

        result = run(args.out, args.figures)
        print(markdown_tables(result))
        print(json.dumps(result["timings"], indent=2))
    elif args.cmd == "tables":
        from forecast_optimize.experiment import markdown_tables

        print(markdown_tables(json.loads(Path(args.results).read_text())))
    elif args.cmd == "plan":
        import pandas as pd

        from forecast_optimize.data.synthetic import generate
        from forecast_optimize.optimize.dayahead import MarketParams, realized_cost
        from forecast_optimize.pipeline import ForecastService

        data = generate()
        svc = ForecastService(data, pd.Timestamp(args.cutoff))
        idx, plan, prices = svc.plan(args.date, args.strategy, MarketParams())
        actual = data.loc[idx, "load_mw"].to_numpy()
        frame = pd.DataFrame(
            {
                "price": prices,
                "buy_MWh": plan.buy,
                "charge_MW": plan.charge,
                "discharge_MW": plan.discharge,
                "soc_end_MWh": plan.soc[1:],
                "actual_MW": actual,
            },
            index=idx,
        )
        print(frame.round(1).to_string())
        cost = realized_cost(plan, prices, actual)
        print(f"capacity nominated: {plan.capacity:.1f} MW; realised cost: {cost['total'][0]:,.0f} EUR")
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("forecast_optimize.api.app:default_app", host=args.host, port=args.port, factory=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
