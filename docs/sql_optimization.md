# SQL optimisation study: the forecast store

Forecast stores grow quickly. A run issued every 6 hours with a 48-hour horizon
and 7 quantile levels writes 336 rows, which is about **1 million rows in two
years** for a single model and a single site. The API and the reports read
this table in a few typical ways. This note measures those queries before and
after migration [`002_forecast_indexes.sql`](../src/forecast_optimize/storage/migrations/sqlite/002_forecast_indexes.sql)
and compares several ways of writing each one.

Everything below is reproducible with

```bash
python scripts/sql_benchmark.py                       # SQLite only
FO_BENCH_POSTGRES_URL=postgresql://... python scripts/sql_benchmark.py   # + PostgreSQL 16
```

The script builds the data set (981 288 forecast values, 2 924 runs, 17 544
measurements), runs every query variant, **asserts that all variants of a query
return identical rows**, and writes the timings and every `EXPLAIN QUERY PLAN` to
[`sql_benchmark_results.md`](sql_benchmark_results.md). The numbers quoted here
come from that file: medians of 5 runs on one laptop-class x86 core, SQLite
3.45.1 and PostgreSQL 16. Treat them as orders of magnitude, not exact figures.

## Schema recap

```sql
forecast_values(run_id, target_ts, horizon_h, quantile, value_mw,
                PRIMARY KEY (run_id, target_ts, quantile))
```

The primary key matches the **write** pattern, where a run inserts its rows
together. The **read** pattern is different. Reads ask for "a quantile, a time
range of target hours, and the most recent issue of each hour". Migration 002
adds an index for that pattern:

```sql
CREATE INDEX ix_fv_quantile_target_horizon
    ON forecast_values (quantile, target_ts, horizon_h, value_mw);  -- SQLite
-- PostgreSQL: (quantile, target_ts, horizon_h) INCLUDE (value_mw)
```

The column order follows the usual rule: **equality columns first, then the
range column, then the column used for ordering**, and finally the selected
value, so that the index alone answers the query (a *covering* index).

## Q1: latest forecast for every target hour (30 days, median quantile)

"Latest" means the smallest horizon, i.e. the most recent forecast issued for
that hour. This is a classic *greatest-n-per-group* query.

| variant (SQLite) | before 002 [ms] | after 002 [ms] |
|---|---:|---:|
| correlated subquery `horizon_h = (SELECT MIN(...) WHERE f2.target_ts = fv.target_ts)` | 28 512 | 3.7 |
| aggregate `best` + join | 3 626 | 1.4 |
| aggregate + join, **range repeated on the joined table** | 30.5 | 1.4 |
| `ROW_NUMBER() OVER (PARTITION BY target_ts ORDER BY horizon_h)` | 19.7 | 3.6 |
| SQLite-only bare column with `MIN()` | 14.9 | 1.05 |

| variant (PostgreSQL 16) | before 002 [ms] | after 002 [ms] |
|---|---:|---:|
| correlated subquery | > 60 000 (cancelled) | 11.3 |
| aggregate + join | 108 | 61.4 |
| aggregate + join, range repeated | 215 | 3.6 |
| `ROW_NUMBER()` window | 51.1 | 3.3 |

What the plans show:

* **Without the index, SQLite still avoids a full scan.** After `ANALYZE` it
  uses a *skip-scan* of the primary key, `ANY(run_id) AND target_ts>? AND
  target_ts<?`: one B-tree probe for every distinct `run_id` (2 924 of them).
  That works for a single range lookup. It is disastrous when it is repeated:
  the correlated subquery repeats it for each of the 5 760 candidate rows
  (28 s), and the plain join repeats it for each of the 720 groups (3.6 s).
* **The covering index turns every variant into a range scan** (`SEARCH ...
  USING COVERING INDEX ix_fv_quantile_target_horizon (quantile=? AND
  target_ts>? AND target_ts<?)`), with speed-ups from 5× to 7 700×.
* **Repeating the range predicate on the joined table** is logically redundant,
  because the join already restricts `target_ts`. It still matters. Without
  it, SQLite probes the joined side once per group, and PostgreSQL chooses a
  hash join over a sequential scan of all median-quantile rows (61 ms even with
  the index). With it, PostgreSQL drops to 3.6 ms. Planners do not always
  propagate a range across an equi-join, so restating it costs nothing and can
  help a lot.
* **The window-function form is robust but not always the fastest.** On SQLite
  it pays for a sort and the window machinery (3.6 ms against 1.4 ms). On
  PostgreSQL it ties with the rewritten join.
* The SQLite *bare column* trick (`SELECT target_ts, MIN(horizon_h), value_mw
  ... GROUP BY target_ts`) is the fastest, but it relies on documented SQLite
  behaviour that is an error in standard SQL and in PostgreSQL. It is not used
  in the application.

**Decision:** `Database.latest_forecast` uses the aggregate + join form with
the repeated range. It is one portable query that sits at or near the fastest
variant on both engines.

## Q2: a whole day of forecasts (one quantile, all horizons)

| variant (SQLite) | before 002 [ms] | after 002 [ms] |
|---|---:|---:|
| `WHERE date(target_ts) = :day` | 87.3 | 17.3 |
| `WHERE target_ts >= :day AND target_ts < date(:day, '+1 day')` | 7.4 | **0.12** |

Wrapping an indexed column in a function makes the predicate *non-sargable*:
the engine cannot turn `date(target_ts) = ...` into an index range. Before the
index the function form even falls back to a full `SCAN forecast_values`, while
the range form can use the skip-scan. After the index the function form uses
only the `quantile=?` prefix and evaluates `date()` on 140 000 rows. The
equivalent half-open range is **about 140× faster** with the index. The fix is
to rewrite the predicate, not to add another index. (PostgreSQL can index the
expression instead, but the plain range keeps the query portable.)

ISO-8601 text time stamps (`YYYY-MM-DD HH:MM:SS`) sort lexicographically in
chronological order, which is why range predicates on SQLite `TEXT` columns
are correct.

## Q3: rolling 7-day MAE of the latest forecasts (91 days)

| variant (SQLite) | before 002 [ms] | after 002 [ms] |
|---|---:|---:|
| correlated subquery: `AVG` over `ts` in `(t - 168 h, t]` for each row | 1 047 | 940 |
| `AVG(abs_err) OVER (ORDER BY ts ROWS BETWEEN 167 PRECEDING AND CURRENT ROW)` | 62.2 | **18.3** |

The correlated form re-reads the materialised error table once per output row,
which is $O(n^2)$ (2 184 × 2 352 comparisons). The index cannot help: the CTE
it scans has no index, and only the small "latest forecast" step (Q1) gets
faster. The window function computes the moving average in one ordered pass,
which is $O(n)$ after sorting: **51× faster with the index, 17× without**.

`ROWS BETWEEN 167 PRECEDING` equals the 168-hour time window only because the
series has no gaps. On data with missing hours, use a `RANGE` frame on a
numeric time column or fill the gaps first. The benchmark checks that both
variants give identical results on its gap-free data.

## Take-aways (the checklist I would apply to a production store)

1. **Index for the read path.** Put equality columns first, then the range
   column, then the ordering column. Include the selected value to make the
   index covering. Check the result with `EXPLAIN QUERY PLAN` / `EXPLAIN`.
2. **Keep predicates sargable.** Write half-open ranges on the raw column and
   never wrap the column in a function.
3. **Replace correlated subqueries** with window functions or aggregate + join.
4. **Help the planner across joins** by restating selective predicates on both
   sides.
5. **Measure on the target engine.** The fastest formulation differed between
   SQLite and PostgreSQL. Run `ANALYZE` before measuring: SQLite's skip-scan is
   chosen only when statistics exist.
6. On PostgreSQL, run `VACUUM` after bulk loads so that index-only scans can be
   used (the benchmark's freshly loaded table shows bitmap heap scans instead).

At larger scale, the next steps would be range partitioning of
`forecast_values` by `target_ts` month (PostgreSQL declarative partitioning),
BRIN indexes for append-only time columns, and a materialised "latest
forecast" table maintained on insert.
