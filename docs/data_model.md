# The synthetic data-generating process

The generator (`src/forecast_optimize/data/synthetic.py`) produces hourly load,
temperature, a day-ahead temperature forecast and day-ahead prices from 2020-01-01
to 2024-12-31 (43 848 hours). One `numpy.random.Generator` seeded by
`SyntheticConfig.seed` (default 7) drives everything, so a seed fixes the output
exactly (`tests/test_data.py::test_generator_reproducible_and_seeded`).

Time stamps are naive local standard time (CET). Daylight saving time is
ignored so that every day has 24 hours; a production system would store UTC
and handle the 23- and 25-hour days explicitly.

Notation: $t$ is the hour index, $h(t)\in\{0,\dots,23\}$ the hour of day,
$s(t)$ the day of the year, $d(t)$ the day index, and $k(t)\in\{0,1,2\}$ the
day type (working day, Saturday, Sunday or public holiday). An AR(1) process
with coefficient $\phi$ and stationary standard deviation $\sigma$ is written
$\mathrm{AR}(\phi,\sigma)$: $X_i=\phi X_{i-1}+\sigma\sqrt{1-\phi^2}\,\varepsilon_i$
with $\varepsilon_i\sim N(0,1)$ i.i.d.

## Calendar

Slovenian public holidays follow the *Zakon o praznikih in dela prostih dnevih*:
1-2 January (2 January was a working day in 2013-2016), 8 February, Easter
Sunday and Monday, 27 April, 1-2 May, Whit Sunday (Easter + 49 days), 25 June,
15 August, 31 October, 1 November, 25-26 December, and the one-off 14 August 2023.
Easter uses the anonymous Gregorian algorithm. The implementation was checked
against the `holidays` package for 2005-2030 (identical sets). A *bridge day* is
a working day with non-working days on both sides.

## Temperature

$$
T_t = \bar T - A\cos\!\Big(\tfrac{2\pi(s-s_0)}{365.25}\Big)
      + D(s)\cos\!\Big(\tfrac{2\pi(h-15)}{24}\Big) + Z_t + \nu_t ,
$$

with $\bar T = 11$ °C, $A = 10.5$ °C, coldest day $s_0 = 20$, and a diurnal
amplitude $D(s)$ that moves between 3 °C (winter) and 6 °C (summer).
$Z_t$ is a daily weather-regime anomaly, $\mathrm{AR}(0.8, 3)$ in days and
linearly interpolated between noons, and $\nu_t \sim \mathrm{AR}(0.7, 0.5)$
is hourly noise.

The **day-ahead temperature forecast** is $T^{fc}_t = T_t + e_t$ with
$e \sim \mathrm{AR}(0.95, 1.2)$ (hourly), so forecast errors persist for many hours,
as they do in practice. Models only see $T^{fc}$ for the target day.

Buildings react to temperature with a delay, so load depends on the
exponentially smoothed temperature $\tilde T_t = (1-\lambda)\tilde T_{t-1} + \lambda T_t$
with a half-life of 12 hours.

## Load

$$
y_t = m_t + \sigma_t\,\eta_t, \qquad \eta \sim \mathrm{AR}(0.9, 1),
$$

$$
m_t = \mu_0\,(1 + g\,t_{\text{years}})\,L_{d}\;P_{k}(h)\;F_t\;G_t
      \;+\;\beta_H\,\mathrm{HDD}_t\;+\;\beta_C\,c(h)\,\mathrm{CDD}_t ,
$$

$$
\mathrm{HDD}_t = \max(0,\,T_H - \tilde T_t), \qquad
\mathrm{CDD}_t = \max(0,\,\tilde T_t - T_C).
$$

| symbol | meaning | value |
|---|---|---|
| $\mu_0$, $g$ | base level, annual growth | 1000 MW, 1 %/year |
| $L_d$ | unobserved slow activity level, $\log L \sim \mathrm{AR}(0.97, 0.025)$ in days | |
| $P_k(h)$ | intraday profile of day type $k$: logistic morning ramp, evening decline, midday and evening bumps; Saturday and Sunday levels are 0.86 and 0.78 of a working day | |
| $F_t$ | special-day factor: bridge day 0.92, 24 Dec-6 Jan 0.93, 1-25 Aug 0.95 | |
| $G_t$ | winter-evening lighting peak (up to +4 % around 18:00 near the winter solstice) | |
| $T_H$, $\beta_H$ | heating threshold and slope | 15 °C, 9 MW/°C |
| $T_C$, $\beta_C$, $c(h)$ | cooling threshold, slope, afternoon-peaked shape in $[0.6, 1]$ | 21 °C, 22 MW/°C |

The temperature response $T \mapsto \beta_H(T_H-T)_+ + \beta_C(T-T_C)_+$ is
piecewise linear and convex (a "hockey stick" on both sides) with a dead band
between 15 and 21 °C. It is nonlinear in $T$, which is why the linear baseline
uses a hinge basis.

**Heteroskedasticity.**

$$
\sigma_t = s_0\, m_t\,\big(1 + \kappa(\mathrm{HDD}_t + \mathrm{CDD}_t)\big)\,\big(1 + 0.5\cdot\mathbb 1\{\text{holiday or bridge}\}\big),
$$

with $s_0 = 2.2\,\%$ and $\kappa = 0.03$. Noise is proportional to the level,
larger in extreme weather and on irregular days. A single global residual
quantile therefore cannot be calibrated everywhere, and this is what makes
the conformal calibration study non-trivial.

## Day-ahead prices

$$
\pi_t = a_{d} + 28\,\mathrm{shape}(h, s) + 0.12\,(m_t - \bar m) + \epsilon_t, \qquad
a \sim 95 + \mathrm{AR}(0.85, 18) \text{ (daily)}, \quad \epsilon_t \sim N(0, 6^2),
$$

where the shape has morning and evening peaks and a midday solar dip that is
deeper in summer; prices are floored at 5 EUR/MWh. Prices enter only the
optimisation and are treated as known when the plan is made (see the README,
*Limitations*).

## Stylised facts checked by tests

`tests/test_data.py::test_generator_stylised_facts` checks that the night
minimum and daytime maximum fall in the right hours, that Sunday < Saturday <
weekdays, that the load-temperature curve is U-shaped, and that the noise is
larger in cold weather.
