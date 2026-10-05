# GME January 2021 JEV case study

**RETROSPECTIVE CASE STUDY ONLY; not out-of-sample evidence**

JEV scored 665 GME-tickered articles; 305 were later revised. A timestamp-safe minute entry was available for 661 articles.

Buy and hold, Jan 22 close to Feb 4 close: GME -17.7%; SPY 0.9%.
The best intraday high after Jan 22 was 483.00 on Jan 28; that was 643.0% above the Jan 22 close, before the subsequent reversal.

Spearman(signal, same-day close return) = 0.08493774729131474; Spearman(signal, 5-session return) = 0.08401623437159395.

This is an inference demonstration, not a clean backtest: JEV postdates the event, there are no manual labels, and the historical window is tiny.

- JEV was released years after January 2021 and may have memorized the squeeze; all outputs are post-hoc.
- Historical story text has no manual ground-truth sentiment or materiality labels, so predictive correlation does not measure classifier accuracy.
- This event window has very few independent daily observations and no executable quote/spread or market-impact data.
- The five-minute availability delay is unverified; revised text is delayed to the vendor update time.
- This result is one famous episode selected after its outcome; it cannot support an alpha claim.
