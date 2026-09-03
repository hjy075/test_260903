# Retail Forecast Survivorship Gate-0.5

Research spike for the question:

**Do Surviving Products Make Retail Forecasts Look Better?**  
**살아남은 상품만 보면 소매 수요예측 성능이 더 좋아 보이는가?**

This repository runs a low-cost Gate-0.5 falsification test on the public M5 retail dataset before training forecasting models.

The first gate asks whether, at multiple forecast origins, item-store series that continue to be sold 26/52 weeks later are sufficiently common and measurably different *at the forecast origin* from item-store series whose selling presence does not continue.

Data are downloaded at runtime and are not committed to this repository.
