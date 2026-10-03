# Causal evaluation repair

The original trainer fit feature and target scalers on the entire dataset before
its chronological split. It also appended a `Close` value from 29 rows ahead to
every input row before making 30-row windows. The final input row therefore
contained a future close at every forecast origin. Those results, including the
old sub-1% MAPE, do not establish held-out performance. Existing checkpoint files
are preserved but remain legacy artifacts; this repair did not retrain them.

`backend/causal.py` now fits feature transforms on training input observations
and target transforms on training labels only. The trainer stores each row's
current observed close, purges training labels that reach the validation origin,
and uses one price anchor for all cumulative forecast-return horizons. It also
removes validation-wide volatility rescaling. The inference reconstruction uses
cumulative returns from the same close instead of compounding them.
Future labels are computed on the original observation time axis. All valid
historical feature rows are retained even when their future labels are missing;
only eligible forecast origins are selected for training/evaluation. Missing
prices/volumes are not forward-filled by percentage-change calculations.
Serving uses target-free features through the latest observation and the same
aligned per-row close channel as training. Forecast dates are explicit weekday
estimates from that observation's date; exchange holidays are not modeled.
Legacy/mixed model/scaler bundles are refused until a new causal-v2 training run
writes a matching protocol/hash marker. Existing assets remain unchanged.

Run from `backend/` with NumPy, pandas, and scikit-learn installed:

```sh
python -m unittest discover -s tests -v
python -m py_compile train.py model.py main.py causal.py artifact_protocol.py data_loader.py
```

Ten tests cover future perturbation invariance, close alignment and horizon
purge, cumulative price anchors, invalid/unsorted input rejection, and label
alignment across feature gaps, actual API input alignment, cumulative prediction
reconstruction with a fake model, origin-based dates, bundle identity, and
train/serve parity when future prices are missing.
The tests do not train a model. Validation still selects early stopping; it is
not an untouched test set. A fresh chronological train/validation/test or
walk-forward experiment and persistence baseline are needed for accuracy claims.
Public market data/provider integrations and old model assets were not exercised.
