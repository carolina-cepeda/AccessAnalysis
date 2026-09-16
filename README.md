

```bash

.venv/bin/python auth_anomaly_pipeline.py \
  data/rba/rba-dataset.csv \
  --sample-rows 5000 \
  --contamination 0.02
```

Runs the pipeline on the first 5,000 RBA records.

- `auth_anomaly_pipeline.py` runs loading, cleaning, feature creation, training, and evaluation.
- `data/rba/rba-dataset.csv` is the input authentication-log dataset.
- `--sample-rows 5000` limits the run to 5,000 rows, making an initial test fast and memory-safe.
- `--contamination 0.02` asks Isolation Forest to mark approximately 2% of rows as anomalies.

```bash

.venv/bin/python auth_anomaly_pipeline.py \
  data/rba/rba-dataset.csv \
  --sample-rows 50000 \
  --contamination 0.02 \
  --schema-map '{"attack_label":"Is Attack IP"}'
```

Runs a larger 50,000-row evaluation and explicitly selects `Is Attack IP` as the
reference label.

- `--sample-rows 50000` uses more records, so the anomaly pattern and metrics are more representative.
- `--schema-map '{"attack_label":"Is Attack IP"}'` overrides automatic label selection. The label is excluded from training and used only to calculate precision, recall, F1, and false-positive rate.
