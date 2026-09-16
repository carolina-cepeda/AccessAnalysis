```bash
cd /home/karo/Documentos/AccessAnalysis
```

Moves the terminal to the project directory, where the script, virtual environment,
and RBA dataset are available.

```bash

.venv/bin/python auth_anomaly_pipeline.py \
  data/rba/rba-dataset.csv \
  --sample-rows 5000 \
  --contamination 0.02
```

Runs the pipeline on the first 5,000 RBA records.

- `.venv/bin/python` uses the project virtual environment, including pandas and scikit-learn.
- `auth_anomaly_pipeline.py` runs loading, cleaning, feature creation, training, and evaluation.
- `data/rba/rba-dataset.csv` is the input authentication-log dataset.
- `--sample-rows 5000` limits the run to 5,000 rows, making an initial test fast and memory-safe.
- `--contamination 0.02` asks Isolation Forest to mark approximately 2% of rows as anomalies.
- The printed report includes up to 10 highest-scoring anomaly examples. Each gives a score and
  rule-based investigation cues, such as a new IP/ASN, a location change, or an elevated history
  of failed logins. These cues are not proof that a login is malicious.

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
