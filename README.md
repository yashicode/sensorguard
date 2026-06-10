# SensorGuard
![SensorGuard dashboard](docs/screenshot_1.png)
![SensorGuard dashboard](docs/screenshot_2.png)

# SensorGuard — Behavioural Anomaly Detection for Industrial Sensors

Unsupervised anomaly detection trained only on normal pump behaviour.
Flags sensor readings that deviate from healthy baselines across 52 channels,
deployed as a Streamlit dashboard.

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-IsolationForest-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-green)



## Why this matters beyond pumps

The core approach here — train only on normal data, flag anything that deviates —
is the same paradigm used in network intrusion detection, endpoint behavioural
analysis, and LLM-output monitoring. The domain is pumps; the method is general.
If you have a system where failures are rare and unlabelled but normal operation
is abundant, one-class detection is usually the right call.

---

## The problem

Pump failures are expensive less because of repair cost and more because of
unplanned downtime. SensorGuard watches 51 sensor channels and raises a flag when
readings stop looking normal. The model never sees a labelled failure during
training — only healthy operation data — which reflects how most production
environments actually look: years of normal readings, almost no documented
failures.

---

## Model selection

I benchmarked eight models on the same data split.

| Model                  | Type         |    P |    R |   F1 |  FPR |
|------------------------|--------------|-----:|-----:|-----:|-----:|
| Random Forest          | supervised   | 1.00 | 1.00 | 1.00 | 0.00 |
| HistGradientBoosting   | supervised   | 1.00 | 1.00 | 1.00 | 0.00 |
| KNN                    | supervised   | 1.00 | 1.00 | 1.00 | 0.00 |
| Decision Tree          | supervised   | 0.97 | 1.00 | 0.98 |  —   |
| Logistic Regression    | supervised   | 0.95 | 1.00 | 0.97 |  —   |
| Isolation Forest       | unsupervised | 0.52 | 0.99 | 0.68 | 0.065|
| Elliptic Envelope      | unsupervised |  —   | 1.00 | 0.67 |  —   |
| Local Outlier Factor   | unsupervised |  —   | 0.84 | 0.58 |  —   |

The supervised models score perfectly. That is a problem, not a result.
A model that scores 1.0 on failure detection usually means it is recognising
a machine already deep in failure — obvious from 52 sensor channels simultaneously
going wrong — rather than catching early deviation. It reads the present state,
not the trajectory.

Isolation Forest ships because it is the only model that never looks at a
failure label. It catches ~99% of failures (0.99 recall) at a false-positive
rate of [X.XX]%, scoring all 220,000 readings in under 2 seconds. The precision
of ~0.52 is visible and tunable via the dashboard threshold slider.

---

## Dataset

Kaggle pump sensor data: ~220,000 minute-by-minute readings across 52 sensors,
labelled NORMAL, RECOVERING, or BROKEN.

The CSV is large and not committed to this repo. Download from Kaggle and drop
`sensor.csv` into the `data/` folder. Labels are used only to evaluate the model,
never to train it.

---

## Project structure
sensorguard/
├── data/
│   └── sensor.csv           
├── model/
│   └── sensorguard.joblib   
├── src/
│   ├── train.ipynb          
│   └── app.py               
├── requirements.txt
└── README.md

---

## Reproduce it

```bash
git clone https://github.com/yashicode/sensorguard.git
cd sensorguard

conda create -n sensorguard python=3.11
conda activate sensorguard
pip install -r requirements.txt

# download sensor.csv from Kaggle → place in data/

# run all cells in src/train.ipynb to produce model/sensorguard.joblib

python -m streamlit run src/app.py
```

Opens at http://localhost:####.

---

## Dashboard

The main tab shows a live anomaly score gauge with the alarm threshold marked,
the raw score, the verdict, and a per-sensor breakdown of which channels are
pulling the score up. The second tab accepts a CSV upload and returns batch
scores. The sensitivity slider is worth spending time on: it lets you trade
recall against false-alarm rate in real time and pick the operating point that
matches your tolerance.

---

## Limitations

Precision at the default threshold is ~0.52, meaning roughly half of alerts are
false positives. In a production setting you would tune the threshold to whatever
false-alarm rate operators can absorb — lower sensitivity, fewer alerts, some
failures caught later.

The model reacts to abnormal readings rather than predicting failure in advance.
Real early-warning would need a time-window target label and more than the seven
labelled failure events this dataset contains.

Each prediction treats a single timestep independently. Rolling-window features
— a sensor's recent mean, rate of change, or inter-sensor correlation — are the
obvious next step.

## LICENSE
MIT. See [LICENSE](LICENSE).
