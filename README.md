# Realtime Financial Fraud Detection System

An end-to-end, low-latency (<100ms) real-time fraud detection pipeline built on the **PaySim** mobile money financial dataset. 

This repository demonstrates a production-grade division of responsibilities between **Data Science** (offline feature engineering, tabular/DL model training, ONNX serialization) and **Data Engineering** (real-time stream ingestion, stateful window processing, feature store serving, and model inference).