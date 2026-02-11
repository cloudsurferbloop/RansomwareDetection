# RansomwareDetection
This project simulates an early-warning system that detects ransomware indicators and visualizes system threat levels through an interactive web interface.
This project demonstrates how early anomaly detection and entropy analysis can be used to identify potential ransomware activity before large-scale encryption occurs.

It is designed as a hackathon prototype to showcase real-time cybersecurity monitoring and visualization.

# Features
Three-Level Threat Classification

GREEN – Safe

ORANGE – Suspicious (1 indicator triggered)

RED – High Risk (2+ indicators triggered)

# Real-Time Dashboard Updates (WebSockets)

Dynamic Threat Distribution Visualization (Chart.js)

Entropy-Based Encryption Detection

Multi-Factor Detection Engine:

Suspicious file extensions

Ransom-related keyword scanning

High Shannon entropy values

File Upload & Instant Analysis

Live Activity Log with timestamps

# Architecture
Backend

Python

Flask

Flask-SocketIO

Frontend

HTML

CSS

JavaScript

Chart.js

Communication

WebSocket-based real-time updates

# The system follows a modular architecture separating:

Detection engine

API layer

Real-time event streaming

Visualization dashboard
