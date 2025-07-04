# Aegis Intelligence Platform: AI-Powered Security & Analysis

Welcome to the Aegis Intelligence Platform, an integrated system designed to enhance security and operational awareness through real-time computer vision. This project was developed to provide a centralized solution for monitoring, detecting, and analyzing activity across a physical space.

For me, this project was a fantastic and challenging introduction to building production-scale AI systems. It was a valuable experience in designing the logic that makes these different modules work together seamlessly.

## About The Project

In any secure environment, manually monitoring all activity is a significant challenge. This platform automates key surveillance tasks by leveraging AI to analyze CCTV footage. It identifies potential security threats, provides insights into spatial usage, and creates a smarter, more responsive security operation.

Our team developed a suite of four core products, three of which are represented in this repository:

* **Vehicle Recognition:** Detects and identifies vehicles, flagging any that are on a restricted list.
* **Person Recognition:** Identifies individuals in real-time and sends alerts if a banned person is detected.
* **Density Analysis:** Generates heatmaps to visualize the flow and concentration of people and objects.
* **(External Module) Speech-to-Text:** Transcribes walkie-talkie audio, summarizes it, and integrates with email for reporting.

All these systems were designed to feed into a centralized dashboard for easy monitoring by a security team.

## Key Features

* **Real-Time Detection:** Utilizes YOLOv8 for high-performance object detection of vehicles and people.
* **Modular Design:** Each core function (vehicle, person, crowd) is built as a separate, containerized service for scalability and maintainability.
* **Automated Alerts:** Designed to trigger notifications when specific, predefined security events occur.
* **Data-Driven Insights:** The crowd density module provides valuable analytics on how a space is being used.

## Technical Snapshot

This repository contains the core logic for the computer vision modules. Looking at the files, the project is structured as follows:

* `vehicle.py`: The main script for vehicle detection and recognition.
* `people.py`: The script responsible for person detection and identification.
* `crowd.py`: Handles the logic for crowd density analysis and heatmap generation.

Each module is designed to be self-contained with its own Dockerfile (`*.Dockerfile`), environment variables (`*.env`), and Python requirements (`*.requirements`), making it easy to build and deploy.

Thank you for checking out this project!
