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
* `multi-vehicle.py`: Likely an enhanced or multi-camera script for the vehicle system.

Each module is designed to be self-contained with its own Dockerfile (`*.Dockerfile`), environment variables (`*.env`), and Python requirements (`*.requirements.txt`), making it easy to build and deploy.

### Getting Started

This project is containerized using Docker for simplified setup and deployment.

1.  **Build the Docker Images:**
    The `build.sh` script can be used to build the necessary Docker images for each service.
    ```sh
    ./build.sh
    ```

2.  **Configure Environment Variables:**
    Update the `.env` files for each module (`vehicle.env`, `people.env`, `crowd.env`) with the necessary configurations, such as camera stream URLs, API keys, or database credentials.

3.  **Run the Services:**
    Once built, you can run each service as a Docker container. For example, to run the vehicle detection service:
    ```sh
    docker run --rm --env-file vehicle.env vehicle-service:latest
    ```
    *(Note: You might use Docker Compose in a full production setup to manage these services together.)*

## Future Improvements

While the system is functional, there's always room for growth. The most challenging part of this project was designing the core logic, and I'm keen to make it even better. My future goals for this project include:

* **Optimizing the Logic:** Refactoring the core detection and tracking logic for greater efficiency and accuracy.
* **Improving Scalability:** Implementing a more robust multi-camera handling system and potentially using a message queue (like RabbitMQ or Kafka) for inter-service communication.
* **Advanced Analytics:** Expanding the density analysis to provide predictive insights, such as forecasting peak crowd times.

Thank you for checking out this project!
