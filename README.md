# IRC_ClusteringPatientData

Clustering patient data for diagnosis using the Support Vector Machine (SVM) learning method. The project is part of an Interdisciplinary Research Computing project for Imperial College London's Horizons course.

## Purpose

A machine learning (ML) model can be successful at processing complex data, such as pixel or random numerical data, and finding patterns in that data that may be otherwise extremely complex to categorise otherwise. There are many examples of ML in the medical field, and we wanted to understand more about what goes into taking medical data and producing interesting results.

## Running the model

Executing main.py:

- pulls the data from the online database and cleans it
- trains the model
- tests the model
- produces accuracy results and relevant graphs

The primary part of the code uses a Model class as defined in model.py. This allows for easy enabling and disabling of functions such as graphs or evaluation mechanisms.

## Attributes

The data attributes are as follows:

Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
Donor:
David W. Aha (aha '@' ics.uci.edu) (714) 856-8779
