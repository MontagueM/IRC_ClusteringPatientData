import pandas as pd

"""
This version of data_handler will just be using the kaggle, already-cleaned dataset so we can start work without
needing to wait for the data to be fully cleaned from the original UCI dataset.

Dataset heart.csv will only be used temporarily, attributes:
    Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
    University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
    University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
    V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
    Donor:
    David W. Aha (aha '@' ics.uci.edu) (714) 856-8779
"""

df = pd.read_csv('heart.csv')
