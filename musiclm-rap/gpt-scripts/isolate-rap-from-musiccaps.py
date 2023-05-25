"""
This script takes the MusicCaps dataset and creates a subset of it containing
only descriptions related to rap music. It outputs a new .csv file with these entries.
"""
import pandas as pd

# Load the MusicCaps dataset
musiccaps = pd.read_csv('musiccaps.csv')

# Filter out rows containing 'rap' in their description
rap_musiccaps = musiccaps[musiccaps['description'].str.contains('rap')]

# Save the new dataframe to a csv file
rap_musiccaps.to_csv('rap_musiccaps.csv', index=False)

