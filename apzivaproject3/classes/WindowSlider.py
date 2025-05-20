# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 16:44:49 2025

@author: Paul
"""
# %% Packages

import random
import pandas as pd

# %% WindowSlider Class


class WindowSlider():
    def __init__(self, df, window_size=10, slide_size=5):
        self.window_size = window_size
        self.slide_size = slide_size
        self.df = df.copy()
        self.starting_indexes = df.index.tolist()
        self.current_window = self.starting_indexes[:self.window_size]
        self.end = False

    def slide(self, slide_size=None, direction='down'):
        if direction == 'down':
            # Move the window by `slide_size` with overlap
            if slide_size is None:
                slide_size = self.slide_size

            new_start = self.current_window[0] + slide_size

            if new_start + self.window_size > len(self.df):
                print('slide, Reached End of DataFrame')
                self.end = True
                return None

            # Update the window
            self.current_window = (
                self.df.index[new_start:new_start + self.window_size].tolist()
                )

            return self.current_window

        elif direction == 'up':
            # Move the window upwards by `slide_size` with overlap
            if slide_size is None:
                slide_size = self.slide_size

            # Calculate the new start index for upwards sliding
            new_start = self.current_window[0] - slide_size

            # Check if the new start is before the beginning of the DataFrame
            if new_start < 0:
                print('slide, Reached Beginning of DataFrame')
                return None

            # Update the window by slicing from the new start index
            self.current_window = (
                self.df.index[new_start:new_start + self.window_size].tolist()
            )

            return self.current_window

        else:
            raise ValueError("direction must be 'down' or 'up'")

    def swap(self, new_idx):
        # Get the old indices (from the current window)
        old_idx = self.current_window

        # Iterate over pairs of old and new indices
        for old, new in zip(old_idx, new_idx):
            # Swap the rows
            temp = self.df.loc[old].copy()
            self.df.loc[old] = self.df.loc[new]
            self.df.loc[new] = temp

        return [old_idx, new_idx]  # Return the swapped indexes

    def reset_starting(self):
        self.current_window = self.starting_indexes[:self.window_size].tolist()

    def get_current_window(self):
        # Returns the current window indexes
        return self.current_window

    def get_df(self):
        return self.df


# %% Test Data

# # Example word list
# words = [
#     'apple', 'banana', 'cherry', 'dragon', 'emerald', 'falcon',
#     'galaxy', 'horizon', 'island', 'jungle', 'krypton', 'lunar',
#     'meteor', 'nebula', 'oasis', 'phantom', 'quartz', 'raven',
#     'sapphire', 'tiger', 'utopia', 'voyager', 'whisper', 'xenon',
#     'zephyr'
# ]

# # Randomly select 25 words
# random_words = random.choices(words, k=30)

# # Create DataFrame
# df = pd.DataFrame({'Random_Word': random_words})

# # Create WindowSlider Object
# ws = WindowSlider(df, 10, 5)
# ws.get_current_window()

# slides = list(range(0, 4, 1))
# way = 'down'

# for s in slides:
#     # Return Current Window
#     current = df.iloc[ws.get_current_window(), :].copy()

#     # Do Something to Resort the Array
#     current['word_length'] = [len(word) for word in current['Random_Word']]
#     current = current.sort_values(by='word_length', ascending=False)
#     new_idx = current.index.tolist()

#     # Swap Values in ws
#     ws.swap(new_idx)

#     # Slide
#     if ws.end is True:
#         way = 'up'
#         ws.slide(direction=way)
#         ws.end = False

#     ws.slide(direction=way)
