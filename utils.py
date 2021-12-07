import colorsys as cs
import numpy as np

def hls_to_rgb(r, g, b):
    # Convert HLS (0-1 floating point) to RGB (0-1 floating point multiplied by 255)
    return [ color * 255 for color in list(cs.hls_to_rgb(r,g,b))]

def ecdf(x):
    """
    Calculate empirical CDF

    Referenced from https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf