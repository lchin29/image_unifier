import colorsys as cs
import numpy as np

def hls_to_rgb(r, g, b):
    """
    Convert HLS (0-1 floating point) to RGB (0-1 floating point multiplied by 255)
    """
    return [ color * 255 for color in list(cs.hls_to_rgb(r,g,b))]

def ecdf(x):
    """
    Calculate empirical CDF

    Referenced from https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    """
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

def match_distribution(source, template):
    """
    Match the distribution of the values of the source list to template list.

    Input:
    source (nparray): list of values to match to template's distribution
    template (nparray): list of values

    Output:
    List of source values matching distribution of template
    """
    # Get the distribution of the counts and indices of unique hls values 
    _, s_unique_indices, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Create a distribution of the cumulative sum of the counts
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    # Normalize by the size of the array
    s_quantiles /= s_quantiles[-1] 
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Use linear interpolation to match the pixel values in the template image
    # that have similar normalized cumulative distribution as in source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[s_unique_indices]

def split_hls(hls_image):
    """
    Return a tuple of an image split by its first, second, and third value to get
    the hues, lightnesses, and saturations.

    Input:
    hls_image (ndarray): image

    Output:
    Tuple with list of hues, list of lightnesses and list of saturations
    """
    h = np.array(hls_image[list(range(0,len(hls_image))),:,0]).ravel()
    l = np.array(hls_image[list(range(0,len(hls_image))),:,1]).ravel()
    s = np.array(hls_image[list(range(0,len(hls_image))),:,2]).ravel()

    return h,l,s