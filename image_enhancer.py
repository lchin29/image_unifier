import colorsys as cs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img

def hls_to_rgb(r, g, b):
    return [ color * 255 for color in list(cs.hls_to_rgb(r,g,b))]

def hist_match_all_hls(source, template):
    """
    Adjust the hue, lightness, and saturation values of an image such 
    that its histogram matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    old_shape = source.shape
    source = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in source])
    template = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in template])
    source = source.ravel()
    template = template.ravel()

    # Get the set of unique pixel hls values and their corresponding indices and counts
    _, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    new_shape = interp_t_values[bin_idx].reshape(old_shape)

    # Transform image back into rgb values
    rgb_image = np.array([[ np.asarray(hls_to_rgb(*hls)).astype(int) for hls in row] for row in new_shape])

    return rgb_image

def hist_match_hls(source, template, hue=False, lightness=False, saturation=False):
    """
    Adjust the hue, lightness, or saturation values of an image such 
    that its histogram matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
    if hue and lightness and saturation:
      return hist_match_all_hls(source, template)

    old_shape = source.shape

    # For template and source, split hue saturation and lightness into their own
    # list by getting the first, second or third of every nested array.
    template = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in template])
    template_h = np.array(template[list(range(0,len(template))),:,0]).ravel()
    template_l = np.array(template[list(range(0,len(template))),:,1]).ravel()
    template_s = np.array(template[list(range(0,len(template))),:,2]).ravel()
    
    source = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in source])
    source_h = np.array(source[list(range(0,len(source))),:,0]).ravel()
    source_l = np.array(source[list(range(0,len(source))),:,1]).ravel()
    source_s = np.array(source[list(range(0,len(source))),:,2]).ravel()

    # Default the new values to the source and update the new values if
    # the parameter to match by the property is passed in.
    new_h = source_h
    new_l = source_l
    new_s = source_s

    if hue:
        # Get the set of unique pixel values and their corresponding indices and counts
        _, bin_idx_h, s_counts_h = np.unique(source_h, return_inverse=True,
                                                return_counts=True)
        t_values_h, t_counts_h = np.unique(template_h, return_counts=True)

        # Take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles_h = np.cumsum(s_counts_h).astype(np.float64)
        s_quantiles_h /= s_quantiles_h[-1]
        t_quantiles_h = np.cumsum(t_counts_h).astype(np.float64)
        t_quantiles_h /= t_quantiles_h[-1]

        # Interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        # ^ needs clarification maybe -Lindsey
        interp_t_values_h = np.interp(s_quantiles_h, t_quantiles_h, t_values_h)

        # Populate new_h values according to new interpolation and previous distribution
        new_h = interp_t_values_h[bin_idx_h]
    if lightness:
        # Same process as for hue
        _, bin_idx_l, s_counts_l = np.unique(source_l, return_inverse=True,
                                              return_counts=True)
        t_values_l, t_counts_l = np.unique(template_l, return_counts=True)

        s_quantiles_l = np.cumsum(s_counts_l).astype(np.float64)
        s_quantiles_l /= s_quantiles_l[-1]
        t_quantiles_l = np.cumsum(t_counts_l).astype(np.float64)
        t_quantiles_l /= t_quantiles_l[-1]

        interp_t_values_l = np.interp(s_quantiles_l, t_quantiles_l, t_values_l)
        new_l = interp_t_values_l[bin_idx_l]
    if saturation:
        # Same process as for hue
        _, bin_idx_s, s_counts_s = np.unique(source_s, return_inverse=True,
                                              return_counts=True)
        t_values_s, t_counts_s = np.unique(template_s, return_counts=True)

        s_quantiles_s = np.cumsum(s_counts_s).astype(np.float64)
        s_quantiles_s /= s_quantiles_s[-1]
        t_quantiles_s = np.cumsum(t_counts_s).astype(np.float64)
        t_quantiles_s /= t_quantiles_s[-1]

        interp_t_values_s = np.interp(s_quantiles_s, t_quantiles_s, t_values_s)
        new_s = interp_t_values_s[bin_idx_s]

    # Create new list of values, by setting every third value too hue, lightness
    # then saturation, then reshaping
    new_values = [None]*(len(new_h)+len(new_s)+len(new_s))
    new_values[::3] = new_h
    new_values[1::3] = new_l
    new_values[2::3] = new_s
    new_shape = np.array(new_values).reshape(old_shape)

    # Convert newly shaped image back into rgb values from hls
    rgb_image = np.array([[ np.asarray(hls_to_rgb(*hls)).astype(int) for hls in row] for row in new_shape])

    return rgb_image

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf


# This isn't done yet
def match_and_display_images(template, sources):
    sources = [np.asarray(source) for source in sources]
    np.asarray(template)
    matches = [hist_match(source, template).astype(int) for source in sources]

    fig = plt.figure()
    gs = plt.GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, :])
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    # make it work for lists
    ax1.imshow(source, cmap=plt.cm.gray)
    ax1.set_title('Source')
    ax2.imshow(template, cmap=plt.cm.gray)
    ax2.set_title('template')
    ax3.imshow(matched, cmap=plt.cm.gray)
    ax3.set_title('Matched')

    # fixup, what to plot? maybe not at all
    x1, y1 = ecdf(source2.ravel()[::3])
    x2, y2 = ecdf(template2.ravel()[::3])
    x3, y3 = ecdf(matched2.ravel()[::3])
    ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
    ax4.plot(x2, y2 * 100, '-k', lw=3, label='Template')
    ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
    ax4.set_xlim(x1[0], x1[-1])
    ax4.set_xlabel('Pixel value')
    ax4.set_ylabel('Cumulative %')
    ax4.legend(loc=5)

