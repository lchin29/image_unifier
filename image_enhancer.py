import colorsys as cs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img

def hls_to_rgb(r, g, b):
    # Convert HLS (0-1 floating point) to RGB (0-1 floating point multiplied by 255)
    return [ color * 255 for color in list(cs.hls_to_rgb(r,g,b))]

def hist_match_all_hls(image, template):
    """
    Match all properties of an image by adjusting pixel values to match the histogram
    distribution of the template image

    Referencing the scikit tool https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/histogram_matching.py
    And stackexchange https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x for hist matching

    Input: 
    source (np.ndarray) - image to modify
    template (np.ndarray) - image to match the histogram of
    
    Output:
    matched (np.ndarray)
    """

    old_shape = source.shape
    source = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in source]) # Convert image to array of hls values
    template = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in template])
    source = source.ravel() # Flatten the image array
    template = template.ravel()

    # Get the distribution of the counts and indices of unique hls values 
    _, s_unique_indices, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Create a cumulative distribution of the image counts, normalized by the size of each image
    s_quantiles = np.cumsum(s_counts).astype(np.float64) # Create a distribution of the cumulative sum of the counts
    s_quantiles /= s_quantiles[-1] # Normalize by the size of the array
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Use linear interpolation to match the pixel values in the template image
    # that have similar normalized cumulative distribution as in the image to modify
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    # Reshape interpolated pixel locations to fit the image's original shape
    new_shape = interp_t_values[s_unique_indices].reshape(old_shape) 

    # Transform image back into rgb values
    rgb_image = np.array([[ np.asarray(hls_to_rgb(*hls)).astype(int) for hls in row] for row in new_shape])

    return rgb_image

def hist_match_hls(source, template, hue=True, lightness=True, saturation=True):
    """
    Match hue, lightness, saturation of an image by adjusting pixel values to match the histogram
    distribution of the template image

    Referencing the scikit tool https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/histogram_matching.py
    And stackexchange https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x for hist matching

    Input: 
    image (np.ndarray) - image to modify
    template (np.ndarray) - image to match the histogram of
    hue/lightness/saturation - Boolean indicating which property to modify
    
    Output:
    matched (np.ndarray)
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
        # Get the distribution of the counts and indices of unique hls values 
        _, s_unique_indices_h, s_counts_h = np.unique(source_h, return_inverse=True,
                                                return_counts=True)
        t_values_h, t_counts_h = np.unique(template_h, return_counts=True)

        # Create a cumulative distribution of the image counts, normalized by the size of each image
        s_quantiles_h = np.cumsum(s_counts_h).astype(np.float64) # Create a distribution of the cumulative sum of the counts
        s_quantiles_h /= s_quantiles_h[-1] # Normalize by the size of the array
        t_quantiles_h = np.cumsum(t_counts_h).astype(np.float64)
        t_quantiles_h /= t_quantiles_h[-1]

        # Use linear interpolation to match the pixel values in the template image
        # that have similar normalized cumulative distribution as in the image to modify
        interp_t_values_h = np.interp(s_quantiles_h, t_quantiles_h, t_values_h)

        # Populate new_h values according to new interpolation and previous distribution
        new_h = interp_t_values_h[s_unique_indices_h]
    if lightness:
        # Same process as for hue
        _, s_unique_indices_l, s_counts_l = np.unique(source_l, return_inverse=True,
                                              return_counts=True)
        t_values_l, t_counts_l = np.unique(template_l, return_counts=True)

        s_quantiles_l = np.cumsum(s_counts_l).astype(np.float64)
        s_quantiles_l /= s_quantiles_l[-1]
        t_quantiles_l = np.cumsum(t_counts_l).astype(np.float64)
        t_quantiles_l /= t_quantiles_l[-1]

        interp_t_values_l = np.interp(s_quantiles_l, t_quantiles_l, t_values_l)
        new_l = interp_t_values_l[s_unique_indices_l]
    if saturation:
        # Same process as for hue
        _, s_unique_indices_s, s_counts_s = np.unique(source_s, return_inverse=True,
                                              return_counts=True)
        t_values_s, t_counts_s = np.unique(template_s, return_counts=True)

        s_quantiles_s = np.cumsum(s_counts_s).astype(np.float64)
        s_quantiles_s /= s_quantiles_s[-1]
        t_quantiles_s = np.cumsum(t_counts_s).astype(np.float64)
        t_quantiles_s /= t_quantiles_s[-1]

        interp_t_values_s = np.interp(s_quantiles_s, t_quantiles_s, t_values_s)
        new_s = interp_t_values_s[s_unique_indices_s]

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
    """
    Calculate empirical CDF

    Referenced from https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

def graph_histogram(source, template, matched):
    """ 
    Graph comparison of the HLS distributions of a source, template, and matched image

    Referenced from https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

    Input: image arrays of RGB values
    
    Output: display plot

    """
    # Might want to break this up into functions so we don't get dinged on modularity

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

    matched = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in matched])
    matched_h = np.array(source[list(range(0,len(matched))),:,0]).ravel()
    matched_l = np.array(source[list(range(0,len(matched))),:,1]).ravel()
    matched_s = np.array(source[list(range(0,len(matched))),:,2]).ravel()

    # Get empirical cumulative distribution functions for pixel values
    x1, y1 = ecdf(source.ravel())
    x2, y2 = ecdf(template.ravel())
    x3, y3 = ecdf(matched.ravel())

    x1_h, y1_h = ecdf(source_h.ravel())
    x2_h, y2_h = ecdf(template_h.ravel())
    x3_h, y3_h = ecdf(matched_h.ravel())

    x1_l, y1_l = ecdf(source_l.ravel())
    x2_l, y2_l = ecdf(template_l.ravel())
    x3_l, y3_l = ecdf(matched_k.ravel())

    x1_s, y1_s = ecdf(source_s.ravel())
    x2_S, y2_s = ecdf(template_s.ravel())
    x3_s, y3_s = ecdf(matched_s.ravel())

    # Plot as array
    fig = plt.figure()
    gs = plt.GridSpec(5, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, :])
    ax5 = fig.add_subplot(gs[2, :])
    ax6 = fig.add_subplot(gs[3, :])
    ax7 = fig.add_subplot(gs[4, :])
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source, cmap=plt.cm.gray)
    ax1.set_title('Source')
    ax2.imshow(template, cmap=plt.cm.gray)
    ax2.set_title('template')
    ax3.imshow(matched, cmap=plt.cm.gray)
    ax3.set_title('Matched')

    ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
    ax4.plot(x2, y2 * 100, '-k', lw=3, label='Template')
    ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
    ax4.set_xlim(x1[0], x1[-1])
    ax4.set_xlabel('Pixel value')
    ax4.set_ylabel('Cumulative %')
    ax4.legend(loc=5)

    ax5.plot(x1_h, y1_h * 100, '-r', lw=3, label='Source')
    ax5.plot(x2_h, y2_h * 100, '-k', lw=3, label='Template')
    ax5.plot(x3_h, y3_h * 100, '--r', lw=3, label='Matched')
    ax5.set_xlim(x1_h[0], x1_h[-1])
    ax5.set_xlabel('Hue value')
    ax5.set_ylabel('Cumulative %')
    ax5.legend(loc=5)

    ax6.plot(x1_l, y1_l * 100, '-r', lw=3, label='Source')
    ax6.plot(x2_l, y2_l * 100, '-k', lw=3, label='Template')
    ax6.plot(x3_l, y3_l * 100, '--r', lw=3, label='Matched')
    ax6.set_xlim(x1_l[0], x1_l[-1])
    ax6.set_xlabel('Lightness value')
    ax6.set_ylabel('Cumulative %')
    ax6.legend(loc=5)

    ax7.plot(x1_s, y1_s * 100, '-r', lw=3, label='Source')
    ax7.plot(x2_s, y2_s * 100, '-k', lw=3, label='Template')
    ax7.plot(x3_s, y3_s * 100, '--r', lw=3, label='Matched')
    ax7.set_xlim(x1_s[0], x1_s[-1])
    ax7.set_xlabel('Saturation value')
    ax7.set_ylabel('Cumulative %')
    ax7.legend(loc=5)

    plt.show()

def match_and_display_images(template_file, source_files, hue=True, lightness=True, saturation=True):
    # Add comments to this?
    template = img.imread(template_file)
    sources = [img.imread(sf) for sf in source_files]
    sources = [np.asarray(source) for source in sources]
    np.asarray(template)

    matches = [hist_match_hls(source, template, hue, lightness, saturation).astype(int) for source in sources]

    fig = plt.figure()
    gs = plt.GridSpec(3, len(matches))

    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(template, cmap=plt.cm.gray)
    ax1.set_title('Template')
    ax1.set_axis_off()

    ax_source = fig.add_subplot(gs[1, 0])
    ax_source.imshow(sources[0], cmap=plt.cm.gray)
    ax_source.set_title('Source 1')
    ax_source.set_axis_off()
    ax_match = fig.add_subplot(gs[2,0])
    ax_match.imshow(matches[0], cmap=plt.cm.gray)
    ax_match.set_title('Matched 1')
    ax_match.set_axis_off()
    
    for i in range(len(matches)):
        if i == 0: continue
        axs = fig.add_subplot(gs[1, i], sharex=ax_source, sharey=ax_source)
        axs.imshow(sources[i], cmap=plt.cm.gray)
        axs.set_title(f'Source {i+1}')
        axs.set_axis_off()
        axm = fig.add_subplot(gs[2, i], sharex=ax_match, sharey=ax_match)
        axm.imshow(matches[i], cmap=plt.cm.gray)
        axm.set_title(f'Matched {i+1}')
        axm.set_axis_off()

    plt.show()


