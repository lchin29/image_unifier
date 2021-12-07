import colorsys as cs
import numpy as np
from matplotlib import image as img
from matplotlib import pyplot as plt
from utils import hls_to_rgb, ecdf, match_distribution, split_hls

class HlsImageMatcher:
    def __init__(self, template_file, source_file):
        self.template = img.imread(template_file)
        self.source = img.imread(source_file)

    def hist_match_all_hls(self):
        """
        Match all properties of an image by adjusting pixel values to match the histogram
        distribution of the template image

        Referencing the scikit tool https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/histogram_matching.py
        And stackexchange https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x for hist matching
        
        Output:
        A new rgb image of the source image matched by hls to the template
        """
        old_shape = self.source.shape
        # Convert image to array of hls values
        source = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in self.source])
        template = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in self.template])
        # Flatten the image arrays
        source = source.ravel()
        template = template.ravel()

        new_values = match_distribution(source, template)
        
        # Reshape interpolated pixel locations to fit the source image's original shape
        new_shape = new_values.reshape(old_shape)

        # Transform image back into rgb values
        rgb_image = np.array([[ np.asarray(hls_to_rgb(*hls)).astype(int) for hls in row] for row in new_shape])

        return rgb_image

    def hist_match_hls(self, hue=True, lightness=True, saturation=True):
        """
        Match hue, lightness, saturation of an image by adjusting pixel values to match the histogram
        distribution of the template image

        Referencing the scikit tool https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/histogram_matching.py
        And stackexchange https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x for hist matching

        Input: 
        hue/lightness/saturation - Boolean indicating which property to modify
        
        Output:
        matched (np.ndarray)
        """
        if hue and lightness and saturation:
            return self.hist_match_all_hls()

        old_shape = self.source.shape

        # For template and source, split hue saturation and lightness into their own
        # list by getting the first, second or third of every nested array.
        template_hls = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in self.template])
        template_h, template_l, template_s = split_hls(template_hls)
        
        source_hls = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in self.source])
        source_h, source_l, source_s = split_hls(source_hls)

        # Default the new values to the source and update the new values to the matched 
        # distribution if the parameter to match by the property is passed in.
        new_h = self.match_distribution(source_h, template_h) if hue else source_h
        new_l = self.match_distribution(source_l, template_l) if lightness else source_l
        new_s = self.match_distribution(source_s, template_s) if saturation else source_s

        # Create new list of values, by setting every third value too hue, lightness
        # then saturation, then reshaping to the initial source image shape
        new_values = [None]*(len(new_h)+len(new_s)+len(new_s))
        new_values[::3] = new_h
        new_values[1::3] = new_l
        new_values[2::3] = new_s
        new_shape = np.array(new_values).reshape(old_shape)

        # Convert newly shaped image back into rgb values from hls
        rgb_image = np.array([[ np.asarray(hls_to_rgb(*hls)).astype(int) for hls in row] for row in new_shape])

        return rgb_image

    def graph_histogram(self, hue=True, lightness=True, saturation=True):
        """ 
        Graph comparison of the HLS distributions of a source, template, and matched image

        Referenced from https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

        Input: 
        hue/lightness/saturation - Boolean indicating which property to modify
        
        Output: display plot
        """
        # Match image
        matched = self.hist_match_hls(hue, lightness, saturation).astype(int)

        # For template and source, split hue saturation and lightness into their own
        # list by getting the first, second or third of every nested array.
        template_hls = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in self.template])
        template_h, template_l, template_s = split_hls(template_hls)
        source_hls = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in self.source])
        source_h, source_l, source_s = split_hls(source_hls)
        matched_hls = np.array([[ list(cs.rgb_to_hls(*rgb/255)) for rgb in row] for row in matched])
        matched_h, matched_l, matched_s = split_hls(matched_hls)

        # Get empirical cumulative distribution functions for pixel values
        x1, y1 = ecdf(source_hls.ravel())
        x2, y2 = ecdf(template_hls.ravel())
        x3, y3 = ecdf(matched_hls.ravel())

        x1_h, y1_h = ecdf(source_h)
        x2_h, y2_h = ecdf(template_h)
        x3_h, y3_h = ecdf(matched_h)

        x1_l, y1_l = ecdf(source_l)
        x2_l, y2_l = ecdf(template_l)
        x3_l, y3_l = ecdf(matched_l)

        x1_s, y1_s = ecdf(source_s)
        x2_s, y2_s = ecdf(template_s)
        x3_s, y3_s = ecdf(matched_s)

        # Plot as array
        fig = plt.figure()
        gs = plt.GridSpec(5, 3)
        fig.subplots_adjust(hspace=.8)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
        ax5 = fig.add_subplot(gs[2, :])
        ax6 = fig.add_subplot(gs[3, :])
        ax7 = fig.add_subplot(gs[4, :])
        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()

        ax1.imshow(self.source, cmap=plt.cm.gray)
        ax1.set_title('Source')
        ax2.imshow(self.template, cmap=plt.cm.gray)
        ax2.set_title('Template')
        ax3.imshow(matched, cmap=plt.cm.gray)
        ax3.set_title('Matched')

        ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
        ax4.plot(x2, y2 * 100, '-k', lw=3, label='Template')
        ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
        ax4.set_xlim(x1[0], x1[-1])
        ax4.set_xticklabels([])
        ax4.set_xlabel('Pixel value')
        ax4.legend(loc=5)

        ax5.plot(x1_h, y1_h * 100, '-r', lw=3)
        ax5.plot(x2_h, y2_h * 100, '-k', lw=3)
        ax5.plot(x3_h, y3_h * 100, '--r', lw=3)
        ax5.set_xticklabels([])
        ax5.set_xlim(x1_h[0], x1_h[-1])
        ax5.set_xlabel('Hue value')

        ax6.plot(x1_l, y1_l * 100, '-r', lw=3)
        ax6.plot(x2_l, y2_l * 100, '-k', lw=3)
        ax6.plot(x3_l, y3_l * 100, '--r', lw=3)
        ax6.set_xticklabels([])
        ax6.set_xlim(x1_l[0], x1_l[-1])
        ax6.set_xlabel('Lightness value')
        ax6.set_ylabel('Cumulative %')

        ax7.plot(x1_s, y1_s * 100, '-r', lw=3)
        ax7.plot(x2_s, y2_s * 100, '-k', lw=3)
        ax7.plot(x3_s, y3_s * 100, '--r', lw=3)
        ax7.set_xlim(x1_s[0], x1_s[-1])
        ax7.set_xlabel('Saturation value')

        plt.show()
