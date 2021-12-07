import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
from hls_image_matcher import HlsImageMatcher

class HlsImagesMatcher:
    def __init__(self, template_file, source_files):
        self.template_file = template_file
        self.source_files = source_files
        self.template = img.imread(template_file)
        self.sources = [img.imread(source) for source in source_files]

    def match_and_display_images(self, hue=True, lightness=True, saturation=True):
        # Read in image
        sources = [np.asarray(source) for source in self.sources]
        np.asarray(self.template)

        # Match image
        matches = [
          HlsImageMatcher(self.template_file, sf).
          hist_match_hls(hue, lightness, saturation).
          astype(int) for sf in self.source_files]

        # Plot images
        fig = plt.figure()
        gs = plt.GridSpec(3, len(matches))

        ax1 = fig.add_subplot(gs[0, :])
        ax1.imshow(self.template, cmap=plt.cm.gray)
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