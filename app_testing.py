from hls_images_matcher import HlsImagesMatcher
from hls_image_matcher import HlsImageMatcher

def main():
    temp = 'cat_test2.jpg'
    sources = ['cat_test.jpeg', 'hoodie cat.jpg', 'surprised cat.jpg']
    matcher = HlsImagesMatcher(temp, sources)
    matcher.match_and_display_images()

    # Lightness and Saturation matching only
    matcher.match_and_display_images(hue=False)

    template= 'cat_test2.jpg'
    source = 'cat_test.jpeg'
    matcher2 = HlsImageMatcher(template, source)
    matcher2.graph_histogram()

    # Lightness and Saturation matching only
    matcher2.graph_histogram(hue=False)

if __name__ == '__main__':
    main()