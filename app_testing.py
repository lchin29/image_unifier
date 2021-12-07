from hls_images_matcher import HlsImagesMatcher
from hls_image_matcher import HlsImageMatcher

def main():
  temp = 'cat_test2.jpg'
  sources = ['cat_test.jpeg', 'hoodie cat.jpg', 'surprised cat.jpg']
  HlsImagesMatcher(temp, sources).match_and_display_images()
  
  template= 'cat_test2.jpg'
  source = 'cat_test.jpeg'
  HlsImageMatcher(template, source).graph_histogram()

if __name__ == '__main__':
  main()