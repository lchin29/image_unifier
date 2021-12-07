from image_enhancer import match_and_display_images
from image_enhancer import graph_histogram

def main():
  temp = 'cat_test2.jpg'
  sources = ['cat_test.jpeg', 'hoodie cat.jpg', 'surprised cat.jpg']
  match_and_display_images(temp, sources)
  
  template= 'cat_test2.jpg'
  source = 'cat_test.jpeg'
  graph_histogram(template, source)

if __name__ == '__main__':
  main()