from image_enhancer import match_and_display_images

def main():
  temp = 'cat_test2.jpg'
  sources = ['cat_test.jpeg', 'hoodie cat.jpg', 'surprised cat.jpg']
  match_and_display_images(temp, sources)

if __name__ == '__main__':
  main()