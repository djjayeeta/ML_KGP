import image_helper as ih

image_path = "data/test_new_image.jpeg"
row = 100
col = 100
ih.create_test_image(image_path,row,col)
ih.save_image_as_pickle(image_path)
