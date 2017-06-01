import image_helper as ih

image_path = "data/test_5050_image.jpeg"
row = 50
col = 50
ih.create_test_image(image_path,row,col)
ih.save_image_as_pickle(image_path)
