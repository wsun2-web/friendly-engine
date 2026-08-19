import imageio.v3 as iio
filenames = ['potato1.png', 'potato2.png']
images = []

for filename in filenames:
    images.append(iio.imread(filename))

iio.imwrite('potatodance.gif', images, duration = 500, loop = 0)