from PIL import Image
import os

# Method to create a Movement Image by overlaying several images from created by test policy
# General Idea: By using an image with black floor and white sky, the horizon can be determinated by finding the position of the first black pixel.
#   Depending on the horizon, the images can be overlayed stabilizing on the horizon line to get the movement to the image.
#   As a background the image base_movement_image.png is used, so original background can be applied.
# Input: basePath => Path of the images
#        imageRangeStart => Index of the first Image to be processed
#        imageRangeEnd => Index of the last Image to be processed, this image will be in the middle of the movementimage 
def createMovementImage(basePath, imageRangeStart = 0, imageRangeEnd = -1):
    for (dirpath, dirnames, filenames) in os.walk(basePath):
        episode = 0
        while True:
            images = list()

            episodeImages = [filename for filename in filenames if "episode{}".format(episode) in filename]
            if len(episodeImages) == 0:
                return

            episodeImages.sort(key=lambda f: int(f[17:].split('.')[0]))

            episodeImages = episodeImages[int(imageRangeStart):int(imageRangeEnd) + 1]

            for image in episodeImages:
                imagePath = basePath + image
                img = Image.open(imagePath)
                img = img.convert("RGBA")
                images.append(img)
            
            firstBlackPixel = list()
            for i in range(len(images)):
                pixdata = images[i].load()

                width, height = img.size
                for y in range(height):
                    for x in range(width):
                        if pixdata[x, y] == (255, 255, 255, 255):
                            pixdata[x, y] = (255, 255, 255, 0)
                        elif pixdata[x, y] == (0, 0, 0, 255):
                            if len(firstBlackPixel) != i+1:
                                firstBlackPixel.append(y)
                            if pixdata[x, y-2] == (255, 255, 255, 0):
                                pixdata[x, y-1] = (0, 0, 0, 0)
                            pixdata[x, y] = (0, 0, 0, 0)
                        else:
                            transparency = 255 + 10 * (i - len(images))
                            (r,g,b,t) = pixdata[x,y]
                            pixdata[x,y] = (r,g,b, transparency)
                            
            resultImage = Image.open("tools/base_movement_image.png")
            resultImage = resultImage.convert("RGBA")

            for i in range(len(images)):
                resultImage.alpha_composite(images[i], dest=((i - len(images) + 3)*40, firstBlackPixel[0] - firstBlackPixel[i]))
            
            path = basePath + "resultImage_epsiode{}.png".format(episode)
            resultImage.save(path, "PNG")
            episode = episode + 1
        

# Method to create a Movement Image by overlaying several images from created by test policy
# Input: basePath => Path of the images
#        imageFrequency => The frequency the images were taken from the simulation
#        imageRangeStart => Timestep of the first Image to be processed
#        imageRangeEnd => Timestep of the last Image to be processed, this image will be in the middle of the movementimage 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--basePath', "-bP", required=True)
    parser.add_argument('--imageFrequency', "-iF", type=int, default=1)
    parser.add_argument('--imageRangeStart', "-irs", type=int)
    parser.add_argument('--imageRangeEnd', "-ire", type=int, default=-1)

    args = parser.parse_args()

    createMovementImage(args.basePath, args.imageRangeStart / args.imageFrequency, args.imageRangeEnd / args.imageFrequency)
