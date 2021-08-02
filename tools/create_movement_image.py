from PIL import Image
import os

def createMovementImage(basePath):
    images = list()
    for (dirpath, dirnames, filenames) in os.walk(basePath):
        episode = 0
        while True:
            episodeImages = [filename for filename in filenames if "episode{}".format(episode) in filename]
            if len(episodeImages) == 0:
                return

            episodeImages.sort()
            for image in episodeImages:
                imagePath = basePath + image
                images.append(Image.open(imagePath))
            
            for i in range(len(images)):
                images[i].putalpha(255 - 20* (i+1))
            
            resultImage = images[0]
            for i in range(len(images)):
                if i == 0:
                    continue
                resultImage.alpha_composite(images[i], dest=(i*20, 0))
            
            path = basePath + "resultImage_epsiode{}.png".format(episode)
            resultImage.save(path, "PNG")
            episode = episode + 1
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--basePath', "-bP", required=True)
    args = parser.parse_args()

    createMovementImage(args.basePath)
