import os
import csv
import numpy as np

def getPrintableArrayFromNumpyArray(array):
    arrayStr = array.astype(str)
    array[array=='nan'] = 'NaN'
    printArray = np.zeros(array.size, dtype=[('key_name', int), ("value", 'U6')])
    printArray['key_name'] = np.arange(1, array.size+1)
    printArray["value"] = array

    return printArray

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def generateMixedCsv(baseFolder):
    directories = [root for root, dirs, files in os.walk(baseFolder)]
    if baseFolder + "/toolOutput" in directories:
        directories.remove(baseFolder + "/toolOutput")

    files = list()

    for directory in directories:
        if os.path.isfile(directory + "/progress.txt"):
            with open(directory + "/progress.txt") as tsv:
                file = dict()
                for column in zip(*[line for line in csv.reader(tsv, dialect="excel-tab")]):
                    if column[0] == "Epoch":
                        continue
                    file[column[0]] = column[1:]
                files.append(file)

    os.makedirs(baseFolder + "/toolOutput", exist_ok=True)

    moving_average_steps = [1, 2, 4, 8, 16]

    for key in files[0]:
        keyValues = np.zeros(shape=(len(files), len(files[0][key])))
        for index in range(len(files)):
            if key in files[index]:
                keyValues[index] = files[index][key]
        
        if "TestEpRet" in key:
            keyValues = np.divide(keyValues, 10)


        for step in moving_average_steps:

            pathToKeyOutput = baseFolder + "/toolOutput/" + "movingAverage" + str(step) + "/" + key
            
            os.makedirs(pathToKeyOutput, exist_ok=True)

            maxValues = moving_average(np.amax(keyValues, axis=0), step)
            minValues = moving_average(np.amin(keyValues, axis=0), step)
            mean = moving_average(np.mean(keyValues, axis=0), step)
            median = moving_average(np.median(keyValues, axis=0), step)
            std = moving_average(np.std(keyValues, axis=0), step)
            std_pos = mean + 2 * std
            std_neg = mean - 2 * std
            

            np.savetxt(pathToKeyOutput + "/max.csv", getPrintableArrayFromNumpyArray(maxValues),fmt="%d ,%s",header="Epoch, " + key)
            np.savetxt(pathToKeyOutput + "/min.csv", getPrintableArrayFromNumpyArray(minValues),fmt="%d ,%s",header="Epoch, " + key)
            np.savetxt(pathToKeyOutput + "/mean.csv", getPrintableArrayFromNumpyArray(mean),fmt="%d ,%s",header="Epoch, " + key)
            np.savetxt(pathToKeyOutput + "/median.csv", getPrintableArrayFromNumpyArray(median),fmt="%d ,%s",header="Epoch, " + key)
            np.savetxt(pathToKeyOutput + "/std.csv", getPrintableArrayFromNumpyArray(std),fmt="%d ,%s",header="Epoch, " + key)
            np.savetxt(pathToKeyOutput + "/std_pos.csv", getPrintableArrayFromNumpyArray(std_pos),fmt="%d ,%s",header="Epoch, " + key)
            np.savetxt(pathToKeyOutput + "/std_neg.csv", getPrintableArrayFromNumpyArray(std_neg),fmt="%d ,%s",header="Epoch, " + key)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', required=True)    
    args = parser.parse_args()

    generateMixedCsv(args.base_folder)
