import os
import csv
import numpy as np

def generateMixedCsv(baseFolder):
    directories = [f.path for f in os.scandir(baseFolder) if f.is_dir()]
    if baseFolder + "/toolOutput" in directories:
        directories.remove(baseFolder + "/toolOutput")

    files = list()

    for directory in directories:
        with open(directory + "/progress.txt") as tsv:
            file = dict()
            for column in zip(*[line for line in csv.reader(tsv, dialect="excel-tab")]):
                if column[0] == "Epoch":
                    continue
                file[column[0]] = column[1:]
            files.append(file)

    os.makedirs(baseFolder + "/toolOutput", exist_ok=True)

    for key in files[0]:
        keyValues = np.zeros(shape=(len(files), len(file[key])))
        for index in range(len(files)):
            keyValues[index] = files[index][key]

        os.makedirs(baseFolder + "/toolOutput/" + key, exist_ok=True)

        maxValues = np.amax(keyValues, axis=0)
        minValues = np.amin(keyValues, axis=0)
        mean = np.mean(keyValues, axis=0)
        median = np.median(keyValues, axis=0)

        pathToKeyOutput = baseFolder + "/toolOutput/" + key

        def getPrintableArrayFromNumpyArray(array):
            arrayStr = array.astype(str)
            array[array=='nan'] = 'NaN'
            printArray = np.zeros(array.size, dtype=[('key_name', int), ("value", 'U6')])
            printArray['key_name'] = np.arange(1, array.size+1)
            printArray["value"] = array

            return printArray

        np.savetxt(pathToKeyOutput + "/max.csv", getPrintableArrayFromNumpyArray(maxValues),fmt="%d ,%s",header="Epoch, " + key)
        np.savetxt(pathToKeyOutput + "/min.csv", getPrintableArrayFromNumpyArray(minValues),fmt="%d ,%s",header="Epoch, " + key)
        np.savetxt(pathToKeyOutput + "/mean.csv", getPrintableArrayFromNumpyArray(mean),fmt="%d ,%s",header="Epoch, " + key)
        np.savetxt(pathToKeyOutput + "/median.csv", getPrintableArrayFromNumpyArray(median),fmt="%d ,%s",header="Epoch, " + key)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', required=True)    
    args = parser.parse_args()

    generateMixedCsv(args.base_folder)
