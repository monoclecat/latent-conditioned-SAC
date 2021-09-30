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

def joinCsvFiles(paths, output):
    files = list()
    for pathIndex, path in enumerate(paths):
        if os.path.isfile(path):
            with open(path) as csvtext:
                file = dict()
                for column in zip(*[line for line in csv.reader(csvtext)]):
                    if column[0] == "Epoch":
                        continue
                    file["Values" + str(pathIndex)] = column[1:]
                files.append(file)
    JoinCsvFiles = np.zeros((len(paths), len(files[0]["Values0"])))

    for index in range(len(files)):
        JoinCsvFiles[index] = files[index]["Values" + str(index)]

    mean = np.mean(JoinCsvFiles, axis=0)
    std = np.std(JoinCsvFiles, axis=0)
    std_pos = mean + 2 * std
    std_neg = mean - 2 * std

    os.makedirs(output, exist_ok=True)

    np.savetxt(output + "/mean.csv", getPrintableArrayFromNumpyArray(mean),fmt="%d ,%s",header="Epoch, Value") 
    np.savetxt(output + "/std.csv", getPrintableArrayFromNumpyArray(std),fmt="%d ,%s",header="Epoch, Value")
    np.savetxt(output + "/std_pos.csv", getPrintableArrayFromNumpyArray(std_pos),fmt="%d ,%s",header="Epoch, Value")
    np.savetxt(output + "/std_neg.csv", getPrintableArrayFromNumpyArray(std_neg),fmt="%d ,%s",header="Epoch, Value")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_path', "-fp", nargs="+", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    joinCsvFiles(args.files_path, args.output)