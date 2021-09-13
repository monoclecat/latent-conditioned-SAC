import itertools
import os

contSteps = [-0.9, -0.45, 0.0, 0.45, 0.9]

def createMovementPictograms(runPath, nDisc, nCont, runType):
    testPolicyCommand = "python -m spinup.run test_policy {runPath} {skills} --renderImage -iF 10"

    skills = ""
    if nDisc > 0:
        skills = skills + "--disc-skill disc0 "
    if nCont > 0:
        skills = skills + "--cont-skill "
        for x in range(nCont):
            skills = skills + "cont" + str(x) + " "
    
    testPolicyCommand = testPolicyCommand.format(runPath=runPath, skills=skills)

    if nCont == 2 and nDisc == 0:
        permutions = list(itertools.product(contSteps, contSteps))
        for permut in permutions:
            deleteCommand = "rm -r {basePath}/images/"
            deleteCommand = deleteCommand.format(basePath=runPath)
            
            os.system(deleteCommand)

            command = testPolicyCommand.replace("cont0", str(permut[0]))
            command = command.replace("cont1", str(permut[1]))

            print(command)
            os.system(command)


            createMovementImageCommand = "python tools/create_movement_image.py -bP {basePath}/images/ -irs 0 -ire 100 -iF 10"
            createMovementImageCommand = createMovementImageCommand.format(basePath=runPath)

            os.system(createMovementImageCommand)

            copyCommand = "cp {source} {target}"

            source = "{basePath}/images/resultImage_epsiode0.png"
            source = source.format(basePath=runPath)

            target = "tools/MovementImages/{type}/disc{nDisc}_cont{nCont}/resultImage_{cont0}_{cont1}.png"
            target = target.format(type=runType, nDisc=nDisc, nCont=nCont, cont0=str(permut[0]), cont1=str(permut[1]))

            copyCommand = copyCommand.format(source=source, target=target)

            os.system(copyCommand)

    if nCont == 1 and nDisc == 3:
        for disc in range(nDisc):
            for cont in contSteps:
                deleteCommand = "rm -r {basePath}/images/"
                deleteCommand = deleteCommand.format(basePath=runPath)
                
                os.system(deleteCommand)

                command = testPolicyCommand.replace("disc0", str(disc+1))
                command = command.replace("cont0", str(cont))

                os.system(command)
                print(command)

                createMovementImageCommand = "python tools/create_movement_image.py -bP {basePath}/images/ -irs 0 -ire 100 -iF 10"
                createMovementImageCommand = createMovementImageCommand.format(basePath=runPath)

                os.system(createMovementImageCommand)

                copyCommand = "cp {source} {target}"

                source = "{basePath}/images/resultImage_epsiode0.png"
                source = source.format(basePath=runPath)

                target = "tools/MovementImages/{type}/disc{nDisc}_cont{nCont}/resultImage_{disc0}_{cont0}.png"
                target = target.format(type=runType, nDisc=nDisc, nCont=nCont, disc0=str(disc+1), cont0=str(cont))

                copyCommand = copyCommand.format(source=source, target=target)

                os.system(copyCommand)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runPath', "-rP", required=True)
    parser.add_argument('--nDisc', "-nD", type=int, required=True)
    parser.add_argument('--nCont', "-nC", type=int, required=True)
    parser.add_argument('--type', "-t", required=True)

    args = parser.parse_args()
    createMovementPictograms(args.runPath, args.nDisc, args.nCont, args.type)