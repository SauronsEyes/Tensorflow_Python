import os
import csv

os.listdir(path='.')
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles
dirName = './simpsons_dataset/kaggle_simpson_testset/kaggle_simpson_testset';
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
number = ['0','1','2','3','4','5','6','7','8','9']
namelist = []
print(listOfFiles[0][65])
a=65
for i in range(len(listOfFiles)):
    name=""
    for j in range(65,len(listOfFiles[i])):
        if(listOfFiles[i][j] in number):
            break
        else:
            name = name+listOfFiles[i][j]
    namelist.append(name)
print(namelist)
for i in range(len(namelist)):
    namelist[i] = namelist[i][:-1]
    
print(namelist[0])

with open("filestructure2.csv", 'w',newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(zip(namelist, listOfFiles))   
