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
dirName = './simpsons_dataset/simpsons_dataset';
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
namelist = [[]]
for i in range(len(listOfFiles)):
    name=""
    listOfFiles[i] = listOfFiles[i].replace("\\","/")
    for j in range(36,len(listOfFiles[i])):
        if(listOfFiles[i][j]=="/"):
            break
        else:
            name=name+listOfFiles[i][j]
    namelist.append(name)
print(listOfFiles[0])
print(namelist[20000])


with open("filestructure.csv", 'w',newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(zip(namelist, listOfFiles))
  
 
