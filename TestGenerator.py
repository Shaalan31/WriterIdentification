import xml.etree.ElementTree as ET
import glob
import cv2
from pathlib import Path
import os, errno
import numpy as np
import shutil
base = 'test/'


imageCount = np.zeros((700,1))
for filename in glob.glob('iAm/*.xml'):
    #temp = cv2.imread(filename)
    tree = ET.parse(filename)
    root = tree.getroot()
    id = root.attrib[ 'writer-id']
    imageCount[int(id)] += 1

    filename = filename.replace('xml', 'png')
    name = Path(filename).name
    print(name)
    try:
        os.makedirs(base+id)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    shutil.copyfile(filename,base+id+'/'+name)
    #cv2.imwrite(base+id+'/'+name,temp)

base = 'Samples/'
try:
    os.makedirs('TestCases')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

np.savetxt("foo.csv", imageCount, delimiter=",")

classNum = 0
print('generating cases')
for i in range(0,700):
    if imageCount[i] < 3:
        continue
    classNum +=1
    id = str(i)
    print(i)
    try:
        os.makedirs(base+'Class'+str(classNum))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    while len(id) < 3:
        id = '0'+id
    count = 0
    for filename in glob.glob('test/'+id+'/*.png'):
        #temp = cv2.imread(filename)
        name = Path(filename).name
        if count<2:
            #cv2.imwrite(base+'Class'+str(classNum)+'/'+name,temp)
            shutil.copyfile(filename, base+'Class'+str(classNum)+'/'+name)

        elif count == 2:
            #cv2.imwrite('TestCases/testing'+str(classNum)+'.png',temp)
            shutil.copyfile(filename, 'TestCases/testing'+str(classNum)+'.png')
        else:
            break

        count += 1