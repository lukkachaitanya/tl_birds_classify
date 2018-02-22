import os, sys
import shutil
import random
from shutil import copyfile

#folder which contains the sub directories
source_dir = 'images/'
total=0
#list sub directories 
for root, dirs, files in os.walk(source_dir):
    
#iterate through them
    for i in dirs: 
        
        #create a new folder with the name of the iterated sub dir
        path = 'test/' + "%s/" % i
        os.makedirs(path)

        #take random sample, here 3 files per sub dir
        filenames = random.sample(os.listdir('images/' + "%s/" % i ), int(len(os.listdir('images/' + "%s/" % i ))*0.3))
        #total+= len(os.listdir('images/' + "%s/" % i ))
        #print total
        #print filenames
        #copy the files to the new destination
        for j in filenames:
            shutil.move('images/' + "%s/" % i  + j, path)