# just some useful code to displays matrices nicer for looking at this
# https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
  
# Try it! 

#---------------------------------------------------------------------------------------------------------
