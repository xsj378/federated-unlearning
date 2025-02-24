import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_excel("../output-0.3.xlsx")
    acc = data.iloc[:,7]
    n = []
    times = 0
    temp = 0
    startAcc = 0
    startAcc = acc[0]
    for i in range(1,len(acc)):
        endAcc = acc[i]
        print((startAcc-endAcc)/startAcc)
        if(abs(startAcc-endAcc)/startAcc>0.1):
            times += 1
            n.append(i - temp)
            temp = i
            startAcc = acc[i]
    print(n)
    print((1*3+5*3+4*2.5+7*2.3)/4)