from random import randrange, shuffle


def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

arr = [64, 34, 25, 12, 22, 11, 90]


def isSU(num):
    end = pow(num,0.5)
    end = num**0.5
    print(end)
    for i in range(2,int(end+1)):
        if(num % i == 0):
            return 0
        else:
            continue
    return 1

print(isSU(37))




shuzu = ["asd","fewsge","fewui","bdf"]
#final 不能改变
yuanzu = (2,3,5,342,6,43,64,6,43)
#key-Value
zidian = {"a":2,"asd":2534}
#Set
jihe = {"asd","asdf3532","sadfe"}

print(jihe)
shuffle(shuzu)

print(shuzu)

shuzu.sort()

print(shuzu)
