from functions import readAllData, computeAverageForClasses, misclassified

# file name
FILE_NAME = 'data.txt'

#read data
data = readAllData(FILE_NAME)

# average
average = computeAverageForClasses(data)
print('Average:')
print(average)

# misclassified
misclasified = misclassified(data)
print('Misclassified:')
print(len(misclasified))

# file write
write_data = [ f'{m[0]}\t{m[1]}' for m in misclasified ]

f = open("Misclassified.txt", "w")
f.write('\n'.join(write_data))
f.close()


