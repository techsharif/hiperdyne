def readAllData(file_name):
	with open(file_name) as file:
		return [ tuple(datam.split('\t')) for datam in file.read().split('\n') if datam][1:]

def computeAverageForClasses(data):
	result = {}
	count = {}
	keys = []
	average = {}
	for datam in data:
		try:
			result[datam[1]] += float(datam[0])
			count[datam[1]] += 1
		except: 
			result[datam[1]] = float(datam[0])
			count[datam[1]] = 1
			keys += [datam[1]]

	for key in keys:
		average[key] = result[key] / count[key]

	return average

def countMisclassified(data):
	average = computeAverageForClasses(data)
	keys = list(average.keys())
	misclassified = []
	for datam in data:
		diff = [abs(float(datam[0])-average[key]) for key in keys]
		mn = min(diff)
		min_index = diff.index(mn)
		if keys[min_index] != datam[1]:
			misclassified += [datam]
	return misclassified


#read data
data = readAllData('data.txt')

# average
average = computeAverageForClasses(data)
print('Average:')
print(average)

# misclassified
misclasified = countMisclassified(data)
print('Misclassified:')
print(len(misclasified))

# file write
write_data = [ f'{m[0]}\t{m[1]}' for m in misclasified ]

f = open("Misclassified.txt", "w")
f.write('\n'.join(write_data))
f.close()


