def readAllData(file_name):
	with open(file_name) as file:
		return [ tuple(datam.split('\t')) for datam in file.read().split('\n') if datam][1:]

def computeAverageForClasses(data):
	result = {}
	count = {}
	keys = []
	average = {}
	for datam in data:
		result[datam[1]] = result.get(datam[1],0)+float(datam[0])
		count[datam[1]] = count.get(datam[1],0) + 1

	keys = list(result.keys())

	for key in keys:
		average[key] = result[key] / count[key]

	return average

def misclassified(data):
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

