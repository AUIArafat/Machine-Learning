def max_dot_product(vectors):
	max_result = 0
	j=0
	while j<len(vectors):
		for i in range(j+1, len(vectors)):
			result = sum(abs(x)*abs(y) for x,y in zip(vectors[j],vectors[i]))
			max_result = max(max_result, result)
		j += 1
	return max_result


vectors = [[-4, 2, 1], [1, 2, -1], [2, 0, -2]]
print(max_dot_product(vectors))