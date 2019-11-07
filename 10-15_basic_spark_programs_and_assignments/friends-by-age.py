# The objective is to find the average number of friends for each person of a given age
from pyspark import SparkConf, SparkContext

# Set Master as "local" - local machine + single thread / process
# Set AppName as "FriendsByAge"
conf = SparkConf().setMaster("local").setAppName("FriendsByAge")
# Instantiate SparkContext object with conf above
sc = SparkContext(conf=conf)

# Define a Python function to use in mapping
def parseLine(line):
    fields = line.split(',')
    age = int(fields[2])
    numFriends = int(fields[3])
    return (age, numFriends)


# Import and parse the input data into an RDD of lines called lines
lines = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/fakefriends.csv")
# Apply the map, defined above as a Python function
rdd = lines.map(parseLine)
# Couple chained operations
# 1. mapValues - map values == numFriends to (numFriends, 1) for counting later --> returns an RDD
# 2. reduceByKey - aggregate everything for each age where two values (numFriends, 1) called x and y are
#                  combined according to the functions defined: result tuple value 0 = x[0] + y[0] and
#                  result tuple value 1 = x[1] + y[1] --> returns an RDD
totalsByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
# Finally, get the average number of friends for each age by dividing the summed number of friends by the
#  number of people that are a given age
# mapValues returns an RDD
averagesByAge = totalsByAge.mapValues(lambda x: x[0] / x[1])
# Final action - collect the results and return them in a python data structure
results: list = averagesByAge.collect()
# print the results in the resulting python list
for result in results:
    print(result)
