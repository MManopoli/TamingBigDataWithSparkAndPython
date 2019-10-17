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
lines = sc.textFile("file:///SparkCourse/fakefriends.csv")
# Apply the map, defined above as a Python function
rdd = lines.map(parseLine)
# Couple chained operations
# 1. mapValues - map values == numFriends to (numFriends, 1) for counting later --> returns an RDD
# 2. reduceByKey - aggregate everything for each age where two values (numFriends, 1) called x and y are
#                  combined according to the functions defined: result tuple value 0 = x[0] + y[0] and
#                  result tuple value 1 = x[1] + y[1]
totalsByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
averagesByAge = totalsByAge.mapValues(lambda x: x[0] / x[1])
results = averagesByAge.collect()
for result in results:
    print(result)
