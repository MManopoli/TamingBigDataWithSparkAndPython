# Objective is to find the maximum temperature observed by various weather stations in the year 1800
from pyspark import SparkConf, SparkContext

# Set Master as "local" - local machine + single thread / process
# Set AppName as "MaxTemperatures"
conf = SparkConf().setMaster("local").setAppName("MaxTemperatures")
# Instantiate SparkContext object with conf above
sc = SparkContext(conf=conf)

# Sample data line: "ITE00100554,18000101,TMAX,-75,,,E,"
# Columns are "weather station id, date, entry type, temperature in tenths of a degree celsius, etc, etc, etc..."
# Note: There is also an entry for precipitation where the 4th column is precipitation rather than temperature
def parseLine(line):
    fields = line.split(',')
    stationID = fields[0]
    entryType = fields[2]
    temperature = float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0  # Convert temp to degrees Fahrenheit
    return (stationID, entryType, temperature)

# Import and parse the input data into an RDD of lines called lines
lines = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/1800.csv")
# Use the parseLine function defined above to parse the data into a usable RDD
parsedLines = lines.map(parseLine)
# The filter transformation takes a function that returns a boolean
# In this case, we want the transform to return an RDD where only entries with "TMAX" as the [1] item are kept
# This is because we are trying to find the maximum observed temperature by each weather station in 1800, and so
#  we only need to consider the maximum temperature observed on any given day (labeled "TMAX")
minTemps = parsedLines.filter(lambda x: "TMAX" in x[1])
# Strip out the entryType because it's always "TMAX" now
# This is now a key value pair, stationID = key, temperature = value
stationTemps = minTemps.map(lambda x: (x[0], x[2]))
# reduce by key = for each key reduce down using the lambda function
# So, the lambda function returns the minimum of two inputs, x and y, which defines the result when two elements
#  x and y are reduced/combined for a given key
minTemps = stationTemps.reduceByKey(lambda x, y: max(x, y))
# Collect action - collect the results into a Python list
results = minTemps.collect()

for result in results:
    print("{0}\t{1:.2f}F".format(result[0], result[1]))
