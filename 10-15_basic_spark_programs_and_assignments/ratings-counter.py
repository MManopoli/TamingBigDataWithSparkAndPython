from pyspark import SparkConf, SparkContext  # The two most important packages to import
import collections  # Standard Python import so we can sort the results when we're done

# First, Set the master machine for the SparkConf object as the local machine.  Single thread, single process.  Simple.
# Set the application name for this job so it can be identified in the Spark UI
conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
# Then, create the SparkContext object (which is always stored as the name sc by convention) from the conf we created
sc = SparkContext(conf=conf)

# Next, create an RDD stored under the name "lines" from the text file specified
# Each element of the RDD is a line of text from the file, which looks like: "196 242 3 881250949"
lines = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/ml-100k/u.data")
# Then, store the data returned by the map transformation below under the name "ratings"
# Lambda below is splitting each element ("line") in lines by space characters (default) and selecting the 3rd element
# e.g. "196 242 3 881250949" -> "3"
# This operation is important - we are extracting (mapping) the data we care about.  Fundamental to Spark...
# Side note: map is not inplace -> Return only :)
ratings = lines.map(lambda x: x.split()[2])
# Finally, store the countByValue() of the ratings RDD as results -> This is our action
# RDD [3, 3, 1, 2, 1] --> [(3,2), (1,2), (2,1)]
# https://spark.apache.org/docs/2.1.0/api/python/pyspark.html
# countByValue returns the count of each unique value in this RDD as a dictionary of (value, count) pairs.
result: dict = ratings.countByValue()

# Python code - Create ordered dictionary and print them
sortedResults = collections.OrderedDict(sorted(result.items()))
for key, value in sortedResults.items():
    print("%s %i" % (key, value))
