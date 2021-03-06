# u.data columns:
#
# User ID, Movie ID, Rating, Timestamp
# 196	242	3	881250949
# 186	302	3	891717742
# 22	377	1	878887116
# 244	51	2	880606923
# 166	346	1	886397596
# 298	474	4	884182806

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[4]").setAppName("PopularMovies")  # Trying out 4 threads on local
sc = SparkContext(conf=conf)

lines = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/ml-100k/u.data")

movies = lines.map(lambda x: (int(x.split()[1]), 1))
movieCounts = movies.reduceByKey(lambda x, y: x + y)

# flipped = movieCounts.map(lambda xy: (xy[1], xy[0]))
# sortedMovies = flipped.sortByKey()
sortedMovies = movieCounts.sortBy(lambda x: x[1])

results: list = sortedMovies.collect()

for result in results:
    print(result)
