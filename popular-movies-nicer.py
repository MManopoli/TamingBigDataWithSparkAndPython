# u.data columns:
#
# User ID, Movie ID, Rating, Timestamp
# 196	242	3	881250949
# 186	302	3	891717742
# 22	377	1	878887116
# 244	51	2	880606923
# 166	346	1	886397596
# 298	474	4	884182806

# u.item columns:
#
# Movie ID|Movie Name|
# 1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
# 2|GoldenEye (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?GoldenEye%20(1995)|0|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0

from pyspark import SparkConf, SparkContext

def loadMovieNames():
    movieNames = {}
    with open("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

conf = SparkConf().setMaster("local").setAppName("PopularMovies")
sc = SparkContext(conf=conf)

nameDict = sc.broadcast(loadMovieNames())

lines = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/ml-100k/u.data")
movies = lines.map(lambda x: (int(x.split()[1]), 1))
movieCounts = movies.reduceByKey(lambda x, y: x + y)

# flipped = movieCounts.map( lambda x : (x[1], x[0]))
# sortedMovies = flipped.sortByKey()
sortedMovies = movieCounts.sortBy(lambda x: x[1])

sortedMoviesWithNames = sortedMovies.map(lambda countMovie : (nameDict.value[countMovie[1]], countMovie[0]))

results = sortedMoviesWithNames.collect()

for result in results:
    print (result)
