from pyspark.sql import SparkSession
from pyspark.sql import Row


# u.item columns:
#
# Movie ID|Movie Name|....
# 1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|...
# 2|GoldenEye (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?GoldenEye%20(1995)|0|1|1|0|0|0|0|0|0|...
def loadMovieNames():
    movieNames = {}
    with open(
        "/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/ml-100k/u.item",
        encoding='iso-8859-1'
    ) as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


# Create a SparkSession (the config bit is only for Windows!)
# spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("PopularMovies").getOrCreate()
spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

# Load up our movie ID -> name dictionary
nameDict = loadMovieNames()

# Get the raw data
lines = spark.sparkContext.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/ml-100k/u.data")

# u.data columns:
#
# User ID, Movie ID, Rating, Timestamp
# 196	242	3	881250949
# 186	302	3	891717742
# 22	377	1	878887116
#
# Convert it to a RDD of Row objects
movies = lines.map(lambda x: Row(movieID=int(x.split()[1])))

# Convert that to a DataFrame
movieDataset = spark.createDataFrame(movies)

# Some SQL-style magic to sort all movies by popularity in one line!
topMovieIDs = movieDataset.groupBy("movieID").count().orderBy("count", ascending=False).cache()

# Show the results at this point:

#|movieID|count|
#+-------+-----+
#|     50|  584|
#|    258|  509|
#|    100|  508|

topMovieIDs.show()

# Grab the top 10
top10 = topMovieIDs.take(10)

# Print the results
print("\n")
for result in top10:
    # Each row has movieID, count as above.
    print("%s: %d" % (nameDict[result[0]], result[1]))

# Stop the session
spark.stop()
