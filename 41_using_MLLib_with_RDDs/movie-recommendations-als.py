import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating

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


conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
sc = SparkContext(conf=conf)

sc.setCheckpointDir('checkpoint')

print("\nLoading movie names...")
nameDict = loadMovieNames()

data = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/ml-100k/u_mod.data")

# u_mod.data columns (u.data where a user 0 has been custom added):
#
# User ID, Movie ID, Rating, Timestamp
# 0	    50	5	881250949
# 0	    172	5	881250949
# 0     133	1	881250949
# 196	242	3	881250949
# 186	302	3	891717742
# 22	377	1	878887116
#
# Map to a MLLib Rating data structure: Rating(User ID, Movie ID, Rating)
ratings = data.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()

# Build the recommendation model using Alternating Least Squares
print("\nTraining recommendation model...")
rank = 10
numIterations = 20
model = ALS.train(ratings, rank, numIterations)

# Gather the userID sent to the python script as the parameter
userID = int(sys.argv[1])

# Print the user's existing movie ratings
print("\nRatings for user ID " + str(userID) + ":")
userRatings = ratings.filter(lambda l: l[0] == userID)
for rating in userRatings.collect():
    print(nameDict[int(rating[1])] + ": " + str(rating[2]))

# Print the algorithm's recommendations for the user
print("\nTop 10 recommendations:")
recommendations = model.recommendProducts(userID, 10)
for recommendation in recommendations:
    print(nameDict[int(recommendation[1])] + \
        " score " + str(recommendation[2]))

# cd ~/
# source ./activate_default_python_venv.sh
# cd /home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/movie-recommendations-als.py
# python movie-recommendations-als.py 0
#
# One example of results print-out (algorithm results change each time because no seed set)
#
# Ratings for user ID 0:
# Star Wars (1977): 5.0
# Empire Strikes Back, The (1980): 5.0
# Gone with the Wind (1939): 1.0
#
# Top 10 recommendations:
# Secret Agent, The (1996) score 6.574588427211987
# Little Princess, The (1939) score 6.5619321858616875
# Shall We Dance? (1937) score 6.521275755350617
# Low Down Dirty Shame, A (1994) score 6.458205282417204
# Love in the Afternoon (1957) score 6.06205588133405
# Cemetery Man (Dellamorte Dellamore) (1994) score 5.973602194521658
# Harlem (1993) score 5.950648942160058
# Fear of a Black Hat (1993) score 5.907738769580926
# Alphaville (1965) score 5.873394326983904
# Roommates (1995) score 5.865793104300861
