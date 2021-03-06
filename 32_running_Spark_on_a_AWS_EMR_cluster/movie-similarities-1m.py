import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

# Commands:
#
# Go to your AWS console home-page and find EMR under Analytics and start a 5 m4.xlarge cluster ($0.06/hr each)
# - Make sure it's a SPARK 2.4.4 cluster
# - Check to make sure the instance type you are selecting is available in your region using your ec2 instance page
#   - As of these notes m4.xlarge is a thing: as noted in the instructions above
#
# Make sure to configure the "master node" security groups to allow SSH connections from the correct IP, or you can't connect!
# e.g. Docker_Tutorial_Instance == 172.31.86.187/32
#
# Copy over this script
# scp -i ~/Ubuntu_Standard_1.pem /home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/movie-similarities-1m.py hadoop@ec2-3-92-210-197.compute-1.amazonaws.com:/home/hadoop
#
# Copy over movies.dat
# scp -i ~/Ubuntu_Standard_1.pem /home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/ml-1m/movies.dat hadoop@ec2-3-92-210-197.compute-1.amazonaws.com:/home/hadoop
#
# Connect to the environment - NOTE: THE IP ADDRESS IS RANDOMLY GENERATED EACH TIME YOU SPIN UP A CLUSTER
# ssh -i ~/Ubuntu_Standard_1.pem hadoop@ec2-3-92-210-197.compute-1.amazonaws.com
#
# Change to Python 3 in environment
# sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh
#
# Execute the script
# spark-submit --executor-memory 3G MovieSimilarities1M.py 260
#
# TERMINATE THE CLUSTER

# movies.dat columns:
#
# Movie ID::Movie Name::...
# 1::Toy Story (1995)::Animation|Children's|Comedy
# 2::Jumanji (1995)::Adventure|Children's|Fantasy
# 3::Grumpier Old Men (1995)::Comedy|Romance
def loadMovieNames():
    movieNames = {}
    with open(
            "movies.dat",
            encoding='ascii',
            errors='ignore'
    ) as f:
        for line in f:
            fields = line.split('::')
            movieNames[int(fields[0])] = fields[1]
    return movieNames


# Python 3 doesn't let you pass around unpacked tuples,
# so we explicitly extract the ratings now.

# Takes in a de-duplicated userRatings of the form userID => ((movieID, rating), (movieID, rating))
# Returns a new pairing of the form below
# The form below will have two movies rated by the user as the key and ratings of either movie as the value
def makePairs(userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return (movie1, movie2), (rating1, rating2)


# Takes in a userRating of the form userID => ((movieID, rating), (movieID, rating)) produced by the self-join
# Returns True for cases where the first/left movieID is less than the second/right movieID
# Used as the argument for .filter() to say keep only those flagged as True
def filterDuplicates(userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2


# Takes in input of the form ((movie1, movie2), Iterable<(rating1, rating2)>)
#  That is - groupByKey has joined the ratings for each user for each possible pair of movies as an iterable
# Output is the Cosine Similarity score and the number of pairs used to calculate that score
def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    # Math here is to divide the sum of each pair of ratings multiplied by each other or E (x*y)1 + (x*y)2...
    #  by the square root of the sum of x^s and the square root of the sum of y^2, because math
    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if denominator:
        score = (numerator / (float(denominator)))

    return score, numPairs


conf = SparkConf()
sc = SparkContext(conf=conf)

print("\nLoading movie names...")
nameDict = loadMovieNames()

data = sc.textFile("s3n://sundog-spark/ml-1m/ratings.dat")

# ratings.dat columns:
#
# User ID::Movie ID::Rating::Timestamp
# 1::1193::5::978300760
# 1::661::3::978302109
# 1::914::3::978301968
#
# Map ratings to key / value pairs: user ID => movie ID, rating
ratings = data.map(lambda l: l.split("::")).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

# Emit every movie rated together by the same user.
# Self-join to find every combination.
ratingsPartitioned = ratings.partitionBy(100)
joinedRatings = ratingsPartitioned.join(ratingsPartitioned)

# At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))

# Filter out duplicate pairs (by only keeping cases where the first/left movieID is less than the second/right movieID)
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs)

# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
moviePairRatings = moviePairs.groupByKey()

# From the spark 2.4.4 documentation...
#
#    groupByKey([numPartitions])
#
# When called on a dataset of (K, V) pairs, .groupByKey() returns a dataset of (K, Iterable<V>) pairs.
#
# Note: If you are grouping in order to perform an aggregation (such as a sum or average) over each key, using
#  reduceByKey or aggregateByKey will yield much better performance.
# Note: By default, the level of parallelism in the output depends on the number of partitions of the parent RDD.
#  You can pass an optional numPartitions argument to set a different number of tasks.

# We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
# Can now compute similarities.
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()

# Save the results if desired - Note: this outputs to 4 different text files
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile("movie-sims")

# Extract similarities for the movie we care about that are "good".
if len(sys.argv) > 1:

    scoreThreshold = 0.97
    coOccurenceThreshold = 50

    movieID = int(sys.argv[1])

    # Filter cached moviePairSimilarities for movies that are "good" as defined by
    #  our quality thresholds above
    filteredResults = moviePairSimilarities.filter(lambda pairSim: \
                                                       (pairSim[0][0] == movieID or pairSim[0][1] == movieID) \
                                                       and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

    # Sort by quality score and take the top 10.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending=False).take(10)

    print("Top 10 similar movies for " + nameDict[movieID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if similarMovieID == movieID:
            similarMovieID = pair[1]
        print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))

# Results:
#
# Top 10 similar movies for Star Wars: Episode IV - A New Hope (1977)
# Star Wars: Episode V - The Empire Strikes Back (1980)	score: 0.9897917106566659	strength: 2355
# Sanjuro (1962)	score: 0.9877157157535862	strength: 60
# Raiders of the Lost Ark (1981)	score: 0.9855548278565054	strength: 1972
# Star Wars: Episode VI - Return of the Jedi (1983)	score: 0.9841248359926177	strength: 2113
# Run Silent, Run Deep (1958)	score: 0.9791463389332324	strength: 145
# Laura (1944)	score: 0.9787290037239523	strength: 187
# Close Shave, A (1995)	score: 0.9782167620836959	strength: 436
# Wrong Trousers, The (1993)	score: 0.978051224484339	strength: 596
# Captain Horatio Hornblower (1951)	score: 0.9778921720044944	strength: 81
# Indiana Jones and the Last Crusade (1989)	score: 0.9774440028650038	strength: 1397