from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[8]").setAppName("PopularHero")  # Trying out 8 threads
sc = SparkContext(conf=conf)

# Marvel-Names.txt data format:
# first value: Marvel hero ID
# Second value: Marvel hero name
#
# Example:
# 2549 "HULK III/BRUCE BANNE"
def parseNames(line):
    fields = line.split('\"')
    # The encoding for fields[1] is weird - IntelliJ confirms it - explicitly encode this as utf8 here
    return int(fields[0]), fields[1].encode("utf8")


names = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/Marvel-Names.txt")
namesRdd = names.map(parseNames)

# Marvel-Graph.txt data format:
# First value = Marvel hero ID
# Remaining values = hero IDs the first hero ID has appeared with in comic books
#
# Note: The entries for a specific Hero ID can span multiple lines
#
# Example:
# 5983 1165 3836 4361 1282 716 4289 4646 6300 5084 2397 4454 1913 5861 5485
def countCoOccurences(line):
    elements = line.split()
    # Return the Hero ID and the number of fellow co-appearing hero IDs (length of line - 1)
    return int(elements[0]), (len(elements) - 1)


lines = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/Marvel-Graph.txt")
pairings = lines.map(countCoOccurences)

# Add up all co-appearing hero IDs for each hero ID
totalFriendsByCharacter = pairings.reduceByKey(lambda x, y: x + y)

# flipped = totalFriendsByCharacter.map(lambda xy : (xy[1], xy[0]))
# mostPopular = totalFriendsByCharacter.max()
mostPopular = totalFriendsByCharacter.max(lambda xy: xy[1])

# Use the lookup function to map / lookup the name
# Note: lookup(key) -> Python list
mostPopularName = namesRdd.lookup(key=mostPopular[0])[0]

# print(str(mostPopularName) + " is the most popular superhero, with " + \
#     str(mostPopular[0]) + " co-appearances.")
print("{0} is the most popular superhero, with {1} co-appearances.".format(mostPopularName, mostPopular[1]))
