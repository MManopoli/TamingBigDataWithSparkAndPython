# Objective is to count the number of words in a book
from pyspark import SparkConf, SparkContext

# Set Master as "local" - local machine + single thread / process
# Set AppName as "WordCount"
conf = SparkConf().setMaster("local").setAppName("WordCount")
# Instantiate SparkContext object with conf above
sc = SparkContext(conf=conf)

input = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/Book.txt")
words = input.flatMap(lambda x: x.split())
wordCounts: dict = words.countByValue()

for word, count in wordCounts.items():
    cleanWord = word.encode(encoding='ascii', errors='ignore')
    if (cleanWord):
        print(cleanWord.decode() + " " + str(count))
