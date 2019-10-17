import re
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf=conf)

input = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/Book.txt")

def normalizeWords(text):
    return re.compile(r"\W+", re.UNICODE).split(text.lower())


words = input.flatMap(normalizeWords)
# 1. map the words to a key value pair (word, 1)
# 2. reduce by key == word by adding the value elements == 1 --> this will give you key == word, value == count
wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
# 1. swap the key and value so count is now the key
# 2. Sort the RDD by the key element (now count)
wordCountsSorted = wordCounts.map(lambda x: (x[1], x[0])).sortByKey()
results: list = wordCountsSorted.collect()

for result in results:
    count = str(result[0])
    word = result[1].encode('ascii', 'ignore')
    if word:
        print(word.decode() + ":\t\t" + count)
