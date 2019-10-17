import re
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf=conf)

input = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/Book.txt")

def normalizeWords(text):
    # Break up text based on words using '\W+', coding is unicode, and make everything lowercase
    # FMI - unicode a-z, A-Z, 0-9 are the only allowed elements of a word when using \W
    return re.compile(r"\W+", re.UNICODE).split(text.lower())


words = input.flatMap(normalizeWords)
wordCounts: dict = words.countByValue()

for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if cleanWord:
        print(cleanWord.decode() + " " + str(count))
