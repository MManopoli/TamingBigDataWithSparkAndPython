# Objective is to count the number of words in a text file (Book.txt)
from pyspark import SparkConf, SparkContext

# Set Master as "local" - local machine + single thread / process
# Set AppName as "WordCount"
conf = SparkConf().setMaster("local").setAppName("WordCount")
# Instantiate SparkContext object with conf above
sc = SparkContext(conf=conf)

# Import and parse the input data into an RDD of lines called lines
bookLines = sc.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/Book.txt")
# flatMap is new for us as
# map transforms each element of an RDD into one new element
# flatMap can transform each element of an RDD into many new elements
# In this case, each split out result (text split by spaces) of each line is output as a new element in a new RDD
bookWords = bookLines.flatMap(lambda x: x.split())
# We know countByValue - count each element and return a dictionary where the key is the element counted and the value
#  is the count of that element
book_word_counts: dict = bookWords.countByValue()

for word, count in book_word_counts.items():
    cleanWord = word.encode(encoding='ascii', errors='ignore')
    if cleanWord:
        print(cleanWord.decode() + " " + str(count))
