from pyspark.sql import SparkSession
from pyspark.sql import Row

# Create a SparkSession (Note, the config section is only for Windows!)
# spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("SparkSQL").getOrCreate()
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()


def mapper(line):
    fields = line.split(',')
    return Row(
        ID=int(fields[0]),
        name=str(fields[1].encode("utf-8")),
        age=int(fields[2]),
        numFriends=int(fields[3])
    )


lines = spark.sparkContext.textFile("/home/mmanopoli/Udemy/TamingBigDataWithSparkAndPython/data/fakefriends.csv")
people = lines.map(mapper)

# Create an official Spark DataFrame from the RDD of Row objects
# Also, cache the Spark DataFrame because we're going to do more than one thing with it
schemaPeople = spark.createDataFrame(people).cache()
# Infer the schema, and register the DataFrame as a table we can query.
schemaPeople.createOrReplaceTempView("people")

# SQL can be run over DataFrames that have been registered as a table.
teenagers = spark.sql("SELECT * FROM people WHERE age >= 13 AND age <= 19")

# The results of SQL queries are RDDs and support all the normal RDD operations.
for teen in teenagers.collect():
    print(teen)

# We can also use functions instead of SQL queries:
schemaPeople.groupBy("age").count().orderBy("age").show()

spark.stop()
